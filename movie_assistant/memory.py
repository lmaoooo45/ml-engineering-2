from contextlib import closing, contextmanager
import json
import re
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone

from filelock import FileLock, Timeout
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def open_memory(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory.sqlite"
    try:
        with FileLock(str(path) + ".lock", timeout=0):
            with closing(sqlite3.connect(path, check_same_thread=False)) as connection:
                yield Memory(connection)
    except Timeout as error:
        raise RuntimeError("Память занята другим запуском") from error


class Memory:
    def __init__(self, connection):
        self.connection = connection
        self.checkpointer = SqliteSaver(connection)
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS movie_sessions (
                id TEXT PRIMARY KEY, profile TEXT NOT NULL, user_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS movie_preferences (
                profile TEXT PRIMARY KEY, settings TEXT NOT NULL
            );
        """)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(movie_preferences)")}
        if "updated_at" not in columns:
            with connection:
                connection.execute("ALTER TABLE movie_preferences ADD COLUMN updated_at TEXT")

    def session(self, profile, session_id, user_id):
        self.check_profile(profile)
        if session_id is not None:
            row = self.connection.execute(
                "SELECT user_id FROM movie_sessions WHERE id = ? AND profile = ?",
                (session_id, profile),
            ).fetchone()
            if row is None or (user_id is not None and user_id != row[0]):
                raise ValueError("Сессия не найдена для этого профиля и ID пользователя")
            return session_id, row[0]

        session_id = uuid4().hex
        user_id = 0 if user_id is None else user_id
        with self.connection:
            self.connection.execute("INSERT INTO movie_sessions VALUES (?, ?, ?)", (session_id, profile, user_id))
        return session_id, user_id

    @staticmethod
    def check_profile(profile):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", profile):
            raise ValueError("Имя профиля: от 1 до 64 букв латиницы, цифр, дефисов или подчёркиваний")

    def preferences(self, profile):
        self.check_profile(profile)
        row = self.connection.execute(
            "SELECT settings FROM movie_preferences WHERE profile = ?", (profile,),
        ).fetchone()
        return json.loads(row[0]) if row else {"genres": [], "exclude_titles": []}

    def save_preferences(self, profile, genres, exclude_titles):
        settings = self.preferences(profile)
        if genres is not None:
            settings["genres"] = list(dict.fromkeys(genres))
        if exclude_titles is not None:
            settings["exclude_titles"] = list(dict.fromkeys(exclude_titles))
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO movie_preferences (profile, settings, updated_at) VALUES (?, ?, ?)",
                                    (profile, json.dumps(settings, ensure_ascii=False),
                                     datetime.now(timezone.utc).isoformat()))
        return settings

    def preference_updated_at(self, profile):
        self.check_profile(profile)
        row = self.connection.execute("SELECT updated_at FROM movie_preferences WHERE profile = ?", (profile,)).fetchone()
        return row[0] if row else None

    def forget(self, profile):
        self.check_profile(profile)
        rows = self.connection.execute("SELECT id FROM movie_sessions WHERE profile = ?", (profile,)).fetchall()
        for (session_id,) in rows:
            self.checkpointer.delete_thread(session_id)
        with self.connection:
            self.connection.execute("DELETE FROM movie_sessions WHERE profile = ?", (profile,))
            self.connection.execute("DELETE FROM movie_preferences WHERE profile = ?", (profile,))
        return len(rows)
