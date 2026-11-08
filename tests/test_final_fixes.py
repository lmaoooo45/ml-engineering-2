import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from movie_assistant.budget import RequestBudget
from movie_assistant.memory import open_memory
from movie_assistant.movie_tools import describe_movie


class FinalFixTests(unittest.TestCase):
    def test_common_deadline_and_call_limit(self):
        with patch("movie_assistant.budget.monotonic", return_value=0):
            budget = RequestBudget()
        with patch("movie_assistant.budget.monotonic", return_value=179):
            self.assertEqual(budget.remaining(), 1)
            for _ in range(12):
                budget.before_call("/api/chat")
            with self.assertRaisesRegex(RuntimeError, "предел"):
                budget.before_call("/api/chat")
        with patch("movie_assistant.budget.monotonic", return_value=180):
            with self.assertRaises(TimeoutError):
                budget.before_call("/v1/systemone")

    def test_old_preferences_migrate_without_inventing_date(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with sqlite3.connect(root / "memory.sqlite") as connection:
                connection.execute("CREATE TABLE movie_preferences(profile TEXT PRIMARY KEY,settings TEXT NOT NULL)")
                connection.execute("INSERT INTO movie_preferences VALUES (?, ?)",
                                   ("old", json.dumps({"genres":["Comedy"],"exclude_titles":[]})))
            connection.close()
            with open_memory(root) as memory:
                self.assertEqual(memory.preferences("old")["genres"], ["Comedy"])
                self.assertIsNone(memory.preference_updated_at("old"))
                memory.save_preferences("old", ["Drama"], None)
                self.assertIsNotNone(memory.preference_updated_at("old"))

    def test_lookup_text_contains_year_and_genres_from_catalog(self):
        text = describe_movie({"title":"Toy Story (1995)","year":1995,"genres":["Animation","Comedy"]})
        self.assertIn("1995", text)
        self.assertIn("Animation, Comedy", text)
