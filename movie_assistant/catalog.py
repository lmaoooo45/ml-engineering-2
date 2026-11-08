import csv
import hashlib
import json
from pathlib import Path
import re


def verify_files(directory: Path) -> None:
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    for name in ("u.item", "u.genre", "u.data", "README"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != checksums[name]:
            raise ValueError(f"Файл {name} изменён после подготовки")


def load_catalog(directory: Path) -> dict[int, dict]:
    genres = {}
    for line in (directory / "u.genre").read_text(encoding="ascii").splitlines():
        if line.strip():
            name, index = line.split("|")
            genres[int(index)] = name
    if not genres or set(genres) != set(range(len(genres))):
        raise ValueError("Неверная нумерация жанров")

    movies = {}
    with (directory / "u.item").open(encoding="latin-1", newline="") as source:
        for row in csv.reader(source, delimiter="|"):
            if len(row) != 5 + len(genres) or any(flag not in {"0", "1"} for flag in row[5:]):
                raise ValueError("Неверный формат строки фильма")
            movie_id = int(row[0])
            if movie_id <= 0 or movie_id in movies:
                raise ValueError("Неверный или повторный ID фильма")
            year = re.search(r"\((\d{4})\)$", row[1])
            movies[movie_id] = {
                "id": movie_id,
                "title": row[1],
                "year": int(year.group(1)) if year else None,
                "genres": [genres[i] for i, flag in enumerate(row[5:]) if flag == "1"],
            }
    return movies
