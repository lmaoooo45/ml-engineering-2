import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from movie_assistant.catalog import load_catalog, verify_files
from scripts.prepare_data import prepare


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / "u.genre").write_text("Comedy|0\nDrama|1\n\n", encoding="ascii")

    def write_movies(self, text):
        (self.folder / "u.item").write_text(text, encoding="latin-1")

    def test_accents_genres_and_unknown_year(self):
        self.write_movies("1|Café (1995)|date||url|1|0\n2|unknown||||0|1\n")
        movies = load_catalog(self.folder)
        self.assertEqual(movies[1], {"id": 1, "title": "Café (1995)", "year": 1995, "genres": ["Comedy"]})
        self.assertIsNone(movies[2]["year"])
        self.assertEqual(movies[2]["genres"], ["Drama"])

    def test_bad_columns_flags_and_duplicate_ids_rejected(self):
        for text in ["1|Movie\n", "1|Movie||||2|0\n", "1|Movie||||1|0\n1|Other||||0|1\n"]:
            with self.subTest(text=text):
                self.write_movies(text)
                with self.assertRaises(ValueError):
                    load_catalog(self.folder)

    def test_modified_file_is_detected(self):
        checksums = {}
        for name in ("u.item", "u.genre", "u.data", "README"):
            path = self.folder / name
            if not path.exists():
                path.write_bytes(b"original")
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.folder / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
        verify_files(self.folder)
        (self.folder / "u.item").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "u.item"):
            verify_files(self.folder)

    def test_wrong_archive_rejected_before_creating_target(self):
        archive = self.folder / "wrong.zip"
        archive.write_bytes(b"not the official archive")
        target = self.folder / "output"
        with self.assertRaisesRegex(ValueError, "сумма"):
            prepare(archive, target)
        self.assertFalse(target.exists())
