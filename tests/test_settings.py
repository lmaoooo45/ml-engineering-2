import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from movie_assistant.settings import PROJECT_ROOT, storage_root


class StorageSettingsTests(unittest.TestCase):
    def test_external_absolute_path_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MOVIE_STORAGE_DIR": directory}):
                self.assertEqual(storage_root(), Path(directory).resolve())

    def test_relative_path_is_rejected(self):
        with patch.dict(os.environ, {"MOVIE_STORAGE_DIR": "data"}):
            with self.assertRaisesRegex(ValueError, "абсолютный"):
                storage_root()

    def test_source_root_and_nested_storage_are_rejected(self):
        for path in [PROJECT_ROOT, PROJECT_ROOT / "data"]:
            with self.subTest(path=path):
                with patch.dict(os.environ, {"MOVIE_STORAGE_DIR": str(path)}):
                    with self.assertRaisesRegex(ValueError, "вне папки"):
                        storage_root()


if __name__ == "__main__":
    unittest.main()
