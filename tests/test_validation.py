from pathlib import Path
import unittest
from unittest.mock import patch

from movie_assistant.contracts import Plan
from movie_assistant.validation import check_catalog_result


class CatalogValidationTests(unittest.TestCase):
    @patch("movie_assistant.validation.load_catalog")
    def test_rejects_film_missing_from_catalog(self, catalog):
        catalog.return_value = {}
        plan = Plan.model_validate({"action": "lookup", "title": "Toy Story"}).root
        result = {"status": "ok", "movies": [{"id": 1, "title": "Toy Story", "year": 1995}]}
        with self.assertRaisesRegex(ValueError, "неизвестный ID"):
            check_catalog_result(Path("."), plan, result, 1)
