import unittest

from pydantic import ValidationError

from movie_assistant.contracts import Plan
from movie_assistant.planner import system_prompt


class PlanTests(unittest.TestCase):
    def test_rejects_wrong_types_unknown_fields_and_missing_title(self):
        valid = '{"action":"recommend","count":3,"genres":["Comedy"],"genre_mode":"and","exclude_titles":[]}'
        self.assertEqual(Plan.model_validate_json(valid).root.count, 3)
        for raw in (valid.replace('3', '"3"'), valid.replace('3', '11'),
                    valid[:-1] + ',"unknown":1}', '{"action":"lookup"}'):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                Plan.model_validate_json(raw)

    def test_coordinator_receives_both_skills(self):
        prompt = system_prompt("coordinator")
        self.assertIn("Слово «или»", prompt)
        self.assertIn("Длительность, актёры и сюжет", prompt)
