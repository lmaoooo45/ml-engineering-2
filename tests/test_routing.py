from pathlib import Path
import unittest
from unittest.mock import patch

from movie_assistant.contracts import Plan
from movie_assistant.ollama_client import MODELS
from movie_assistant.routing import choose_route, decision
from movie_assistant.validation import Verdict
from movie_assistant.workflow import build_graph


FAST_RESPONSE = {"answers": {"route": {
    "type": "choice", "choice": "fast", "probabilities": {"fast": 0.9, "standard": 0.1},
}}}


class RoutingTests(unittest.TestCase):
    @patch("movie_assistant.routing.simple_year", return_value=False)
    @patch("movie_assistant.routing.request_json", return_value=FAST_RESPONSE)
    def test_code_gate_blocks_wrong_fast_decision(self, request, eligible):
        route = choose_route("Подбери фильмы", [], Path("."))
        self.assertEqual(route["model"], MODELS["gemma"])

    @patch("movie_assistant.routing.request_json", side_effect=RuntimeError("Offline"))
    def test_decision_failure_keeps_standard_model(self, request):
        route = choose_route("Какого года фильм?", [], Path("."))
        self.assertEqual(route["model"], MODELS["gemma"])
        self.assertEqual(route["reason"], "decision_unavailable")

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            decision({"answers": {"route": {
                "type": "choice", "choice": "fast", "probabilities": {"fast": True, "standard": 0},
            }}})

    @patch("movie_assistant.workflow.check_meaning")
    @patch("movie_assistant.workflow.build_plan")
    def test_second_attempt_switches_fast_model_to_gemma(self, plan, validate):
        plan.return_value = Plan.model_validate({"action": "clarify", "message": "Уточни фильм"})
        validate.side_effect = [
            Verdict(accepted=False, reason="Нужно другое уточнение"),
            Verdict(accepted=True, reason="Уточнение принято"),
        ]
        state = build_graph(Path("."), model=MODELS["ministral"]).invoke({
            "question": "Вопрос", "user_id": 0,
        })
        self.assertEqual(plan.call_args_list[0].args[1], MODELS["ministral"])
        self.assertEqual(plan.call_args_list[1].args[1], MODELS["gemma"])
        self.assertTrue(state["routing"]["fallback"])
