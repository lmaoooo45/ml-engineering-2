from pathlib import Path
import unittest
from unittest.mock import patch

from movie_assistant.contracts import Plan
from movie_assistant.validation import Verdict
from movie_assistant.workflow import build_graph


class WorkflowTests(unittest.TestCase):
    @patch("movie_assistant.workflow.run_tool")
    @patch("movie_assistant.workflow.chat")
    @patch("movie_assistant.workflow.build_plan")
    def test_specialist_cannot_change_lookup_title(self, plan, chat, run_tool):
        plan.return_value = Plan.model_validate({"action": "lookup", "title": "Toy Story"})
        chat.return_value = {"message": {"tool_calls": [
            {"function": {"name": "search_movies", "arguments": {"title": "Alien"}}},
        ]}}
        graph = build_graph(Path("."))
        state = graph.invoke({"question": "Какого года Toy Story?", "user_id": 1})
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["result"]["status"], "error")
        self.assertIn("недопустимый вызов", state["result"]["text"])
        run_tool.assert_not_called()

    @patch("movie_assistant.workflow.check_meaning")
    @patch("movie_assistant.workflow.chat")
    @patch("movie_assistant.workflow.build_plan")
    def test_unsupported_question_skips_tool_selection(self, plan, chat, check):
        plan.return_value = Plan.model_validate({"action": "unsupported", "message": "Нет данных об актёрах."})
        check.return_value = Verdict(accepted=True, reason="Каталог не содержит актёров")
        state = build_graph(Path(".")).invoke({"question": "Кто снялся в Toy Story?", "user_id": 1})
        self.assertEqual(state["result"]["status"], "unsupported")
        chat.assert_not_called()

    @patch("movie_assistant.workflow.check_meaning")
    @patch("movie_assistant.workflow.build_plan")
    def test_rejection_feedback_reaches_coordinator_once(self, plan, check):
        plan.side_effect = [
            Plan.model_validate({"action": "unsupported", "message": "Нет года"}),
            Plan.model_validate({"action": "clarify", "message": "Какой фильм?"}),
        ]
        check.side_effect = [
            Verdict(accepted=False, reason="Уточни название"),
            Verdict(accepted=True, reason="Нужно уточнение"),
        ]
        state = build_graph(Path(".")).invoke({"question": "Какого года фильм?", "user_id": 1})
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["result"]["status"], "clarify")
        self.assertEqual(plan.call_args.kwargs["feedback"], "Уточни название")
        self.assertTrue(state["validation"]["accepted"])
