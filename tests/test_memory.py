from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from movie_assistant.contracts import Plan
from movie_assistant.memory import open_memory
from movie_assistant.validation import Verdict
from movie_assistant.workflow import run_session


class MemoryTests(unittest.TestCase):
    def setUp(self):
        router = patch("movie_assistant.workflow.choose_route", return_value={
            "model": "gemma4-e2b-local:q4_k_m", "reason": "test", "fallback": False,
        })
        router.start()
        self.addCleanup(router.stop)

    @patch("movie_assistant.workflow.check_meaning")
    @patch("movie_assistant.workflow.build_plan")
    def test_history_survives_reopen_and_stays_with_its_profile(self, plan, check):
        plan.return_value = Plan.model_validate({"action": "clarify", "message": "Какой фильм?"})
        check.return_value = Verdict(accepted=True, reason="Нужно уточнение")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_id, _ = run_session(root, "Первый вопрос", profile="test", user_id=1)
            _, state = run_session(root, "Второй вопрос", profile="test", session_id=session_id)
            self.assertEqual(plan.call_args.kwargs["history"][0]["content"], "Первый вопрос")
            self.assertEqual(state["attempts"], 1)
            self.assertEqual(state["user_id"], 1)
            with self.assertRaisesRegex(ValueError, "Сессия не найдена"):
                run_session(root, "Другой вопрос", profile="other", session_id=session_id)
            with open_memory(root) as memory:
                self.assertEqual(memory.forget("test"), 1)
                config = {"configurable": {"thread_id": session_id}}
                self.assertIsNone(memory.checkpointer.get(config))

    def test_preferences_survive_reopen_and_can_be_removed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with open_memory(root) as memory:
                memory.save_preferences("test", ["Comedy"], ["Toy Story"])
            with open_memory(root) as memory:
                self.assertEqual(memory.preferences("test"), {"genres": ["Comedy"], "exclude_titles": ["Toy Story"]})
                self.assertEqual(memory.preferences("other")["genres"], [])
                memory.forget("test")
                self.assertEqual(memory.preferences("test")["exclude_titles"], [])
