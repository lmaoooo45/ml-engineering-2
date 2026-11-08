import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from filelock import FileLock

from movie_assistant.gateway import isolated_request


class RequestLogTests(unittest.TestCase):
    def setUp(self):
        gpu = patch("movie_assistant.resources.read_gpu", side_effect=OSError("test without GPU"))
        gpu.start()
        self.addCleanup(gpu.stop)

    def test_synthetic_call_skips_busy_worker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("movie_assistant.gateway.storage_root", return_value=root), \
                 patch("movie_assistant.gateway.run_worker") as worker, \
                 FileLock(str(root / "gateway.lock")):
                result = isolated_request("chat", source="synthetic")
                self.assertEqual(result["status"], "skipped_busy")
                worker.assert_not_called()
            record = json.loads((root / "logs/requests.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["source"], "synthetic")
            self.assertEqual(record["status"], "busy")

    def test_unconfirmed_answer_without_verdict_does_not_crash_logging(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("movie_assistant.gateway.storage_root", return_value=root), \
                 patch("movie_assistant.gateway.sandbox_settings", return_value={}), \
                 patch("movie_assistant.gateway.run_worker", return_value={"status":"error", "movies":[],"validation":None,"routing":None}):
                result=isolated_request("chat")
            self.assertEqual(result["status"],"error")
            row=json.loads((root/"logs/requests.jsonl").read_text(encoding="utf-8"))
            self.assertIsNone(row["validator_model"])
            self.assertEqual(row["status"],"error")

    def test_success_and_failure_saved_without_question(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("movie_assistant.gateway.storage_root", return_value=root), \
                 patch("movie_assistant.gateway.sandbox_settings", return_value={}), \
                 patch("movie_assistant.gateway.run_worker") as worker:
                worker.return_value = {"status": "ok", "movies": []}
                result = isolated_request("chat", {"question": "private question"})
                worker.side_effect = RuntimeError("private details")
                with self.assertRaises(RuntimeError):
                    isolated_request("chat", {"question": "another question"})
            content = (root / "logs/requests.jsonl").read_text(encoding="utf-8")
            rows = [json.loads(line) for line in content.splitlines()]
            self.assertEqual([row["status"] for row in rows], ["ok", "error"])
            self.assertEqual(rows[0]["run_id"], result["run_id"])
            self.assertNotEqual(rows[0]["run_id"], rows[1]["run_id"])
            self.assertEqual(rows[1]["error_type"], "RuntimeError")
            self.assertNotIn("private", content)
            self.assertNotIn("another question", content)
