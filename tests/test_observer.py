from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from movie_assistant.observer import sample_once, check_model


class ObserverTests(unittest.TestCase):
    def test_model_check_validates_fact_and_does_not_alert_when_busy(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()), \
             patch("movie_assistant.observer.isolated_request") as transport:
            root = Path(directory)
            transport.return_value = {"status":"ok", "run_id":"one", "movies":[{"id":1,"year":1995}]}
            self.assertEqual(check_model(root)["status"], "passed")
            self.assertEqual(transport.call_args.kwargs["source"], "synthetic")
            transport.return_value = {"status":"skipped_busy"}
            self.assertEqual(check_model(root)["status"], "skipped_busy")
            self.assertFalse((root / "logs/alerts.jsonl").exists())
            transport.return_value = {"status":"ok", "movies":[{"id":1,"year":2000}]}
            self.assertEqual(check_model(root)["status"], "failed")
            self.assertIn("model_check_failed", (root / "logs/alerts.jsonl").read_text(encoding="utf-8"))

    def test_outage_alerts_on_transition_and_samples_persist(self):
        def unavailable(url):
            raise OSError("service offline")

        def healthy(url):
            return json.dumps({"scope":"saved_gateway_requests"}) if url.endswith("/monitor") else "OK"

        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            with patch("movie_assistant.observer.fetch", side_effect=unavailable):
                first = sample_once(root)
                second = sample_once(root, first["services"])
            with patch("movie_assistant.observer.fetch", side_effect=healthy):
                recovered = sample_once(root, second["services"])
            with patch("movie_assistant.observer.fetch", side_effect=unavailable):
                sample_once(root, recovered["services"])
            alerts = (root / "logs/alerts.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(alerts), 8)
            self.assertEqual(len((root / "logs/observer.jsonl").read_text(encoding="utf-8").splitlines()), 4)
            self.assertTrue(all(recovered["services"].values()))
            self.assertFalse(any(first["services"].values()))
            self.assertIsNone(first["summary"])

    def test_invalid_summary_is_not_reported_as_healthy(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()), \
             patch("movie_assistant.observer.fetch", return_value='{}'):
            sample = sample_once(Path(directory))
            self.assertTrue(sample["services"]["api"])
            self.assertFalse(sample["services"]["monitor"])
            self.assertIsNone(sample["summary"])
