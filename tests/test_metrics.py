import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from movie_assistant.api import create_app
from movie_assistant.metrics import read_summary
from movie_assistant.request_log import write_request


class MetricsTests(unittest.TestCase):
    def test_last_resource_measurement_exposes_missing_gpu_as_missing(self):
        from movie_assistant.metrics import prometheus_text
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_request(root, {"mode":"chat", "status":"ok", "seconds":1, "source":"synthetic",
                                 "worker_peak_rss_bytes":1024, "gpu_status":"unavailable", "gpu_sample_count":0})
            summary = read_summary(root)
            self.assertEqual(summary["latest_resources"]["worker_peak_rss_bytes"], 1024)
            self.assertIsNone(summary["latest_resources"]["gpu_utilization_observed_peak_percent"])
            metrics = prometheus_text(summary)
            self.assertIn('movie_gateway_last_worker_peak_rss_bytes{source="synthetic"} 1024', metrics)
            self.assertNotIn("# TYPE movie_gateway_last_gpu_utilization_observed_peak_percent", metrics)

    def test_synthetic_calls_do_not_change_user_mean(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_request(root, {"mode":"chat", "status":"ok", "seconds":2})
            write_request(root, {"mode":"chat", "status":"ok", "source":"synthetic", "seconds":20})
            summary = read_summary(root)
            self.assertEqual(summary["chat_count"], 1)
            self.assertEqual(summary["synthetic_chat_count"], 1)
            self.assertEqual(summary["chat_seconds_mean"], 2)

    def test_empty_journal_has_no_invented_mean(self):
        with TemporaryDirectory() as directory:
            summary = read_summary(Path(directory))
            self.assertEqual(summary["chat_count"], 0)
            self.assertIsNone(summary["chat_seconds_mean"])

    def test_api_aggregates_only_chat_time_and_reports_bad_lines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, status, seconds in [("chat", "ok", 2), ("chat", "error", 4), ("probe", "ok", 100)]:
                write_request(root, {"mode": mode, "status": status, "seconds": seconds})
            with (root / "logs/requests.jsonl").open("a", encoding="utf-8") as target:
                target.write('broken\n' + json.dumps({"mode":"chat","status":"ok","seconds":-1}) + '\n')
            with patch("movie_assistant.api.storage_root", return_value=root), TestClient(create_app()) as client:
                response = client.get("/monitor")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["chat_seconds_mean"], 3)
                self.assertEqual(response.json()["skipped_lines"], 2)
                metrics = client.get("/metrics")
                self.assertIn('movie_gateway_requests_total{mode="chat",status="error",source="user"} 1', metrics.text)
                self.assertIn("movie_gateway_chat_seconds_sum 6.0", metrics.text)
                self.assertIn("text/plain", metrics.headers["content-type"])
