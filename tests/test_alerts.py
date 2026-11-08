from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from movie_assistant.request_log import write_request
from movie_assistant import tracing


class AlertTests(unittest.TestCase):
    def test_streak_warns_once_and_recovers(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()) as console:
            root = Path(directory)
            for i, status in enumerate(("error", "error", "error", "error", "ok", "error", "error", "error")):
                write_request(root, {"run_id": str(i), "mode": "chat", "status": status, "seconds": 1})
            rows = [json.loads(line) for line in (root / "logs/alerts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["run_id"] for row in rows], ["2", "7"])
            self.assertEqual(console.getvalue().count("Три запроса"), 2)

    def test_timeout_and_slow_chat_but_not_probe(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            write_request(root, {"mode":"probe", "status":"ok", "seconds":100})
            self.assertFalse((root / "logs/alerts.jsonl").exists())
            write_request(root, {"mode":"chat", "status":"error", "seconds":181, "error_type":"TimeoutError"})
            rows = [json.loads(line) for line in (root / "logs/alerts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["kind"] for row in rows}, {"request_timeout", "slow_request"})

    def test_failed_export_warns_without_breaking_request(self):
        from opentelemetry.sdk.trace.export import SpanExportResult
        delegate = Mock()
        delegate.export.return_value = SpanExportResult.FAILURE
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()), \
             patch.dict("os.environ", {"MOVIE_TRACING":"1"}), \
             patch.object(tracing, "_provider", None), \
             patch("movie_assistant.settings.storage_root", return_value=Path(directory)), \
             patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", return_value=delegate):
            with tracing.traced_span("export-failure-check"):
                pass
            tracing._provider.shutdown()
            record = json.loads((Path(directory) / "logs/alerts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["kind"], "trace_export_failed")
