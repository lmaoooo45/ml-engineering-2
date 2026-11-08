import json
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from movie_assistant.bridge import emit_tool_event
from movie_assistant.tool_events import validate_tool_event
from movie_assistant.tracing import record_tool_event, traced_span


class ToolEventTests(unittest.TestCase):
    def test_worker_message_validated_and_saved_as_trace_event(self):
        with patch.dict("os.environ", {"MOVIE_OLLAMA_BRIDGE":"stdio"}), redirect_stdout(StringIO()) as output:
            emit_tool_event("search_movies", .01, "ok", 1, 1)
        event = validate_tool_event(json.loads(output.getvalue()))
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with patch("movie_assistant.tracing.get_tracer", return_value=provider.get_tracer("test")):
            with traced_span("gateway.chat"):
                record_tool_event(event)
        saved = exporter.get_finished_spans()[0].events[0]
        self.assertEqual(saved.name, "tool.search_movies")
        self.assertEqual(saved.attributes["movie_count"], 1)
        self.assertEqual(saved.attributes["seconds"], .01)
        provider.shutdown()

    def test_unknown_fields_names_and_invalid_measurements_rejected(self):
        valid = {"type":"tool_event", "tool":"search_movies", "seconds":.01,
                 "status":"ok", "movie_count":1, "attempt":1}
        for changed in ({"tool":"run_shell"}, {"question":"private"}, {"seconds":float("nan")},
                        {"seconds":-1}, {"movie_count":True}, {"attempt":3}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_tool_event({**valid, **changed})
