import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from movie_assistant.tracing import traced_span, annotate_request


class TracingTests(unittest.TestCase):
    def test_parent_child_and_error_without_private_message(self):
        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with patch("movie_assistant.tracing.get_tracer", return_value=provider.get_tracer("test")):
            with self.assertRaises(RuntimeError):
                with traced_span("gateway.chat") as parent:
                    annotate_request({"run_id":"example", "status":"error"})
                    with traced_span("ollama.api.chat"):
                        raise RuntimeError("private question")
        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].parent.span_id, parent.get_span_context().span_id)
        self.assertEqual(spans[1].attributes["request.run_id"], "example")
        self.assertEqual(spans[0].status.status_code.name, "ERROR")
        self.assertNotIn("private question", str([(s.attributes, s.events, s.status.description) for s in spans]))
        provider.shutdown()
