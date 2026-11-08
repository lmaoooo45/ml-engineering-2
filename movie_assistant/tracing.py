from contextlib import contextmanager
import os
import threading

_provider = None
_lock = threading.Lock()


def get_tracer():
    global _provider
    from opentelemetry import trace
    if os.environ.get("MOVIE_TRACING") != "1":
        return trace.NoOpTracer()
    with _lock:
        if _provider is None:
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from movie_assistant.alerts import emit_alert
            from movie_assistant.settings import storage_root
            import requests

            class ReportingExporter(SpanExporter):
                def __init__(self, delegate):
                    self.delegate = delegate

                def export(self, spans):
                    try:
                        result = self.delegate.export(spans)
                    except Exception:
                        result = SpanExportResult.FAILURE
                    if result == SpanExportResult.FAILURE:
                        emit_alert(storage_root(), "trace_export_failed",
                                   "Не удалось отправить трассу в локальный Phoenix")
                    return result

                def shutdown(self):
                    self.delegate.shutdown()

            session = requests.Session()
            session.trust_env = False
            _provider = TracerProvider(resource=Resource.create({
                "service.name": "movie-assistant-gateway",
                "openinference.project.name": "movie-assistant",
            }))
            _provider.add_span_processor(SimpleSpanProcessor(ReportingExporter(OTLPSpanExporter(
                endpoint="http://127.0.0.1:6006/v1/traces", timeout=2, session=session,
            ))))
    return _provider.get_tracer("movie-assistant")


@contextmanager
def traced_span(name, attributes=None):
    from opentelemetry.trace import Status, StatusCode
    with get_tracer().start_as_current_span(name, attributes=attributes,
                                           record_exception=False,
                                           set_status_on_exception=False) as span:
        try:
            yield span
        except Exception as error:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            raise


def annotate_request(record):
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    span = trace.get_current_span()
    for key, value in record.items():
        if value is not None:
            span.set_attribute("request." + key, value)
    if record["status"] in ("error", "busy"):
        span.set_status(Status(StatusCode.ERROR, record["status"]))


def record_tool_event(event):
    from opentelemetry import trace
    trace.get_current_span().add_event("tool." + event["tool"], attributes=event)
