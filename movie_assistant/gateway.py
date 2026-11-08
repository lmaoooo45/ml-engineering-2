import json
import queue
import subprocess
import threading
from time import monotonic
from uuid import uuid4

from filelock import FileLock, Timeout

from deployment.snapshot import make_snapshot
from movie_assistant.ollama_client import MODELS, request_json
from movie_assistant.routing import DECISION_MODEL, ROUTE_QUESTION
from movie_assistant.request_log import write_request
from movie_assistant.tracing import traced_span, annotate_request, record_tool_event
from movie_assistant.tool_events import validate_tool_event
from movie_assistant.settings import PROJECT_ROOT, storage_root
from movie_assistant.resources import GpuSampler, validate_worker_resources
from movie_assistant.budget import RequestBudget

MAX_BYTES = 1024 * 1024
MODELS_ALLOWED = set(MODELS.values()) | {DECISION_MODEL}


def allowed_request(endpoint, payload):
    if endpoint in {"/api/tags", "/api/version"}:
        return payload is None
    if (not isinstance(payload, dict) or type(payload.get("model")) is not str
            or payload["model"] not in MODELS_ALLOWED):
        return False
    if endpoint == "/v1/systemone":
        return (set(payload) == {"model", "state", "questions", "keep_alive"}
                and payload["model"] == DECISION_MODEL and payload["keep_alive"] == 0
                and payload["questions"] == {"route": ROUTE_QUESTION})
    if endpoint != "/api/chat":
        return False
    options = payload.get("options", {})
    return (
        set(payload) <= {"model", "messages", "stream", "keep_alive", "options", "think", "format", "tools"}
        and payload["model"] != DECISION_MODEL and payload.get("stream") is False
        and payload.get("keep_alive") == 0 and isinstance(options, dict)
        and set(options) <= {"temperature", "seed", "num_ctx", "num_predict"}
        and type(options.get("num_ctx")) is int and 0 < options["num_ctx"] <= 8192
        and type(options.get("num_predict")) is int and 0 < options["num_predict"] <= 4096
    )


def sandbox_settings():
    settings = json.loads((PROJECT_ROOT / "runtime-sandbox.json").read_text(encoding="utf-8"))
    stage = settings["stage"]
    if not stage.startswith("/home/") or "/pi-block2/sandbox/stage-" not in stage:
        raise ValueError("Недопустимый путь песочницы")
    _, fingerprint = make_snapshot(storage_root())
    if fingerprint != settings["snapshot_sha256"]:
        raise ValueError("Исходники или данные изменились. Выполни setup_isolation")
    return settings


def run_worker(settings, task):
    stage = settings["stage"]
    command = ["wsl.exe", "-d", settings["distro"], "--exec", "python3",
               stage + "/app/deployment/linux_launch.py", stage]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)
    messages = queue.Queue()
    errors = []
    tool_events = []

    def read_stdout():
        while True:
            line = process.stdout.readline(MAX_BYTES + 1)
            messages.put(line)
            if not line or len(line.encode()) > MAX_BYTES:
                break

    def read_stderr():
        for line in process.stderr:
            if len(errors) < 50:
                errors.append(line)

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    budget = RequestBudget()
    usage = {"llm_calls": 0, "decision_calls": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "calls_with_token_counts": 0}
    try:
        process.stdin.write(json.dumps(task, ensure_ascii=False) + "\n")
        process.stdin.flush()
        while True:
            remaining = budget.remaining()
            try:
                line = messages.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError("Песочница не ответила вовремя") from error
            if not line:
                raise RuntimeError("Worker завершился без результата: " + "".join(errors)[-1500:])
            if len(line.encode()) > MAX_BYTES:
                raise ValueError("Слишком большое сообщение worker")
            message = json.loads(line)
            if message["type"] == "tool_event":
                if task["mode"] != "chat" or len(tool_events) >= 12:
                    raise ValueError("Слишком много событий или неверный режим")
                event = validate_tool_event(message)
                tool_events.append(event)
                record_tool_event(event)
                continue
            if message["type"] == "result":
                try:
                    process.wait(timeout=min(10, budget.remaining()))
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError("Worker не завершился в пределах бюджета") from error
                if process.returncode:
                    raise RuntimeError("Worker завершился с ошибкой")
                result = message["result"]
                if task["mode"] in ("chat", "judge"):
                    result["tool_events"] = tool_events
                    result["worker_resources"] = validate_worker_resources(result["worker_resources"])
                    result["usage"] = usage
                return result
            if message["type"] == "error":
                if message["error_type"] == "ValueError":
                    raise ValueError(message["message"])
                if message["error_type"] == "TimeoutError":
                    raise TimeoutError(message["message"])
                raise RuntimeError(message["message"])
            if message["type"] != "rpc":
                raise ValueError("Неизвестное сообщение worker")
            try:
                endpoint, payload = message["endpoint"], message["payload"]
                if not allowed_request(endpoint, payload):
                    raise ValueError("Операция запрещена")
                budget.before_call(endpoint)
                if endpoint == "/api/chat":
                    usage["llm_calls"] += 1
                elif endpoint == "/v1/systemone":
                    usage["decision_calls"] += 1
                with traced_span("ollama" + endpoint.replace("/", "."), {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": payload.get("model", "") if payload else "",
                }) as span:
                    response = request_json(endpoint, payload, timeout=min(60, budget.remaining()))
                    known = 0
                    for field, attribute in (("prompt_eval_count", "llm.token_count.prompt"),
                                             ("eval_count", "llm.token_count.completion")):
                        count = response.get(field)
                        if type(count) is int and count >= 0:
                            span.set_attribute(attribute, count)
                            usage["prompt_tokens" if field == "prompt_eval_count" else "completion_tokens"] += count
                            known += 1
                    if known == 2:
                        usage["calls_with_token_counts"] += 1
                reply = {"ok": True, "result": response}
            except (RuntimeError, ValueError) as error:
                reply = {"ok": False, "error": type(error).__name__}
            encoded = json.dumps(reply)
            if len(encoded.encode()) > MAX_BYTES:
                encoded = json.dumps({"ok": False, "error": "ReplyTooLarge"})
            process.stdin.write(encoded + "\n")
            process.stdin.flush()
    except Exception as error:
        error.usage = dict(usage)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def isolated_request(mode, body=None, *, source="user", route_mode="auto"):
    if source not in ("user", "synthetic"):
        raise ValueError("Неизвестный источник запроса")
    if route_mode not in ("auto", "standard"):
        raise ValueError("Неизвестный маршрут")
    with traced_span("gateway." + mode, {"openinference.span.kind": "CHAIN"}):
        return _isolated_request(mode, body, source, route_mode)


def _isolated_request(mode, body=None, source="user", route_mode="auto"):
    root = storage_root()
    started = monotonic()
    record = {"run_id": uuid4().hex, "mode": mode, "source": source, "status": "error"}
    sampler = None
    try:
        try:
            with FileLock(str(root / "gateway.lock"), timeout=0):
                settings = sandbox_settings()
                if mode in ("chat", "judge"):
                    with GpuSampler() as sampler:
                        result = run_worker(settings, {"mode": mode, "body": body or {}, "route_mode": route_mode})
                    result["resources"] = {"worker": result.pop("worker_resources", None), "gpu": sampler.report()}
                else:
                    result = run_worker(settings, {"mode": mode, "body": body or {}})
        except Timeout as error:
            record["status"] = "busy"
            if source == "synthetic":
                return {"status": "skipped_busy", "run_id": record["run_id"]}
            raise RuntimeError("Киноассистент занят другим запросом") from error
        record["status"] = result.get("status", "ok")
        if mode == "chat":
            record.update({
                "route": (result.get("routing") or {}).get("choice"),
                "model": (result.get("routing") or {}).get("model"),
                "validator_model": (result.get("validation") or {}).get("model"),
                "attempts": result.get("attempts"),
                "movie_count": len(result.get("movies", [])),
                "method": result.get("method"),
                "fallback": result.get("fallback"),
                "tool_calls": len(result.get("tool_events", [])),
                "validator_rejections": result.get("validator_rejections", 0),
            })
        if mode in ("chat", "judge"):
            record.update(result.get("usage", {}))
            result["run_id"] = record["run_id"]
            result["processing_seconds"] = round(monotonic() - started, 6)
            worker = result["resources"]["worker"]
            if worker is not None:
                record["worker_peak_rss_bytes"] = worker["peak_rss_bytes"]
        return result
    except Exception as error:
        record["error_type"] = type(error).__name__
        record.update(getattr(error, "usage", {}))
        raise
    finally:
        record["seconds"] = round(monotonic() - started, 6)
        if sampler is not None:
            gpu = sampler.report()
            record.update(gpu_status=gpu["status"], gpu_sample_count=len(gpu["samples"]),
                          gpu_utilization_observed_peak_percent=gpu["utilization_observed_peak_percent"],
                          gpu_memory_used_observed_peak_mib=gpu["memory_used_observed_peak_mib"])
        annotate_request(record)
        write_request(root, record)
