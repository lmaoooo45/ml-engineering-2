from datetime import datetime, timezone
import json
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from filelock import FileLock, Timeout

from movie_assistant.alerts import emit_alert
from movie_assistant.gateway import isolated_request

HTTP = build_opener(ProxyHandler({}))
ENDPOINTS = {
    "api": "http://127.0.0.1:8001/health",
    "ollama": "http://127.0.0.1:11434/api/version",
    "phoenix": "http://127.0.0.1:6006/healthz",
    "monitor": "http://127.0.0.1:8001/monitor",
}


def check_model(root):
    try:
        result = isolated_request("chat", {
            "question": "Какого года Toy Story?",
            "profile": "observer-self-check", "session_id": None, "user_id": None,
        }, source="synthetic")
        if result["status"] == "skipped_busy":
            return result
        movies = result.get("movies", [])
        passed = (result["status"] == "ok" and len(movies) == 1
                  and movies[0].get("id") == 1 and movies[0].get("year") == 1995)
        if not passed:
            emit_alert(root, "model_check_failed", "Учебная проверка модели получила неверный ответ",
                       result.get("run_id"))
        return {"status": "passed" if passed else "failed", "run_id": result.get("run_id")}
    except (OSError, RuntimeError, ValueError, KeyError):
        emit_alert(root, "model_check_failed", "Не удалось выполнить учебную проверку модели")
        return {"status": "failed"}


def fetch(url):
    with HTTP.open(url, timeout=2) as response:
        if response.status != 200:
            raise ValueError("Unexpected HTTP status")
        data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Too much monitoring data")
        return data.decode("utf-8")


def sample_once(root, previous=None):
    previous = previous or {}
    services = {}
    summary = None
    for name, url in ENDPOINTS.items():
        try:
            content = fetch(url)
            if name == "monitor":
                summary = json.loads(content)
                if not isinstance(summary, dict) or summary.get("scope") != "saved_gateway_requests":
                    raise ValueError("Unexpected monitor response")
            services[name] = True
        except (OSError, URLError, ValueError):
            services[name] = False
            if previous.get(name) is not False:
                emit_alert(root, "service_unavailable", "Наблюдатель: служба " + name + " недоступна")
    sample = {
        "time": datetime.now(timezone.utc).isoformat(),
        "services": services,
        "summary": summary if services["monitor"] else None,
    }
    directory = root / "logs"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / "observer-log.lock"), timeout=2):
            with (directory / "observer.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except (OSError, Timeout):
        emit_alert(root, "observer_log_failed", "Не удалось сохранить результаты наблюдения")
    return sample
