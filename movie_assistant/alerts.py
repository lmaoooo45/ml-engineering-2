from datetime import datetime, timezone
import json
import sys

from filelock import FileLock, Timeout

SLOW_SECONDS = 60


def emit_alert(root, kind, message, run_id=None):
    record = {"time": datetime.now(timezone.utc).isoformat(),
              "kind": kind, "message": message, "run_id": run_id}
    print("Предупреждение: " + message, file=sys.stderr)
    try:
        directory = root / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / "alerts.lock"), timeout=2):
            with (directory / "alerts.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, Timeout):
        print("Предупреждение: не удалось сохранить предупреждение в файл", file=sys.stderr)


def check_request_alerts(root, record, recent):
    if record["mode"] != "chat":
        return
    run_id = record.get("run_id")
    if record.get("error_type") == "TimeoutError":
        emit_alert(root, "request_timeout", "Запрос превысил допустимое время выполнения", run_id)
    if record["status"] != "busy" and record["seconds"] > SLOW_SECONDS:
        emit_alert(root, "slow_request", "Обработка запроса заняла больше 60 секунд", run_id)
    statuses = [row["status"] for row in recent]
    if statuses[-3:] == ["error"] * 3 and (len(statuses) == 3 or statuses[-4] != "error"):
        emit_alert(root, "error_streak", "Три запроса к агенту подряд завершились ошибкой", run_id)
