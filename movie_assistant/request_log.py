from datetime import datetime, timezone
from collections import deque
import json
from pathlib import Path

from filelock import FileLock, Timeout
from movie_assistant.alerts import check_request_alerts, emit_alert


def write_request(root: Path, record: dict):
    try:
        directory = root / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        entry = {"finished_at": datetime.now(timezone.utc).isoformat(), **record}
        with FileLock(str(directory / "requests.lock"), timeout=2):
            with (directory / "requests.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if record["mode"] == "chat":
                recent = deque(maxlen=4)
                with (directory / "requests.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        try:
                            row = json.loads(line)
                            if (row["mode"] == "chat"
                                    and row.get("source", "user") == record.get("source", "user")
                                    and row["status"] in (
                                    "ok", "clarify", "unsupported", "error", "busy")):
                                recent.append(row)
                        except (ValueError, KeyError, TypeError):
                            continue
                check_request_alerts(root, record, recent)
    except (OSError, TimeoutError, Timeout):
        emit_alert(root, "request_log_failed", "Не удалось записать журнал запроса")
