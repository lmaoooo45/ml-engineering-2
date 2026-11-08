from datetime import datetime, timezone

from movie_assistant.request_log import write_request
from movie_assistant.settings import storage_root


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = storage_root() / "results" / "alerts-demo" / stamp
    print("Учебная проверка: время и ошибки заданы искусственно. Модели не вызываются.", flush=True)
    for index in range(3):
        row = {"mode": "chat", "run_id": f"demo-{index}", "status": "error", "seconds": 1}
        if index == 2:
            row.update(seconds=181, error_type="TimeoutError")
        write_request(root, row)
    print("Файл учебных предупреждений:")
    print(root / "logs" / "alerts.jsonl")
    print("Рабочий журнал запросов не изменён.")


if __name__ == "__main__":
    main()
