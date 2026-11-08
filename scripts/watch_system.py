import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from filelock import FileLock, Timeout

from movie_assistant.observer import sample_once, check_model
from movie_assistant.settings import storage_root


def main():
    parser = argparse.ArgumentParser(description='Проверять локальные службы раз в 10 секунд. Остановка: Ctrl+C.')
    parser.add_argument("--once", action="store_true", help="Выполнить одну проверку и завершиться")
    parser.add_argument("--check-model", action="store_true", help="Учебный запрос сейчас и каждые 15 минут, если агент свободен")
    args = parser.parse_args()
    root = storage_root()
    previous = None
    pending = None
    next_check = 0
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with FileLock(str(root / "observer.lock"), timeout=0):
            while True:
                started = time.monotonic()
                sample = sample_once(root, previous)
                print(json.dumps(sample, ensure_ascii=False), flush=True)
                previous = sample["services"]
                if pending is not None and pending.done():
                    result = pending.result()
                    print("Учебная проверка: " + json.dumps(result, ensure_ascii=False), flush=True)
                    next_check = time.monotonic() + (10 if result["status"] == "skipped_busy" else 900)
                    pending = None
                if args.check_model and pending is None and time.monotonic() >= next_check:
                    pending = executor.submit(check_model, root)
                if args.once:
                    if pending is not None:
                        print("Учебная проверка: " + json.dumps(pending.result(), ensure_ascii=False), flush=True)
                    return
                time.sleep(max(0, 10 - (time.monotonic() - started)))
    except Timeout:
        parser.exit(1, "Наблюдатель уже запущен\n")
    except KeyboardInterrupt:
        print("Остановка наблюдения. Если учебный запрос уже выполняется, ожидаем его завершения.")
    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
