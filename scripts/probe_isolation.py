import json

from movie_assistant.gateway import isolated_request


def main():
    try:
        result = isolated_request("probe")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Проверка не выполнена: {error}") from None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
