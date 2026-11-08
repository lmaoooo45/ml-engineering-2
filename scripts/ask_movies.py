import argparse
import json

from movie_assistant.gateway import isolated_request


def main():
    parser = argparse.ArgumentParser(description='Запустить киноассистента из терминала.')
    parser.add_argument("question")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--session", help="ID предыдущей сессии из вывода")
    args = parser.parse_args()
    try:
        result = isolated_request("chat", {
            "question": args.question, "profile": args.profile,
            "session_id": args.session, "user_id": args.user_id,
        })
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"Запрос не выполнен: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
