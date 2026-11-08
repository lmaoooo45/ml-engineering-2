import argparse
import json

from movie_assistant.gateway import isolated_request


def main():
    parser = argparse.ArgumentParser(description='Показать, изменить или удалить локальный профиль.')
    parser.add_argument("--profile", default="default")
    parser.add_argument("--genres", nargs="*", help="Жанры по умолчанию")
    parser.add_argument("--exclude", nargs="*", help="Названия фильмов, которые нужно исключить")
    parser.add_argument("--forget", action="store_true", help="Удалить предпочтения и сессии профиля")
    args = parser.parse_args()
    editing = args.genres is not None or args.exclude is not None
    if args.forget and editing:
        parser.error("Удаление и изменение запускаются отдельно")
    try:
        result = isolated_request("profile", {
            "profile": args.profile, "genres": args.genres,
            "exclude_titles": args.exclude, "forget": args.forget,
        })
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"Ошибка профиля: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
