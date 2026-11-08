import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from movie_assistant.settings import storage_root

ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
FILES = ("u.item", "u.genre", "u.data", "README")


def prepare(archive: Path, target: Path) -> None:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Контрольная сумма архива не совпала")
    with ZipFile(archive) as source:
        contents = {name: source.read("ml-100k/" + name) for name in FILES}
    
    for name, data in contents.items():
        path = target / name
        if path.is_symlink() or (path.exists() and path.read_bytes() != data):
            raise FileExistsError(f"Отказ от перезаписи изменённого файла: {path}")
    manifest = json.dumps({name: hashlib.sha256(data).hexdigest()
                           for name, data in contents.items()}, indent=2) + "\n"
    metadata = target / "checksums.json"
    if metadata.is_symlink() or (metadata.exists() and metadata.read_text(encoding="utf-8") != manifest):
        raise FileExistsError(f"Отказ от перезаписи контрольных сумм: {metadata}")
    target.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        if not (target / name).exists():
            (target / name).write_bytes(data)
    if not metadata.exists():
        metadata.write_text(manifest, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description='Подготовить уже скачанный официальный архив, не изменяя источник.')
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    target = storage_root() / "data/ml-100k"
    try:
        prepare(args.archive, target)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка подготовки: {error}\n")
    print("Архив и четыре файла проверены. Данные подготовлены:", target)


if __name__ == "__main__":
    main()
