import hashlib
from io import BytesIO
from zipfile import ZipFile, ZipInfo

from movie_assistant.settings import PROJECT_ROOT


def make_snapshot(storage):
    files = {}
    for folder in ("movie_assistant", "prompts", "skills"):
        for path in (PROJECT_ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".txt"}:
                files["app/" + path.relative_to(PROJECT_ROOT).as_posix()] = path.read_bytes()
    for name in ("SOUL.txt", "scripts/__init__.py", "scripts/sandbox_worker.py", "deployment/linux_launch.py"):
        files["app/" + name] = (PROJECT_ROOT / name).read_bytes()
    for name in ("u.data", "u.item", "u.genre", "README", "checksums.json"):
        files["data/" + name] = (storage / "data/ml-100k" / name).read_bytes()
    files["models/two-tower.pt"] = (storage / "models/two-tower.pt").read_bytes()
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, content in sorted(files.items()):
            archive.writestr(ZipInfo(name), content)
    data = stream.getvalue()
    return data, hashlib.sha256(data).hexdigest()
