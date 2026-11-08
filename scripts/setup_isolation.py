import argparse
import base64
import json
import sqlite3
import subprocess

from deployment.snapshot import make_snapshot
from movie_assistant.settings import PROJECT_ROOT, storage_root

INSTALL = """
from io import BytesIO
import base64, hashlib, json, os, pathlib, subprocess, sys, zipfile
payload = json.load(sys.stdin)
archive = base64.b64decode(payload['archive'], validate=True)
base = pathlib.Path.home()/'.local/share/pi-block2/sandbox'
base.mkdir(parents=True, exist_ok=True)
stage = base/('stage-'+hashlib.sha256(archive).hexdigest()[:16])
stage.mkdir(exist_ok=True)
with zipfile.ZipFile(BytesIO(archive)) as source:
    for name in source.namelist():
        target = (stage/name).resolve()
        if not target.is_relative_to(stage.resolve()):
            raise ValueError('Unsafe archive path')
        data = source.read(name)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Staged file was modified')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
venv = pathlib.Path(payload['venv'] or str(pathlib.Path.home()/'.local/share/pi-block2/venv')).resolve()
python = str(venv/'bin/python')
python_base = subprocess.check_output([python, '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()
python_alias = str(pathlib.Path(os.readlink(venv/'bin/python')).parent.parent)
state = base/'state'
for directory in (state, state/'data/ml-100k', state/'models'):
    directory.mkdir(parents=True, exist_ok=True)
state.chmod(0o700)
memory = state/'memory.sqlite'
if not memory.exists() and payload['memory']:
    memory.write_bytes(base64.b64decode(payload['memory'], validate=True))
(base/'host-canary.txt').write_text('Host-only marker')
(stage/'launch.json').write_text(json.dumps({
    'venv':str(venv), 'python_base':python_base, 'python_alias':python_alias,
    'state':str(state), 'host_home':str(pathlib.Path.home()),
}))
print(json.dumps({'stage':str(stage), 'venv':str(venv), 'state':str(state)}))
"""


def main():
    parser = argparse.ArgumentParser(description='Сохранить проверенный снимок проекта в Ubuntu и настроить запуск.')
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--venv", help="Полный путь Linux-окружения")
    args = parser.parse_args()
    storage = storage_root()
    archive, fingerprint = make_snapshot(storage)
    memory = b""
    path = storage / "memory.sqlite"
    if path.exists():
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(":memory:") as backup:
                source.backup(backup)
                memory = backup.serialize()
    payload = {"archive": base64.b64encode(archive).decode(), "venv": args.venv,
               "memory": base64.b64encode(memory).decode()}
    process = subprocess.run(
        ["wsl.exe", "-d", args.distro, "--exec", "python3", "-c", INSTALL],
        input=json.dumps(payload).encode(), capture_output=True, timeout=60,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace")[-2000:])
    settings = {**json.loads(process.stdout), "distro": args.distro, "snapshot_sha256": fingerprint}
    (PROJECT_ROOT / "runtime-sandbox.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
