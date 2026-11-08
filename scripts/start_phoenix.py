import os
import subprocess

from movie_assistant.settings import storage_root


def main():
    root = storage_root()
    python = root / "phoenix-venv/Scripts/python.exe"
    if not python.is_file():
        raise SystemExit("Создай отдельное окружение phoenix-venv и установи requirements-phoenix.txt")
    environment = dict(os.environ)
    environment.update({
        "PHOENIX_WORKING_DIR": str(root / "phoenix"),
        "PHOENIX_HOST": "127.0.0.1",
        "PHOENIX_PORT": "6006",
        "PHOENIX_ENABLE_PROMETHEUS": "false",
        "PHOENIX_TELEMETRY_ENABLED": "false",
        "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
        "PHOENIX_DISABLE_AGENT_ASSISTANT": "true",
        "PHOENIX_ENABLE_MCP_SERVER": "false",
    })
    raise SystemExit(subprocess.call([str(python), "-m", "phoenix.server.main", "serve"],
                                    env=environment))


if __name__ == "__main__":
    main()
