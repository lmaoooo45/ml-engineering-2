import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def storage_root() -> Path:
    
    configured = os.environ.get("MOVIE_STORAGE_DIR")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("MOVIE_STORAGE_DIR должен содержать абсолютный путь")
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise ValueError("Не задан LOCALAPPDATA; укажи MOVIE_STORAGE_DIR")
        path = Path(local) / "ml-engineering-block2-stepwise"
    else:
        path = Path.home() / ".local/share/ml-engineering-block2-stepwise"

    path = path.resolve()
    if path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Хранилище должно находиться вне папки исходников")
    return path
