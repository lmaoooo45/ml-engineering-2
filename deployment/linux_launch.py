import json
from pathlib import Path
import resource
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
config = json.loads((root / "launch.json").read_text())
venv, python_base, state = (Path(config[k]).resolve() for k in ("venv", "python_base", "state"))
if root.parent.name != "sandbox" or venv.name != "venv" or state.parent != root.parent:
    raise ValueError("Unexpected isolated runtime layout")
resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
resource.setrlimit(resource.RLIMIT_FSIZE, (64*1024**2, 64*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5", "190",
           "/usr/bin/bwrap", "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
           "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--proc", "/proc", "--dev", "/dev", "--size", "67108864", "--tmpfs", "/tmp",
           "--ro-bind", str(venv), str(venv), "--ro-bind", str(python_base), str(python_base),
           "--ro-bind", str(python_base), config["python_alias"],
           "--ro-bind", str(root / "app"), "/app", "--bind", str(state), "/state",
           "--ro-bind", str(root / "data"), "/state/data/ml-100k",
           "--ro-bind", str(root / "models"), "/state/models",
           "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/state",
           "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PYTHONPATH", "/app",
           "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "2",
           "--setenv", "MOVIE_OLLAMA_BRIDGE", "stdio", "--chdir", "/app",
           "--setenv", "MOVIE_STORAGE_DIR", "/state",
           "--setenv", "MOVIE_SANDBOX_CANARY", str(root.parent / "host-canary.txt"),
           "--setenv", "MOVIE_SANDBOX_HOST_HOME", config["host_home"],
           str(venv / "bin/python"), "-m", "scripts.sandbox_worker"]
raise SystemExit(subprocess.call(command))
