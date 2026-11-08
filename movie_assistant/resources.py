from datetime import datetime, timezone
import math
import shutil
import subprocess
import threading


def worker_resources():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"scope": "linux_worker_process", "peak_rss_bytes": usage.ru_maxrss * 1024,
            "cpu_user_seconds": usage.ru_utime, "cpu_system_seconds": usage.ru_stime}


def validate_worker_resources(data):
    if not isinstance(data, dict) or data.get("scope") != "linux_worker_process":
        raise ValueError("Недопустимые измерения worker")
    if type(data.get("peak_rss_bytes")) is not int or not 0 < data["peak_rss_bytes"] <= 4 * 1024**3:
        raise ValueError("Недопустимая память worker")
    for key in ("cpu_user_seconds", "cpu_system_seconds"):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Недопустимое CPU-время worker")
    return {key: data[key] for key in ("scope", "peak_rss_bytes", "cpu_user_seconds", "cpu_system_seconds")}


def read_gpu():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise OSError("nvidia-smi unavailable")
    completed = subprocess.run([
        executable, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    readings = []
    for line in completed.stdout.splitlines():
        index, utilization, used, total = [part.strip() for part in line.split(",")]
        index = int(index)
        utilization, used, total = float(utilization), float(used), float(total)
        if (index < 0 or not all(math.isfinite(value) for value in (utilization, used, total))
                or not 0 <= utilization <= 100 or not 0 <= used <= total or total <= 0):
            raise ValueError("Invalid GPU reading")
        readings.append({"gpu_index": index, "utilization_percent": utilization,
                         "memory_used_mib": used, "memory_total_mib": total})
    if not readings:
        raise ValueError("No GPU readings")
    return readings


class GpuSampler:
    def __init__(self, interval=2):
        self.interval = interval
        self.samples = []
        self.errors = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                readings = read_gpu()
                self.samples.append({"time": datetime.now(timezone.utc).isoformat(), "gpus": readings})
            except (OSError, ValueError, subprocess.SubprocessError):
                self.errors += 1
            if len(self.samples) >= 120 or self.stop.wait(self.interval):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()

    def report(self):
        readings = [gpu for sample in self.samples for gpu in sample["gpus"]]
        return {
            "scope": "whole_gpu_all_processes", "interval_seconds": self.interval,
            "status": ("partial" if self.errors else "ok") if readings else "unavailable",
            "sample_errors": self.errors,
            "utilization_observed_peak_percent": max((gpu["utilization_percent"] for gpu in readings), default=None),
            "memory_used_observed_peak_mib": max((gpu["memory_used_mib"] for gpu in readings), default=None),
            "samples": list(self.samples),
        }
