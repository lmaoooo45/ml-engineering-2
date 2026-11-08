import json

from movie_assistant.gateway import isolated_request
from movie_assistant.settings import storage_root


def main():
    print("Запускается настоящий учебный запрос к моделям.", flush=True)
    result = isolated_request("chat", {
        "question": "Какого года Toy Story?", "profile": "resource-check",
        "session_id": None, "user_id": None,
    }, source="synthetic")
    if result["status"] != "ok":
        raise SystemExit("Учебный запрос не завершился успешно: " + result["status"])
    resources = result["resources"]
    worker, gpu = resources["worker"], resources["gpu"]
    path = storage_root() / "results" / ("resources-check-" + result["run_id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": result["run_id"],
                      "worker_peak_rss_mib": round(worker["peak_rss_bytes"] / 1024**2, 2),
                      "gpu_status": gpu["status"], "gpu_sample_count": len(gpu["samples"]),
                      "gpu_utilization_observed_peak_percent": gpu["utilization_observed_peak_percent"],
                      "gpu_memory_used_observed_peak_mib": gpu["memory_used_observed_peak_mib"]},
                     ensure_ascii=False, indent=2))
    print("Полные измерения: " + str(path))


if __name__ == "__main__":
    main()
