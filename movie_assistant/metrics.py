from collections import Counter
import json
import math

from filelock import FileLock


def read_summary(root):
    counts = Counter()
    seconds = 0.0
    chat_count = 0
    synthetic_count = 0
    skipped = 0
    latest_resources = None
    usage_totals = {source: {key: 0 for key in ("llm_calls", "decision_calls", "prompt_tokens",
                                               "completion_tokens", "calls_with_token_counts", "validator_rejections")}
                    for source in ("user", "synthetic")}
    path = root / "logs" / "requests.jsonl"
    if path.exists():
        with FileLock(str(path.parent / "requests.lock"), timeout=2):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        mode, status, duration = row["mode"], row["status"], row["seconds"]
                        source_name = row.get("source", "user")
                        if (mode not in ("chat", "profile", "probe", "judge")
                                or source_name not in ("user", "synthetic")
                                or status not in ("ok", "clarify", "unsupported", "error", "busy")
                                or type(duration) not in (int, float)
                                or not math.isfinite(duration) or duration < 0):
                            raise ValueError("Invalid journal entry")
                    except (ValueError, KeyError, TypeError):
                        skipped += 1
                        continue
                    counts[mode, status, source_name] += 1
                    for key in usage_totals[source_name]:
                        value = row.get(key)
                        if type(value) is int and value >= 0:
                            usage_totals[source_name][key] += value
                    if mode == "chat" and "gpu_status" in row:
                        latest_resources = {"source": source_name, "gpu_status": row["gpu_status"]}
                        for key in ("worker_peak_rss_bytes", "gpu_sample_count",
                                    "gpu_utilization_observed_peak_percent", "gpu_memory_used_observed_peak_mib"):
                            value = row.get(key)
                            latest_resources[key] = value if (type(value) in (int, float)
                                                            and math.isfinite(value) and value >= 0) else None
                    if mode == "chat" and source_name == "synthetic":
                        synthetic_count += 1
                    if mode == "chat" and source_name == "user":
                        chat_count += 1
                        seconds += duration
    return {
        "scope": "saved_gateway_requests",
        "requests": [
            {"mode": mode, "status": status, "source": source_name, "count": count}
            for (mode, status, source_name), count in sorted(counts.items())
        ],
        "chat_count": chat_count,
        "synthetic_chat_count": synthetic_count,
        "chat_seconds_sum": seconds,
        "chat_seconds_mean": seconds / chat_count if chat_count else None,
        "skipped_lines": skipped,
        "latest_resources": latest_resources,
        "usage_totals": usage_totals,
    }


def prometheus_text(summary):
    lines = [
        "# HELP movie_gateway_requests_total Completed gateway calls in the retained journal.",
        "# TYPE movie_gateway_requests_total counter",
    ]
    for row in summary["requests"]:
        lines.append('movie_gateway_requests_total{mode="%s",status="%s",source="%s"} %d'
                     % (row["mode"], row["status"], row["source"], row["count"]))
    lines += [
        "# HELP movie_gateway_chat_seconds Total gateway wall time for retained user chat calls.",
        "# TYPE movie_gateway_chat_seconds summary",
        f'movie_gateway_chat_seconds_sum {summary["chat_seconds_sum"]}',
        f'movie_gateway_chat_seconds_count {summary["chat_count"]}',
        "# HELP movie_gateway_skipped_lines Unreadable journal lines.",
        "# TYPE movie_gateway_skipped_lines gauge",
        f'movie_gateway_skipped_lines {summary["skipped_lines"]}',
    ]
    latest = summary["latest_resources"]
    if latest:
        for key in ("worker_peak_rss_bytes", "gpu_sample_count",
                    "gpu_utilization_observed_peak_percent", "gpu_memory_used_observed_peak_mib"):
            if latest[key] is not None:
                name = "movie_gateway_last_" + key
                lines += [f"# HELP {name} Last completed measured chat call; GPU values cover all processes.",
                          f"# TYPE {name} gauge",
                          f'{name}{{source="{latest["source"]}"}} {latest[key]}']
    for source, values in summary["usage_totals"].items():
        for key, value in values.items():
            name = "movie_gateway_" + key + "_total"
            if source == "user":
                lines += [f"# HELP {name} Retained measured gateway usage; token counts may be incomplete.",
                          f"# TYPE {name} counter"]
            lines.append(f'{name}{{source="{source}"}} {value}')
    return "\n".join(lines) + "\n"
