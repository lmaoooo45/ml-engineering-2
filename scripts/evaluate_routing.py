import hashlib
import json
from datetime import datetime, timezone

from movie_assistant.ollama_client import MODELS
from movie_assistant.routing import choose_route
from movie_assistant.settings import PROJECT_ROOT, storage_root


def main():
    root = storage_root()
    raw = (PROJECT_ROOT / "benchmarks/routing_cases.json").read_bytes()
    cases = json.loads(raw)["cases"]
    rows = []
    for case in cases:
        route = choose_route(case["question"], case.get("history", []), root)
        row = {
            "case": case["id"], "split": case["split"], "expected": case["expected_choice"],
            "correct_choice": route.get("choice") == case["expected_choice"], "routing": route,
        }
        rows.append(row)
    summary = {}
    for split in ("dev", "holdout"):
        selected = [row for row in rows if row["split"] == split]
        summary[split] = {
            "total": len(selected),
            "correct_choice": sum(row["correct_choice"] for row in selected),
            "fast_allowed": sum(row["routing"]["model"] == MODELS["ministral"] for row in selected),
            "unsafe_fast": sum(row["routing"]["model"] == MODELS["ministral"] and row["expected"] != "fast"
                               for row in selected),
        }
    created = datetime.now(timezone.utc)
    path = root / "results" / created.strftime("routing-%Y%m%dT%H%M%S%fZ.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_sha256": hashlib.sha256((PROJECT_ROOT / "movie_assistant/routing.py").read_bytes()).hexdigest(),
        "note": "Наборы перенесены из предыдущей версии блока. Это повтор известного опыта.",
        "summary": summary, "rows": rows,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Результаты сохранены:", path)


if __name__ == "__main__":
    main()
