import argparse
import json
from datetime import datetime, timezone

import torch

from movie_assistant.catalog import load_catalog, verify_files
from movie_assistant.evaluation import evaluate, relevant_movies
from movie_assistant.recommendations import METHODS, MovieRecommender, load_ratings, temporal_split
from movie_assistant.settings import storage_root
from movie_assistant.two_tower import TwoTower

SEEDS = (42, 43, 44)
TOP_K = 10


def model_recommendations(model, baseline):
    def recommend(user_id, k):
        scores, fallback = model.scores(user_id)
        return baseline.rank(user_id, scores, k), fallback
    return recommend


def choose_epochs(movies, ratings, validation, seed, max_epochs):
    baseline = MovieRecommender(movies, ratings)
    relevant, _ = relevant_movies(baseline, validation)
    torch.manual_seed(seed)
    model = TwoTower(baseline, ratings)
    recommend = model_recommendations(model, baseline)
    history = []
    for epoch, loss in model.fit(max_epochs, seed):
        ndcg = evaluate(baseline, relevant, recommend, TOP_K)["all"]["ndcg"]
        history.append({"epoch": epoch, "loss": loss, "ndcg": ndcg})
    
    return max(history, key=lambda row: row["ndcg"])["epoch"], history


def print_summary(name, summary):
    overall = summary["all"]
    print(f"{name:18} {overall['precision']:.4f} {overall['recall']:.4f} "
          f"{overall['ndcg']:.4f} {summary['coverage']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Подобрать число эпох внутри обучающих данных и сравнить четыре метода.')
    parser.add_argument("--max-epochs", type=int, default=20, choices=range(1, 21), metavar="1..20")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = storage_root()
    directory = root / "data/ml-100k"
    verify_files(directory)
    movies = load_catalog(directory)
    train, test = temporal_split(load_ratings(directory))
    inner_train, validation = temporal_split(train, 0.85)
    baseline = MovieRecommender(movies, train)
    relevant, skipped = relevant_movies(baseline, test)
    print(f"Обучение: {len(train)}. Тест: {len(test)}. Пользователей: {len(relevant)}.", flush=True)
    print("Метод              Precision Recall NDCG   Покрытие", flush=True)
    results = {}
    for method in METHODS:
        results[method] = evaluate(
            baseline, relevant, lambda user_id, k: baseline.recommend(user_id, method, k), TOP_K,
        )
        print_summary(method, results[method])

    for seed in SEEDS:
        epochs, history = choose_epochs(movies, inner_train, validation, seed, args.max_epochs)
        print(f"Two-Tower seed {seed}: по внутренней проверке выбрано эпох: {epochs}", flush=True)
        torch.manual_seed(seed)
        model = TwoTower(baseline, train)
        for _ in model.fit(epochs, seed):
            pass
        name = f"two-tower-{seed}"
        results[name] = evaluate(baseline, relevant, model_recommendations(model, baseline), TOP_K)
        results[name]["selected_epochs"] = epochs
        results[name]["validation_history"] = history
        print_summary(name, results[name])

    created = datetime.now(timezone.utc)
    output = root / "results" / created.strftime("comparison-%Y%m%dT%H%M%S%fZ.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": created.isoformat(), "top_k": TOP_K, "max_epochs": args.max_epochs,
        "train": len(train), "test": len(test), "inner_train": len(inner_train),
        "validation": len(validation), "cutoff": train[-1][3], "eligible_movies": len(baseline.available),
        "skipped_positive_ratings": skipped, "fingerprint": model.fingerprint,
        "main_seed": SEEDS[0], "methods": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Результаты сохранены:", output)


if __name__ == "__main__":
    main()
