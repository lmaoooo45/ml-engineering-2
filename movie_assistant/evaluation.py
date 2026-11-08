from collections import defaultdict
from math import log2
from statistics import mean


def relevant_movies(baseline, later_ratings):
    relevant = defaultdict(set)
    skipped = 0
    for user_id, movie_id, rating, _ in later_ratings:
        if rating < 4:
            continue
        if movie_id in baseline.available and movie_id not in baseline.seen.get(user_id, ()):
            relevant[user_id].add(movie_id)
        else:
            skipped += 1
    if not relevant:
        raise ValueError("Нет пользователей с доступными фильмами для проверки")
    return relevant, skipped


def ranking_metrics(recommended, relevant, k):
    hits = [movie_id in relevant for movie_id in recommended[:k]]
    dcg = sum(hit / log2(position + 2) for position, hit in enumerate(hits))
    best_dcg = sum(1 / log2(position + 2) for position in range(min(k, len(relevant))))
    return {
        "precision": sum(hits) / k,
        "recall": sum(hits) / len(relevant),
        "ndcg": dcg / best_dcg,
    }


def evaluate(baseline, relevant, recommend, k):
    groups = {"all": [], "warm": [], "cold": []}
    unique_films = set()
    fallbacks = 0
    for user_id, expected in sorted(relevant.items()):
        movies, fallback = recommend(user_id, k)
        ids = [movie["id"] for movie in movies]
        metrics = ranking_metrics(ids, expected, k)
        group = "warm" if user_id in baseline.seen else "cold"
        groups["all"].append(metrics)
        groups[group].append(metrics)
        unique_films.update(ids)
        fallbacks += int(fallback)

    summary = {}
    for group, rows in groups.items():
        summary[group] = {
            "users": len(rows),
            **{key: mean(row[key] for row in rows) if rows else None
               for key in ("precision", "recall", "ndcg")},
        }
    summary["coverage"] = len(unique_films) / len(baseline.available)
    summary["unique_films"] = len(unique_films)
    summary["fallback_users"] = fallbacks
    return summary
