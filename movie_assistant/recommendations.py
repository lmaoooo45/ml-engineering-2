from collections import Counter, defaultdict
from math import log, sqrt
from pathlib import Path

METHODS = ("content", "collaborative", "popular")
Rating = tuple[int, int, int, int]


def load_ratings(directory: Path) -> list[Rating]:
    ratings = []
    for line in (directory / "u.data").read_text(encoding="ascii").splitlines():
        user_id, movie_id, rating, timestamp = map(int, line.split())
        ratings.append((user_id, movie_id, rating, timestamp))
    return ratings


def temporal_split(ratings: list[Rating], train_fraction: float = 0.8) -> tuple[list[Rating], list[Rating]]:
    
    if not 0 < train_fraction < 1:
        raise ValueError("Доля обучения должна быть между 0 и 1")
    ordered = sorted(ratings, key=lambda row: row[3])
    boundary = int(len(ordered) * train_fraction)
    while 0 < boundary < len(ordered) and ordered[boundary - 1][3] == ordered[boundary][3]:
        boundary += 1
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("Не удалось разделить оценки по времени")
    return ordered[:boundary], ordered[boundary:]


def unit_vector(values: dict[str, float]) -> dict[str, float]:
    length = sqrt(sum(value ** 2 for value in values.values()))
    return {key: value / length for key, value in values.items()} if length else {}


class MovieRecommender:
    def __init__(self, movies: dict[int, dict], train: list[Rating]):
        self.movies = movies
        self.seen = defaultdict(set)
        self.likes = defaultdict(set)
        self.available = set()
        for user_id, movie_id, rating, _ in train:
            self.seen[user_id].add(movie_id)
            self.available.add(movie_id)
            if rating >= 4:
                self.likes[user_id].add(movie_id)

        self.popularity = Counter(movie_id for liked in self.likes.values() for movie_id in liked)
        self.genre_names = {genre for movie in movies.values() for genre in movie["genres"]}
        tags = {movie_id: movies[movie_id]["genres"] or ["unknown"] for movie_id in self.available}
        frequency = Counter(genre for genres in tags.values() for genre in genres)
        self.features = {}
        for movie_id, genres in tags.items():
            weights = {genre: 1 + log((1 + len(tags)) / (1 + frequency[genre])) for genre in genres}
            self.features[movie_id] = unit_vector(weights)

    def content_scores(self, liked: set[int]) -> dict[int, float]:
        profile = defaultdict(float)
        for movie_id in sorted(liked):
            for genre, weight in self.features[movie_id].items():
                profile[genre] += weight
        profile = unit_vector(profile)
        return {
            movie_id: sum(weight * profile.get(genre, 0) for genre, weight in features.items())
            for movie_id, features in self.features.items()
        }

    def collaborative_scores(self, user_id: int, liked: set[int]) -> dict[int, float]:
        neighbors = []
        for other_user, other_likes in self.likes.items():
            shared = len(liked & other_likes)
            if other_user != user_id and shared:
                similarity = shared / sqrt(len(liked) * len(other_likes))
                neighbors.append((other_user, similarity))
        neighbors.sort(key=lambda neighbor: (-neighbor[1], neighbor[0]))
        nearest = neighbors[:30]
        total_weight = sum(weight for _, weight in nearest)
        scores = defaultdict(float)
        for other_user, weight in nearest:
            for movie_id in self.likes[other_user]:
                scores[movie_id] += weight / total_weight
        return scores

    def recommend(self, user_id: int, method: str, k: int = 10, *,
                  genres=(), genre_mode: str = "and", exclude_ids=()) -> tuple[list[dict], bool]:
        if method not in METHODS:
            raise ValueError("Неизвестный метод")
        liked = self.likes.get(user_id, set())
        scores = self.popularity
        fallback = False
        if method != "popular":
            if not liked:
                fallback = True
            elif method == "content":
                scores = self.content_scores(liked)
            else:
                neighbor_scores = self.collaborative_scores(user_id, liked)
                if neighbor_scores:
                    scores = neighbor_scores
                else:
                    fallback = True

        return self.rank(user_id, scores, k, genres=genres, genre_mode=genre_mode,
                         exclude_ids=exclude_ids), fallback

    def rank(self, user_id, scores, k, *, genres=(), genre_mode="and", exclude_ids=()):
        if genre_mode not in {"and", "or"}:
            raise ValueError("Неизвестный режим жанров")
        if user_id < 0 or not 1 <= k <= 10:
            raise ValueError("ID пользователя должен быть неотрицательным, k должно быть от 1 до 10")
        requested = set(genres)
        if requested - self.genre_names:
            raise ValueError("Неизвестный жанр: " + ", ".join(sorted(requested - self.genre_names)))
        candidates = self.available - self.seen.get(user_id, set()) - set(exclude_ids)
        allowed = []
        for movie_id in candidates:
            movie_genres = set(self.movies[movie_id]["genres"])
            if requested:
                matches = requested <= movie_genres if genre_mode == "and" else bool(requested & movie_genres)
                if not matches:
                    continue
            allowed.append(movie_id)
        allowed.sort(key=lambda movie_id: (-scores.get(movie_id, 0), movie_id))
        return [{**self.movies[movie_id], "score": scores.get(movie_id, 0)}
                for movie_id in allowed[:k]]
