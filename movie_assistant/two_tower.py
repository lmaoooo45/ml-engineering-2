import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.nn.functional import softplus


class Tower(nn.Module):
    def __init__(self, count, genre_count):
        super().__init__()
        self.embedding = nn.Embedding(count, 16)
        self.layers = nn.Sequential(nn.Linear(16 + genre_count, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, ids, genres):
        return self.layers(torch.cat((self.embedding(ids), genres), dim=-1))


class TwoTower(nn.Module):
    def __init__(self, baseline, train):
        super().__init__()
        self.baseline = baseline
        self.movie_ids = sorted(baseline.available)
        self.item_index = {movie_id: index for index, movie_id in enumerate(self.movie_ids)}
        self.user_index = {user_id: index for index, user_id in enumerate(sorted(baseline.seen))}
        genres = sorted(baseline.genre_names)
        self.item_genres = torch.tensor([
            [float(genre in baseline.movies[movie_id]["genres"]) for genre in genres]
            for movie_id in self.movie_ids
        ])
        self.profiles = torch.zeros((len(self.user_index), len(genres)))
        for user_id, liked in baseline.likes.items():
            indices = [self.item_index[movie_id] for movie_id in sorted(liked)]
            self.profiles[self.user_index[user_id]] = self.item_genres[indices].mean(dim=0)
        self.user_tower = Tower(len(self.user_index), len(genres))
        self.item_tower = Tower(len(self.movie_ids), len(genres))
        self.film_vectors = None
        payload = json.dumps([baseline.movies, sorted(train)], sort_keys=True).encode()
        self.fingerprint = hashlib.sha256(payload).hexdigest()

    def fit(self, epochs, seed):
        if epochs < 1:
            raise ValueError("Нужна хотя бы одна эпоха")
        rng = random.Random(seed)
        negatives = {
            self.user_index[user_id]: [self.item_index[movie_id] for movie_id in
                                       sorted(self.baseline.available - self.baseline.seen[user_id])]
            for user_id in self.baseline.likes
        }
        pairs = [(self.user_index[user_id], self.item_index[movie_id])
                 for user_id, likes in sorted(self.baseline.likes.items())
                 for movie_id in sorted(likes) if negatives[self.user_index[user_id]]]
        if not pairs:
            raise ValueError("Нет пар для обучения")
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.003)
        for epoch in range(1, epochs + 1):
            self.train()
            rng.shuffle(pairs)
            total_loss = 0.0
            for start in range(0, len(pairs), 1024):
                batch = pairs[start:start + 1024]
                user_indices = []
                liked_film_indices = []
                unscored_film_indices = []
                for user_index, liked_film in batch:
                    unscored_films = rng.choices(negatives[user_index], k=3)
                    for unscored_film in unscored_films:
                        user_indices.append(user_index)
                        liked_film_indices.append(liked_film)
                        unscored_film_indices.append(unscored_film)

                users = torch.tensor(user_indices)
                positives = torch.tensor(liked_film_indices)
                sampled = torch.tensor(unscored_film_indices)
                user_vectors = self.user_tower(users, self.profiles[users])
                positive_vectors = self.item_tower(positives, self.item_genres[positives])
                sampled_vectors = self.item_tower(sampled, self.item_genres[sampled])
                
                film_difference = sampled_vectors - positive_vectors
                score_difference = (user_vectors * film_difference).sum(dim=-1)
                loss = softplus(score_difference).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.film_vectors = None
            yield epoch, total_loss / len(pairs)

    @torch.inference_mode()
    def scores(self, user_id):
        if not self.baseline.likes.get(user_id):
            return self.baseline.popularity, True
        self.eval()
        index = torch.tensor([self.user_index[user_id]])
        user_vector = self.user_tower(index, self.profiles[index])[0]
        if self.film_vectors is None:
            self.film_vectors = self.item_tower(torch.arange(len(self.movie_ids)), self.item_genres)
        return dict(zip(self.movie_ids, (self.film_vectors @ user_vector).tolist())), False

    def save(self, path, epochs, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as temporary:
            staged = Path(temporary) / path.name
            torch.save({"format": 1, "fingerprint": self.fingerprint, "weights": self.state_dict(),
                        "epochs": epochs, "seed": seed}, staged)
            staged.replace(path)

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["format"] != 1 or saved["fingerprint"] != self.fingerprint:
            raise ValueError("Веса Two-Tower не соответствуют каталогу или обучающим оценкам")
        self.load_state_dict(saved["weights"])
        self.film_vectors = None
