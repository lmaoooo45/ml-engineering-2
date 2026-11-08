from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from movie_assistant.recommendations import MovieRecommender
from movie_assistant.two_tower import TwoTower


class TwoTowerTests(unittest.TestCase):
    def test_learns_both_towers_and_reloads_matching_weights(self):
        movies = {movie_id: {"id": movie_id, "genres": ["Comedy"]} for movie_id in (1, 2)}
        train = [(1, 1, 5, 1), (2, 2, 5, 2)]
        baseline = MovieRecommender(movies, train)
        torch.manual_seed(42)
        model = TwoTower(baseline, train)
        old_user = model.user_tower.embedding.weight.detach().clone()
        old_film = model.item_tower.embedding.weight.detach().clone()
        list(model.fit(1, 42))
        self.assertFalse(torch.equal(old_user, model.user_tower.embedding.weight))
        self.assertFalse(torch.equal(old_film, model.item_tower.embedding.weight))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            model.save(path, 1, 42)
            restored = TwoTower(baseline, train)
            restored.load(path)
            self.assertEqual(restored.scores(1), model.scores(1))
            changed_train = [(1, 1, 4, 1), (2, 2, 5, 2)]
            other = TwoTower(MovieRecommender(movies, changed_train), changed_train)
            with self.assertRaisesRegex(ValueError, "не соответствуют"):
                other.load(path)
