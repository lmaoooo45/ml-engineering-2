import unittest

from movie_assistant.evaluation import ranking_metrics, relevant_movies
from movie_assistant.recommendations import MovieRecommender


class EvaluationTests(unittest.TestCase):
    def test_rank_matters_for_ndcg(self):
        first = ranking_metrics([7, 8], {7}, 2)
        second = ranking_metrics([8, 7], {7}, 2)
        self.assertEqual(first, {"precision": 0.5, "recall": 1.0, "ndcg": 1.0})
        self.assertEqual(second["precision"], first["precision"])
        self.assertEqual(second["recall"], first["recall"])
        self.assertLess(second["ndcg"], first["ndcg"])

    def test_unavailable_and_already_seen_films_are_excluded_from_targets(self):
        movies = {movie_id: {"id": movie_id, "genres": ["Comedy"]} for movie_id in (1, 2, 3)}
        baseline = MovieRecommender(movies, [(1, 1, 5, 1), (2, 2, 4, 2)])
        later = [(1, 1, 5, 3), (1, 2, 4, 4), (1, 3, 5, 5), (1, 3, 2, 6)]
        relevant, skipped = relevant_movies(baseline, later)
        self.assertEqual(relevant, {1: {2}})
        self.assertEqual(skipped, 2)
