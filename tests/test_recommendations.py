import unittest

from movie_assistant.recommendations import METHODS, MovieRecommender, temporal_split


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        movies = {1: {"id": 1, "genres": ["Comedy", "Drama"]},
                  2: {"id": 2, "genres": ["Comedy", "Drama"]},
                  3: {"id": 3, "genres": ["Comedy"]},
                  4: {"id": 4, "genres": ["Drama"]},
                  5: {"id": 5, "genres": ["Comedy", "Drama"]}}
        ratings = [(1, 1, 5, 1), (1, 3, 1, 2),
                   (2, 1, 5, 3), (2, 2, 5, 4), (2, 3, 5, 5), (2, 4, 5, 6)]
        self.model = MovieRecommender(movies, ratings)

    def test_and_and_or_with_shared_exclusions(self):
        for method in METHODS:
            with self.subTest(method=method):
                both, _ = self.model.recommend(1, method, genres=["Comedy", "Drama"])
                either, _ = self.model.recommend(1, method, genres=["Comedy", "Drama"], genre_mode="or")
                excluded, _ = self.model.recommend(1, method, genres=["Comedy", "Drama"], exclude_ids=[2])
                self.assertEqual({row["id"] for row in both}, {2})
                self.assertEqual({row["id"] for row in either}, {2, 4})
                self.assertEqual(excluded, [])

    def test_unknown_user_gets_popular_with_flag(self):
        popular, _ = self.model.recommend(999, "popular")
        for method in ("content", "collaborative"):
            with self.subTest(method=method):
                result, fallback = self.model.recommend(999, method)
                self.assertTrue(fallback)
                self.assertEqual(result, popular)

    def test_split_keeps_equal_timestamps_together(self):
        ratings = [(1, i, 5, time) for i, time in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 8, 10], 1)]
        train, test = temporal_split(ratings)
        self.assertEqual(len(train), 9)
        self.assertLess(train[-1][3], test[0][3])
