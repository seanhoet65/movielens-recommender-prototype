import numpy as np
import pandas as pd
from .base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Recommends n random unseen items. Absolute lower bound baseline."""
    name = "Random"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.all_movies = list(set(ratings_df["movieId"]))
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.global_mean = ratings_df["rating"].mean()
        self._rng = np.random.default_rng(42)

    def predict(self, user_id, movie_id):
        return float(self._rng.uniform(0.5, 5.0))

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        unseen = [m for m in self.all_movies if m not in seen]
        if not unseen:
            return []
        chosen = self._rng.choice(unseen, min(n, len(unseen)), replace=False)
        scores = self._rng.uniform(0.5, 5.0, len(chosen))
        result = list(zip(chosen.tolist(), scores.tolist()))
        result.sort(key=lambda x: -x[1])
        return result


class UserAverageRecommender(BaseRecommender):
    """Predicts the user's own mean rating for every item. Simple personalised baseline."""
    name = "User Average"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.user_means = ratings_df.groupby("userId")["rating"].mean()
        self.global_mean = ratings_df["rating"].mean()
        # Use item means as tie-breaker when all predicted scores are the same
        self.item_means = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False)

    def predict(self, user_id, movie_id):
        return float(self.user_means.get(user_id, self.global_mean))

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        predicted = float(self.user_means.get(user_id, self.global_mean))
        # All get same predicted score; rank by item mean as tie-breaker
        candidates = [(mid, predicted) for mid in self.item_means.index if mid not in seen]
        return candidates[:n]


class ItemAverageRecommender(BaseRecommender):
    """Predicts each item's mean rating regardless of user. Strong cold-start baseline."""
    name = "Item Average"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.item_means = ratings_df.groupby("movieId")["rating"].mean().sort_values(ascending=False)
        self.global_mean = ratings_df["rating"].mean()

    def predict(self, user_id, movie_id):
        return float(self.item_means.get(movie_id, self.global_mean))

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        return [
            (mid, float(score))
            for mid, score in self.item_means.items()
            if mid not in seen
        ][:n]
