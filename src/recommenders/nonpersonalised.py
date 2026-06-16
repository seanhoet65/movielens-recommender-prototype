import numpy as np
import pandas as pd
from .base import BaseRecommender


class TopNRecommender(BaseRecommender):
    name = "Top-N (Most Popular)"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.global_mean = ratings_df["rating"].mean()
        counts = ratings_df.groupby("movieId")["rating"].count()
        self.scores = counts.sort_values(ascending=False)

    def predict(self, user_id, movie_id):
        # TopN scores are raw counts — not rating predictions.
        # Fall back to global mean as a valid rating proxy.
        return float(self.global_mean)

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        return [
            (mid, float(score))
            for mid, score in self.scores.items()
            if mid not in seen
        ][:n]


class BayesianRecommender(BaseRecommender):
    name = "Bayesian Weighted Rating"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.global_mean = ratings_df["rating"].mean()
        stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
        C = self.global_mean
        m = stats["count"].quantile(0.5)  # median count as minimum threshold
        stats["wr"] = (
            stats["count"] / (stats["count"] + m) * stats["mean"]
            + m / (stats["count"] + m) * C
        )
        self.scores = stats["wr"].sort_values(ascending=False)

    def predict(self, user_id, movie_id):
        return float(self.scores.get(movie_id, self.global_mean))

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        return [
            (mid, float(score))
            for mid, score in self.scores.items()
            if mid not in seen
        ][:n]


class AssociationRecommender(BaseRecommender):
    name = "Product Associations (Lift)"

    def __init__(self, rating_threshold: float = 4.0):
        self.rating_threshold = rating_threshold

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.global_mean = ratings_df["rating"].mean()
        n_users = ratings_df["userId"].nunique()

        liked = ratings_df[ratings_df["rating"] >= self.rating_threshold]
        self.liked_by_user = liked.groupby("userId")["movieId"].apply(set).to_dict()

        # P(item) — fraction of users who liked each item
        item_support: dict[int, int] = {}
        for items in self.liked_by_user.values():
            for item in items:
                item_support[item] = item_support.get(item, 0) + 1
        self.item_prob = {k: v / n_users for k, v in item_support.items()}
        self.n_users = n_users

        # Index: item → set of users who liked it
        self.users_who_liked: dict[int, set] = {}
        for uid, items in self.liked_by_user.items():
            for mid in items:
                self.users_who_liked.setdefault(mid, set()).add(uid)

        # Per-user seed: highest-lift item the user has liked
        self.user_seed = {}
        for uid, items in self.liked_by_user.items():
            if items:
                self.user_seed[uid] = max(items, key=lambda x: self.item_prob.get(x, 0))

    def _lift_scores(self, seed: int) -> dict[int, float]:
        if seed not in self.users_who_liked:
            return {}
        seed_users = self.users_who_liked[seed]
        p_seed = self.item_prob.get(seed, 0)
        if p_seed == 0:
            return {}
        co_counts: dict[int, int] = {}
        for uid in seed_users:
            for mid in self.liked_by_user.get(uid, set()):
                if mid != seed:
                    co_counts[mid] = co_counts.get(mid, 0) + 1
        lifts = {}
        for mid, count in co_counts.items():
            p_item = self.item_prob.get(mid, 0)
            if p_item > 0:
                lifts[mid] = (count / self.n_users) / (p_seed * p_item)
        return lifts

    def predict(self, user_id, movie_id):
        # Lift values are not rating predictions — fall back to global mean.
        return float(self.global_mean)

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        seed = self.user_seed.get(user_id)
        if seed is None:
            return []
        lifts = self._lift_scores(seed)
        candidates = [(mid, lift) for mid, lift in lifts.items() if mid not in seen]
        candidates.sort(key=lambda x: -x[1])
        return candidates[:n]
