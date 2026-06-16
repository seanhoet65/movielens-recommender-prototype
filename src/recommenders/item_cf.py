import numpy as np
import pandas as pd
from .base import BaseRecommender


class ItemCFRecommender(BaseRecommender):
    name = "Item-Based CF (Adjusted Cosine)"

    def __init__(self, k: int = 50, min_ratings: int = 10):
        self.k = k
        self.min_ratings = min_ratings  # filter low-support items to keep sim matrix small

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.global_mean = ratings_df["rating"].mean()
        self.user_means = ratings_df.groupby("userId")["rating"].mean()
        self.item_means = ratings_df.groupby("movieId")["rating"].mean()
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.all_movies = set(ratings_df["movieId"])

        # Keep only items with enough ratings for stable similarity estimates
        counts = ratings_df.groupby("movieId")["rating"].count()
        active = set(counts[counts >= self.min_ratings].index)

        matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")
        self.matrix = matrix  # full matrix for predict()
        active_matrix = matrix[[c for c in matrix.columns if c in active]]

        # Adjusted cosine: centre per user, then cosine between item vectors
        centered = active_matrix.subtract(self.user_means, axis=0).fillna(0)
        item_vecs = centered.T.values  # (n_active_items, n_users)
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_vecs_norm = item_vecs / norms

        sim_matrix = item_vecs_norm @ item_vecs_norm.T  # (n_active, n_active)
        np.fill_diagonal(sim_matrix, 0.0)

        self.active_ids = list(active_matrix.columns)
        self.active_idx = {mid: i for i, mid in enumerate(self.active_ids)}

        # Store top-k similar items per item (among active items only)
        self.item_neighbours: dict[int, list[tuple[int, float]]] = {}
        for i, mid in enumerate(self.active_ids):
            top_k = np.argsort(-sim_matrix[i])[: self.k]
            self.item_neighbours[mid] = [
                (self.active_ids[j], float(sim_matrix[i, j])) for j in top_k
            ]

    def predict(self, user_id, movie_id):
        if user_id not in self.matrix.index:
            return float(np.clip(self.item_means.get(movie_id, self.global_mean), 0.5, 5.0))
        if movie_id not in self.item_neighbours:
            return float(np.clip(self.item_means.get(movie_id, self.global_mean), 0.5, 5.0))

        user_row = self.matrix.loc[user_id]
        r_bar_i = float(self.item_means.get(movie_id, self.global_mean))
        numerator = 0.0
        denominator = 0.0
        for j, w in self.item_neighbours[movie_id]:
            if j not in self.matrix.columns:
                continue
            r_uj = user_row.get(j, np.nan)
            if np.isnan(r_uj):
                continue
            r_bar_j = float(self.item_means.get(j, self.global_mean))
            numerator += (r_uj - r_bar_j) * w
            denominator += abs(w)

        if denominator == 0:
            return float(np.clip(r_bar_i, 0.5, 5.0))
        return float(np.clip(r_bar_i + numerator / denominator, 0.5, 5.0))

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        if user_id not in self.matrix.index:
            return []

        user_row = self.matrix.loc[user_id].dropna()
        numerator: dict[int, float] = {}
        denominator: dict[int, float] = {}

        for rated_mid, r_uj in user_row.items():
            if rated_mid not in self.item_neighbours:
                continue
            r_bar_j = float(self.item_means.get(rated_mid, self.global_mean))
            dev = r_uj - r_bar_j
            for target_mid, w in self.item_neighbours[rated_mid]:
                if target_mid in seen:
                    continue
                numerator[target_mid] = numerator.get(target_mid, 0.0) + dev * w
                denominator[target_mid] = denominator.get(target_mid, 0.0) + abs(w)

        results = []
        for mid, num in numerator.items():
            if denominator[mid] > 0:
                r_bar_i = float(self.item_means.get(mid, self.global_mean))
                score = r_bar_i + num / denominator[mid]
                results.append((mid, score))

        results.sort(key=lambda x: -x[1])
        return results[:n]
