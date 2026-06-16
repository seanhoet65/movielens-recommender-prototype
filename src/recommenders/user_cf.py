import numpy as np
import pandas as pd
from .base import BaseRecommender


class UserCFRecommender(BaseRecommender):
    name = "User-Based CF (Pearson)"

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.global_mean = ratings_df["rating"].mean()
        self.user_means = ratings_df.groupby("userId")["rating"].mean()
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()

        # Build user-item matrix, centre per user, compute vectorised Pearson
        matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")
        self.matrix = matrix
        self.users = list(matrix.index)
        self.movie_ids = list(matrix.columns)
        self.user_idx = {u: i for i, u in enumerate(self.users)}

        centered = matrix.subtract(self.user_means, axis=0).fillna(0).values  # (n_users, n_movies)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        C = centered / norms
        self.sim_matrix = C @ C.T  # Pearson correlation approximation
        np.fill_diagonal(self.sim_matrix, 0.0)

        # Precompute top-k neighbour indices per user (sorted by similarity)
        self._top_k_idx = np.argsort(-self.sim_matrix, axis=1)[:, :self.k]

    def _recommend_vectorised(self, user_id: int, n: int):
        if user_id not in self.user_idx:
            return []

        i = self.user_idx[user_id]
        neigh_idx = self._top_k_idx[i]
        neigh_sims = self.sim_matrix[i, neigh_idx]  # (k,)

        # Only use neighbours with positive similarity
        pos_mask = neigh_sims > 0
        neigh_idx = neigh_idx[pos_mask]
        neigh_sims = neigh_sims[pos_mask]
        if len(neigh_idx) == 0:
            return []

        r_bar_u = float(self.user_means[user_id])
        seen = self.seen.get(user_id, set())

        # Neighbour rating matrix subset: (n_neighbours, n_movies)
        neigh_ratings = self.matrix.iloc[neigh_idx].values  # raw ratings, NaN for unrated
        neigh_means = np.array([float(self.user_means.iloc[j]) for j in neigh_idx])
        deviations = neigh_ratings - neigh_means[:, np.newaxis]  # (k, n_movies)

        valid = ~np.isnan(deviations)
        w = neigh_sims[:, np.newaxis]  # (k, 1)

        numerator = np.where(valid, deviations * w, 0.0).sum(axis=0)
        denominator = np.where(valid, np.abs(w), 0.0).sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            predicted = np.where(denominator > 0, r_bar_u + numerator / denominator, r_bar_u)

        # Mask already-seen items
        seen_mask = np.array([mid in seen for mid in self.movie_ids], dtype=bool)
        predicted[seen_mask] = -np.inf

        top_idx = np.argsort(-predicted)[:n]
        return [
            (self.movie_ids[j], float(predicted[j]))
            for j in top_idx
            if predicted[j] > -np.inf
        ]

    def predict(self, user_id, movie_id):
        if user_id not in self.user_idx or movie_id not in self.matrix.columns:
            return float(np.clip(self.user_means.get(user_id, self.global_mean), 0.5, 5.0))

        i = self.user_idx[user_id]
        mid_pos = self.matrix.columns.get_loc(movie_id)
        neigh_idx = self._top_k_idx[i]
        neigh_sims = self.sim_matrix[i, neigh_idx]

        r_bar_u = float(self.user_means[user_id])
        numerator = 0.0
        denominator = 0.0
        for j, w in zip(neigh_idx, neigh_sims):
            if w <= 0:
                continue
            r_vj = self.matrix.iloc[j, mid_pos]
            if np.isnan(r_vj):
                continue
            r_bar_v = float(self.user_means.iloc[j])
            numerator += (r_vj - r_bar_v) * w
            denominator += abs(w)

        if denominator == 0:
            return float(np.clip(r_bar_u, 0.5, 5.0))
        return float(np.clip(r_bar_u + numerator / denominator, 0.5, 5.0))

    def recommend(self, user_id, n=10):
        return self._recommend_vectorised(user_id, n)
