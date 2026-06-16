import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from .base import BaseRecommender


class MatrixFactorisationRecommender(BaseRecommender):
    """Truncated-SVD latent factor model.

    Builds a dense user-item matrix, mean-centres each user, factorises with
    scipy.sparse.linalg.svds (k latent factors) and reconstructs a full
    predicted-rating matrix. Captures hidden user/item structure from sparse data.
    """

    name = "Matrix Factorisation (SVD)"

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.global_mean = ratings_df["rating"].mean()
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()

        # Build dense user-item matrix (users x movies), missing -> 0
        matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")
        self.users = list(matrix.index)
        self.movie_ids = list(matrix.columns)
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        self.movie_pos = {m: j for j, m in enumerate(self.movie_ids)}

        # Per-user mean over observed ratings only (ignore the implicit zeros)
        self.user_means = matrix.mean(axis=1).values  # (n_users,)
        filled = matrix.fillna(0.0).values  # (n_users, n_movies)
        # Mean-centre only where a rating exists; zeros stay at -mean otherwise.
        # Centring the observed entries and leaving missing at 0 keeps the matrix
        # sparse-friendly while removing per-user rating-scale bias.
        mask = matrix.notna().values
        centred = np.where(mask, filled - self.user_means[:, None], 0.0)

        # k must be < min(matrix dimensions)
        k = min(self.k, min(centred.shape) - 1)
        U, sigma, Vt = svds(centred, k=k)
        sigma = np.diag(sigma)

        # Reconstruct predicted (centred) ratings, then add user means back
        self.pred_matrix = (U @ sigma @ Vt) + self.user_means[:, None]

    def predict(self, user_id, movie_id):
        if user_id not in self.user_idx or movie_id not in self.movie_pos:
            return float(self.global_mean)
        val = self.pred_matrix[self.user_idx[user_id], self.movie_pos[movie_id]]
        return float(np.clip(val, 0.5, 5.0))

    def recommend(self, user_id, n=10):
        if user_id not in self.user_idx:
            return []
        i = self.user_idx[user_id]
        seen = self.seen.get(user_id, set())
        preds = self.pred_matrix[i].copy()
        seen_mask = np.array([m in seen for m in self.movie_ids], dtype=bool)
        preds[seen_mask] = -np.inf
        top_idx = np.argsort(-preds)[:n]
        return [
            (self.movie_ids[j], float(np.clip(preds[j], 0.5, 5.0)))
            for j in top_idx
            if preds[j] > -np.inf
        ]
