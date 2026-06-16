import numpy as np
import pandas as pd
from .base import BaseRecommender
from .matrix_factorisation import MatrixFactorisationRecommender
from .content_based import ContentBasedRecommender


def _minmax_norm(scores: dict) -> dict:
    """Normalise a {item: score} dict to [0, 1]. Constant input -> all 1.0."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRecommender(BaseRecommender):
    """Weighted hybrid: alpha * SVD + (1 - alpha) * Content-Based.

    Each base model's scores are min-max normalised to [0, 1] over the
    candidate pool before being combined, so the two score scales are
    comparable. Demonstrates Session 5 hybrid strategies: trading the
    accuracy of matrix factorisation against the novelty/coverage of
    content-based filtering.
    """

    name = "Hybrid (SVD + Content)"

    def __init__(self, alpha: float = 0.6, candidate_pool: int = 200):
        self.alpha = alpha  # weight on SVD; (1 - alpha) on content-based
        self.candidate_pool = candidate_pool
        self.svd = MatrixFactorisationRecommender()
        self.content = ContentBasedRecommender()

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.global_mean = ratings_df["rating"].mean()
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        self.svd.fit(ratings_df, movies_df, tags_df)
        self.content.fit(ratings_df, movies_df, tags_df)

    def predict(self, user_id, movie_id):
        # Combine the two predicted ratings directly on the [0.5, 5] scale.
        s = self.svd.predict(user_id, movie_id)
        c = self.content.predict(user_id, movie_id)
        return float(self.alpha * s + (1 - self.alpha) * c)

    def recommend(self, user_id, n=10):
        # Pull a generous candidate pool from each base model, normalise each
        # model's scores to [0, 1] over the union, then combine.
        pool = self.candidate_pool
        svd_recs = dict(self.svd.recommend(user_id, pool))
        content_recs = dict(self.content.recommend(user_id, pool))
        if not svd_recs and not content_recs:
            return []

        svd_norm = _minmax_norm(svd_recs)
        content_norm = _minmax_norm(content_recs)

        candidates = set(svd_norm) | set(content_norm)
        combined = {}
        for mid in candidates:
            s = svd_norm.get(mid, 0.0)
            c = content_norm.get(mid, 0.0)
            combined[mid] = self.alpha * s + (1 - self.alpha) * c

        ranked = sorted(combined.items(), key=lambda x: -x[1])
        return [(mid, float(score)) for mid, score in ranked[:n]]
