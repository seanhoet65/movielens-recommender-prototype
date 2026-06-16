import numpy as np
import pandas as pd
from math import log2
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Ranking / list metrics
# ---------------------------------------------------------------------------

def _dcg(relevances: list, k: int) -> float:
    return sum(rel / log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def precision_at_k(rec_ids: list, relevant: set, k: int) -> float:
    return sum(1 for mid in rec_ids[:k] if mid in relevant) / k if k else 0.0


def recall_at_k(rec_ids: list, relevant: set, k: int) -> float:
    if not relevant:
        return np.nan
    return sum(1 for mid in rec_ids[:k] if mid in relevant) / len(relevant)


def ndcg_at_k(rec_ids: list, relevant: set, k: int, ratings_map: dict = None) -> float:
    if not relevant:
        return np.nan
    if ratings_map:
        rels = [ratings_map.get(mid, 0.0) for mid in rec_ids[:k]]
        ideal = sorted(ratings_map.values(), reverse=True)[:k]
    else:
        rels = [1.0 if mid in relevant else 0.0 for mid in rec_ids[:k]]
        ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else np.nan


def mrr_score(rec_ids: list, relevant: set, k: int) -> float:
    for rank, mid in enumerate(rec_ids[:k], start=1):
        if mid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------

def novelty_score(rec_ids: list, item_popularity: dict) -> float:
    """Mean self-information of recommended items. Higher = more novel."""
    scores = [-log2(max(item_popularity.get(mid, 1e-10), 1e-10)) for mid in rec_ids]
    return float(np.mean(scores)) if scores else np.nan


def diversity_score(rec_ids: list, item_matrix: np.ndarray, feature_idx: dict) -> float:
    """Mean pairwise cosine distance within the list. Higher = more diverse."""
    idxs = [feature_idx[mid] for mid in rec_ids if mid in feature_idx]
    if len(idxs) < 2:
        return 0.0
    vecs = item_matrix[idxs]
    sim = cosine_similarity(vecs)
    n = len(idxs)
    total = sum(1 - sim[i, j] for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) / 2
    return float(total / pairs) if pairs > 0 else 0.0


def popularity_bias_score(rec_ids: list, top_popular: set) -> float:
    """Fraction of recommendations from the top-10% most popular items."""
    if not rec_ids:
        return np.nan
    return sum(1 for mid in rec_ids if mid in top_popular) / len(rec_ids)


def serendipity_score(rec_ids: list, relevant: set, primitive_recs: set) -> float:
    """
    Fraction of recommendations that are both relevant and unexpected.
    primitive_recs: set of movie_ids that a naive baseline (e.g. Top-N) would have recommended.
    Serendipitous = relevant AND not in the primitive set.
    """
    if not rec_ids:
        return np.nan
    count = sum(1 for mid in rec_ids if mid in relevant and mid not in primitive_recs)
    return count / len(rec_ids)


# ---------------------------------------------------------------------------
# Full evaluation harness
# ---------------------------------------------------------------------------

def evaluate_recommender(
    recommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    rel_threshold: float = 4.0,
    sample_users: int = None,
    item_matrix: np.ndarray = None,
    feature_idx: dict = None,
    primitive_recs: dict = None,
) -> dict:
    """
    Run the full evaluation suite and return a dict of metric → value.

    item_matrix / feature_idx: optional, used for intra-list diversity.
    If not supplied, diversity is skipped.
    """
    n_users_train = train_df["userId"].nunique()
    item_counts = train_df.groupby("movieId")["rating"].count()
    item_popularity = (item_counts / n_users_train).to_dict()

    top10pct_n = max(1, int(len(item_counts) * 0.1))
    top_popular = set(item_counts.nlargest(top10pct_n).index)
    all_movies = set(train_df["movieId"])

    test_users = test_df["userId"].unique()
    if sample_users and len(test_users) > sample_users:
        rng = np.random.default_rng(42)
        test_users = rng.choice(test_users, sample_users, replace=False)

    mae_vals, rmse_vals = [], []
    prec_vals, rec_vals, ndcg_vals, mrr_vals = [], [], [], []
    nov_vals, div_vals, bias_vals = [], [], []
    ser_vals = []
    all_recommended: set = set()

    for uid in test_users:
        user_test = test_df[test_df["userId"] == uid]
        if user_test.empty:
            continue

        # Rating prediction
        y_true, y_pred = [], []
        for _, row in user_test.iterrows():
            try:
                pred = recommender.predict(int(uid), int(row["movieId"]))
                y_true.append(row["rating"])
                y_pred.append(np.clip(pred, 0.5, 5.0))
            except Exception:
                pass
        if y_true:
            mae_vals.append(mae(y_true, y_pred))
            rmse_vals.append(rmse(y_true, y_pred))

        # Recommendation list
        try:
            recs = recommender.recommend(int(uid), k)
        except Exception:
            continue
        if not recs:
            continue

        rec_ids = [mid for mid, _ in recs]
        all_recommended.update(rec_ids)

        relevant = set(user_test[user_test["rating"] >= rel_threshold]["movieId"])
        ratings_map = dict(zip(user_test["movieId"].astype(int), user_test["rating"]))

        prec_vals.append(precision_at_k(rec_ids, relevant, k))
        rv = recall_at_k(rec_ids, relevant, k)
        if not np.isnan(rv):
            rec_vals.append(rv)
        nv = ndcg_at_k(rec_ids, relevant, k, ratings_map)
        if not np.isnan(nv):
            ndcg_vals.append(nv)
        mrr_vals.append(mrr_score(rec_ids, relevant, k))
        if primitive_recs is not None:
            prim = set(primitive_recs.get(int(uid), []))
            srv = serendipity_score(rec_ids, relevant, prim)
            if not np.isnan(srv):
                ser_vals.append(srv)
        nov_vals.append(novelty_score(rec_ids, item_popularity))
        bias_vals.append(popularity_bias_score(rec_ids, top_popular))

        if item_matrix is not None and feature_idx is not None:
            div_vals.append(diversity_score(rec_ids, item_matrix, feature_idx))

    def _mean(lst):
        return float(np.nanmean(lst)) if lst else None

    return {
        "mae": _mean(mae_vals),
        "rmse": _mean(rmse_vals),
        "precision_at_k": _mean(prec_vals),
        "recall_at_k": _mean(rec_vals),
        "ndcg_at_k": _mean(ndcg_vals),
        "mrr": _mean(mrr_vals),
        "coverage": len(all_recommended) / len(all_movies) if all_movies else None,
        "novelty": _mean(nov_vals),
        "diversity": _mean(div_vals) if div_vals else None,
        "popularity_bias": _mean(bias_vals),
        "serendipity": _mean(ser_vals) if ser_vals else None,
        "k": k,
        "n_users": int(len(test_users)),
    }
