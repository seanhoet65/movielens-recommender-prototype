import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_data, temporal_split
from src.recommenders.baselines import RandomRecommender, UserAverageRecommender, ItemAverageRecommender
from src.recommenders.nonpersonalised import TopNRecommender, BayesianRecommender, AssociationRecommender
from src.recommenders.user_cf import UserCFRecommender
from src.recommenders.item_cf import ItemCFRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.matrix_factorisation import MatrixFactorisationRecommender
from src.recommenders.hybrid import HybridRecommender
from src.evaluation.metrics import evaluate_recommender

RESULTS_DIR = Path(__file__).parent / "results"

print("Loading data...")
ratings, movies, tags = load_data()
train, test = temporal_split(ratings)
print(f"Train: {len(train)}, Test: {len(test)}")

# All models: (filename_key, model_instance)
model_specs = [
    ("random",                   RandomRecommender()),
    ("user_average",             UserAverageRecommender()),
    ("item_average",             ItemAverageRecommender()),
    ("topn_most_popular",        TopNRecommender()),
    ("bayesian_weighted_rating", BayesianRecommender()),
    ("product_associations",     AssociationRecommender()),
    ("userbased_cf",             UserCFRecommender(k=50)),
    ("itembased_cf",             ItemCFRecommender(k=50)),
    ("contentbased",             ContentBasedRecommender()),
    ("matrix_factorisation",     MatrixFactorisationRecommender(k=50)),
    ("hybrid",                   HybridRecommender(alpha=0.6)),
]

print("Fitting all models...")
for key, model in model_specs:
    t0 = time.time()
    model.fit(train, movies, tags)
    print(f"  Fitted {key} in {time.time()-t0:.1f}s")

# Extract item_matrix and feature_idx from Content-Based for diversity scoring
cb_model = next(m for k, m in model_specs if k == "contentbased")
item_matrix = cb_model.item_matrix
feature_idx = cb_model.feature_idx

# Build primitive_recs cache (TopN recommendations per user)
topn_model = next(m for k, m in model_specs if k == "topn_most_popular")
print("Building primitive_recs cache...")
primitive_recs_cache = {}
for uid in test["userId"].unique():
    recs = topn_model.recommend(int(uid), 10)
    primitive_recs_cache[int(uid)] = [mid for mid, _ in recs]

print(f"Primitive recs built for {len(primitive_recs_cache)} users")

# Evaluate all models
print("Running evaluations (sample_users=100 per model)...")
all_results = {}
for key, model in model_specs:
    print(f"  Evaluating {key}...")
    t0 = time.time()
    results = evaluate_recommender(
        model, train, test,
        k=10,
        sample_users=100,
        item_matrix=item_matrix,
        feature_idx=feature_idx,
        primitive_recs=primitive_recs_cache,
    )
    elapsed = time.time() - t0
    all_results[key] = results
    out_path = RESULTS_DIR / f"{key}.json"
    out_path.write_text(json.dumps(results, indent=2))
    ser = results.get("serendipity")
    ser_str = f"{ser:.4f}" if ser is not None else "None"
    print(f"    Done in {elapsed:.1f}s | precision@10={results.get('precision_at_k', 0):.4f} | serendipity={ser_str}")

print("\nAll evaluations complete.")
