# MovieLens Recommender System — Individual Project
> ESADE MiBA, Recommender Systems, Prof. Marc Torrens
> Student: Sean Hoet

## What this project is

A progressive recommender system prototype built on the MovieLens dataset. Each algorithm is implemented as a standalone, comparable module. The final deliverable is a working prototype + slide deck.

**Course reference:** See `session_notes.md` in this directory for all course content, algorithms, formulas, and evaluation criteria as taught by Prof. Torrens.

---

## Project structure

```
Individual_ASNM/
├── CLAUDE.md               # This file — project context for Claude Code
├── session_notes.md        # Course content summary, updated weekly
├── data/                   # Raw and processed datasets (not committed to git)
│   ├── raw/                # Original MovieLens files
│   └── processed/          # Cleaned, split datasets
├── src/
│   ├── data/               # Loading, EDA, preprocessing
│   ├── recommenders/       # One file per algorithm
│   │   ├── nonpersonalised.py
│   │   ├── user_cf.py
│   │   ├── item_cf.py
│   │   ├── content_based.py
│   │   └── matrix_factorisation.py  # to be added
│   ├── evaluation/         # All evaluation metrics
│   └── ui/                 # Prototype interface
├── notebooks/              # Exploration and demos
├── results/                # Saved evaluation outputs for comparison
└── app.py                  # Entry point for the prototype UI
```

---

## Dataset

**Primary:** MovieLens Latest Small (`ml-latest-small`)
- Source: https://grouplens.org/datasets/movielens/latest/
- Files: `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`
- ~100k ratings, ~9k movies, ~600 users
- Ratings: explicit, scale 0.5–5.0

**Do not use** the full MovieLens dataset during development — iterate fast on the small set.

---

## Data contracts between modules

Every recommender module must implement the same interface so they can be swapped and compared:

```python
class BaseRecommender:
    def fit(self, ratings_df: pd.DataFrame) -> None: ...
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        # returns list of (movie_id, predicted_score), sorted descending
        ...
    def predict(self, user_id: int, movie_id: int) -> float:
        # returns predicted rating for a single user-item pair
        ...
```

All recommenders must:
- Filter out already-seen items from recommendations by default.
- Accept a `movies_df` for metadata when needed (content-based).
- Return scores in a consistent range (normalise if needed).

---

## Evaluation setup

**Split strategy:** temporal split — train on older interactions, test on newer ones. Do NOT use random split (data leakage risk).

**Always run these baselines first:**
- Random recommender
- Most popular items
- User average rating
- Item average rating

**Metrics to compute for every algorithm:**

| Metric | Type | Priority |
|---|---|---|
| MAE, RMSE | Rating prediction | Medium |
| Precision@10, Recall@10 | List quality | High |
| NDCG@10 | Ranking quality | High |
| MRR | First hit position | Medium |
| Coverage | Catalog reach | High |
| Novelty | Long tail reach | High |
| Diversity | Intra-list variety | High |
| Popularity bias | Fairness | High |

Prof. Torrens weights evaluation at 30% of the grade and explicitly emphasises beyond-accuracy metrics (diversity, novelty, bias). Do not optimise only for NDCG.

Store all evaluation results to `results/` as JSON so methods can be compared side by side.

---

## Algorithm implementation notes

**Non-personalised:**
- Implement: Top-N (by count), weighted average rating (Bayesian), product associations (lift metric).
- Bayesian formula: WR(j) = (v/(v+m)) * U(j) + (m/(v+m)) * C

**User-based CF:**
- Similarity: Pearson correlation over co-rated items only.
- Prediction: deviation-from-mean formula (normalised).
- Neighbourhood: start with top-k (k=50), experiment with combined strategy.

**Item-based CF:**
- Similarity: adjusted cosine (normalise per user before computing cosine).
- More stable than user-based; cache item-item similarity matrix.

**Content-based filtering:**
- Item features: genres (from `movies.csv`), user tags (from `tags.csv`), year extracted from title.
- Apply TF-IDF weighting to tag features.
- User profile: weighted sum of item vectors (weight by centred rating).
- Prediction: cosine similarity between user profile vector and item vector.

**Matrix factorisation (upcoming):**
- Implement SVD or ALS.
- Use surprise or implicit library if time-constrained.

---

## UI / Prototype

- Simple interface: user selects a user ID, selects recommender method, sees top-10 recommendations with movie title and score.
- Must allow switching between methods to enable direct comparison (key for slide deck).
- Streamlit is the recommended framework — fast to build, no frontend knowledge needed.
- UX is 20% of the grade. Keep it clean: show method name, metrics summary, and recommendation list.

---

## Conventions

- Python 3.10+
- Dependencies: pandas, numpy, scikit-learn, scipy, streamlit. Add others only if needed.
- All scripts runnable from `Individual_ASNM/` as working directory.
- No Jupyter notebooks in final deliverable — notebooks are for exploration only.
- Comments in English.
- Commit after each algorithm is working and evaluated.

---

## Grading weights (from assignment)

| Component | Weight |
|---|---|
| Technical method implementations | 50% |
| Evaluation | 30% |
| User experience | 20% |

---

## Progress tracker

- [x] Dataset download and folder setup
- [x] EDA and preprocessing (`src/data/loader.py`)
- [x] Non-personalised recommender (`src/recommenders/nonpersonalised.py` — TopN, Bayesian, Associations)
- [x] User-based CF (`src/recommenders/user_cf.py` — Pearson, deviation-from-mean, vectorised)
- [x] Item-based CF (`src/recommenders/item_cf.py` — adjusted cosine, cached top-k)
- [x] Content-based filtering (`src/recommenders/content_based.py` — TF-IDF tags + genres)
- [x] Matrix factorisation (`src/recommenders/matrix_factorisation.py` — truncated SVD)
- [x] Hybrid (SVD + Content-Based, weighted combination)
- [x] Evaluation harness with all metrics (`src/evaluation/metrics.py`)
- [x] Pre-computed results saved to `results/*.json`
- [x] Prototype UI (`app.py` — Streamlit, run with `streamlit run app.py`)
- [x] Baseline recommenders (Random, User Average, Item Average)
- [x] Serendipity metric added to evaluation harness
- [x] Live serendipity/novelty slider on Hybrid method
- [x] Netflix-style genre row tab
- [x] Cold start sensitivity tab
- [x] K-sensitivity analysis for CF methods
- [ ] Slide deck (file exists: Sean_recommender_slides.pptx — needs review)
