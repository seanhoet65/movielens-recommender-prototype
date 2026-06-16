# Recommender Systems — Session Notes
> Course by Prof. Marc Torrens, ESADE. Updated weekly. Used as context for Claude Code.

---

## Session 1: Introduction to Recommender Engines

**Core framing:**
- The Long Tail (Chris Anderson, 2004): internet reduced production and distribution costs, exploding the number of available items. Most items sit in the long tail — low popularity but collectively significant.
- Paradox of Choice (Barry Schwartz, 2004): more options reduce satisfaction beyond a point. Recommenders solve this.
- Filter Bubble (Eli Pariser, 2011): personalisation can trap users in echo chambers. Good design gives users control and explanations.

**Search vs Recommendation:**
- Search: user has an explicit query, system filters to match it.
- Recommendation: system infers interest from preferences, no explicit query needed.
- In practice these overlap — most products combine both.
- Recommendation is NOT always personalisation (e.g. editorial picks, trending lists).

**Personalisation spectrum:**
- Non-personalised search → personalised search → non-personalised recommendation → personalised recommendation.

**Taxonomy of recommender systems (Konstan & Ekstrand):**
- Domain, Purpose, Context, Whose opinions, Personalisation level, Privacy, Interface, Algorithm.
- No single best method — depends on domain, goal, data.

**Professor's view:**
- Personalisation is great but must include user control and explanations.
- Accuracy is not enough: diversity, novelty, and avoiding bias matter.

---

## Session 2: Non-Personalised Recommendations

**Core concept:**
- When user is unknown, assume they are like the majority.
- Best fallback strategy; also solves the cold-start problem.

**Four methods:**
1. **Editorial content** — human curated (e.g. "Staff picks"). High quality, low scalability.
2. **Top-N** — most sold, most liked, trending, newest. Simple, widely used.
3. **Average ratings** — mean rating per item. Watch out for popularity bias.
4. **Product associations** — "people who liked X also liked Y."

**Average rating formula:**
- Simple: U(j) = (1/n) * sum of ratings for item j
- Weighted scoring (Bayesian): WR(j) = (v/(v+m)) * U(j) + (m/(v+m)) * C
  - v = number of ratings for item j, m = minimum ratings threshold, C = global mean
- Normalised version accounts for users rating on different scales (deviation from user mean).

**Product associations:**
- Basic: P(Y | X) = count(X and Y) / count(X)
- Normalised (lift): P(Y | X) / P(Y) = P(X and Y) / [P(X) * P(Y)]
- Lift > 1 means X and Y co-occur more than by chance.

**Key insight:** There is no best method — choice depends on domain, goal, and data availability.

---

## Session 3: Collaborative Filtering

**Formal definition:**
- Utility function u: C × S → R (users × items → rating space).
- Goal: for each user c, find items s that maximise u. Unknown ratings are predicted from known ones.
- Framed as a supervised learning problem over the user-item matrix.

**User-based CF:**
- Assumption: past agreement predicts future agreement.
- Steps: collect ratings → find similar users (neighbours) → predict from neighbour ratings.
- Prediction formula (personalised, normalised):
  S(u, i) = r̄_u + [Σ_v (r_vi − r̄_v) * w_uv] / Σ_v |w_uv|
- Normalisation accounts for users rating on different scales.

**Similarity functions (w_uv):**
- Pearson correlation (preferred): measures agreement on co-rated items; higher weight when users agree on extremes.
  sim(u,v) = Σ(r_ui − r̄_u)(r_vi − r̄_v) / [σ_u * σ_v]
- Also: Euclidean distance, cosine, Jaccard (for unary ratings), Manhattan.
- Take baseline metric and challenge it iteratively.

**Neighbourhood selection (key design decision):**
- All users: most accurate but O(n²·m) compute cost.
- Top-k most similar: manageable but may miss diversity.
- Combined: e.g. 200 random + top-50 most similar.

**Item-based CF:**
- Compute similarity between items based on their rating vectors (not users).
- Predict: S(u,i) = r̄_i + [Σ_j (r_uj − r̄_j) * w_ij] / Σ_j |w_ij| where j ∈ neighbours of i.
- Similarity: vector correlation, cosine, adjusted cosine (normalises per user).

**User-based vs Item-based:**
| | User-based | Item-based |
|---|---|---|
| Discovery | Better | Worse (obvious) |
| Compute cost | Expensive O(n²m) | More stable, cacheable |
| Sparsity | Problem | More robust |
| Best when | More items than users | More users than items |

**Pros/Cons of CF overall:**
- Pro: content-agnostic, no item metadata needed.
- Con: sparsity, cold start for new users/items, items must be in a shared catalog.

---

## Session 4/5: Content-Based Filtering

**Core idea:**
- Recommend items similar in content to what the user has liked before.
- Uses item features (metadata, tags, text) rather than other users' opinions.

**Two design questions:**
A. How to model items as feature vectors?
B. How to model user preferences as feature vectors?

**Item modelling approaches:**
| Approach | Example | Advantage | Limitation |
|---|---|---|---|
| Manual taxonomy | Genre, director, language | Interpretable | Hard to scale |
| User-generated tags | "dark", "funny", "road trip" | Rich, flexible | Noisy |
| Text features | Plot summary, reviews | Scalable | Needs NLP preprocessing |
| Numerical features | Year, runtime, budget | Easy to compute | May miss meaning |

**User preference modelling:**
- Explicit: ask the user their preferences directly. Clear signal but users resist.
- Implicit from ratings: aggregate item features weighted by ratings.
  - Centred version: weight by (rating − user mean) to account for rating scale differences.
- Implicit behaviour: clicks, watches, purchases (easy to collect but ambiguous).
- Negative behaviour: skips, dislikes (very useful but often unavailable).

**Vector space model:**
- Items and user profiles are vectors in keyword/feature space.
- Similarity between item and user profile = cosine similarity.
- cos(θ) = (x · y) / (||x|| * ||y||) — ranges from -1 to 1.

**TF-IDF weighting:**
- Rare, specific features are stronger signals than common ones.
- TF-IDF = term frequency * log(total docs / docs containing term).
- IDF("the") ≈ 0 (useless); IDF("stethoscope") is high (specific).
- Apply to item feature vectors before computing cosine similarity.

**Prediction:**
- predicted_rating(user, item) = cosine(user_profile_vector, item_feature_vector)

**Hybrid approaches (introduced in this session):**
- Weighting: combine scores from multiple recommenders P = w1*p1 + w2*p2.
- Mixed: show recommendations from multiple sources in one list.
- Switching: use different recommender based on context or user profile.
- Cascading/Concatenating: one recommender filters, another re-ranks.
- Cold start is often solved by switching to content-based or non-personalised until enough ratings exist.

---

## Session 6: Evaluation

**Core message:**
- Predicting ratings accurately is not sufficient. The question is: are we recommending useful items?
- Evaluation has three modes: offline, online (A/B testing), and user studies.

**Offline evaluation (primary for this project):**
- Split historical data into train / validation / test sets.
- Temporal split preferred over random split — avoids using future data to predict the past.
- Do not recommend already-seen items (except music where repeat is valid).

**Rating prediction metrics:**
- MAE = mean(|predicted − actual|). Easy to interpret.
- RMSE = sqrt(mean((predicted − actual)²)). Penalises large errors more.

**Recommendation list metrics (more important for this project):**
- Precision@k: fraction of top-k recommendations that are relevant.
- Recall@k: fraction of all relevant items that appear in top-k.
- DCG@k: rewards relevant items appearing higher in the list. DCG += relevance / log2(rank+1).
- NDCG@k: DCG normalised by ideal DCG (IDCG). Range [0,1].
- MRR: reciprocal rank of the first relevant item. Cares only about where the first hit is.
- Spearman rank correlation: compares recommender ranking to true preference ranking.
- FCP (Fraction of Concordant Pairs): proportion of item pairs ranked consistently with user preferences.

**Beyond accuracy — professor emphasises these strongly:**
- Diversity: are recommended items different from each other?
- Novelty: are items non-obvious? Proxy = mean(-log2(popularity)).
- Coverage: what fraction of the catalog can the system recommend?
- Popularity bias: what % of recommendations come from the top 10% most popular items?
- Serendipity: unexpected AND useful recommendations.

**Key evaluation pitfalls (from slides):**
1. Recommending already-seen items — filter them out.
2. Random split instead of temporal split — causes data leakage.
3. Ignoring baselines — always compare to random, most-popular, user-average, item-average.
4. Optimising a single metric — trade-offs between precision, diversity, novelty.

**Always-run baselines:**
- Random recommender
- Most popular items
- User average rating
- Item average rating

---

## To Add (future sessions)
- Session 7: Matrix Factorisation (SVD, ALS, NMF)
