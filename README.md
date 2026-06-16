# MovieLens Recommender Prototype

### Live demo: [movielens-recommender-prototype.streamlit.app](http://movielens-recommender-prototype.streamlit.app/)

> GitHub repo: [github.com/seanhoet65/movielens-recommender-prototype](https://github.com/seanhoet65/movielens-recommender-prototype)

---

A web app that recommends movies to users based on their past ratings. Built as an individual project for the Recommender Systems course at ESADE MiBA.

## What it does

You pick a user from the MovieLens dataset and a recommendation algorithm, and the app shows you their top 10 personalised movie recommendations — along with posters, scores, and a summary of how well the algorithm performed.

You can switch between algorithms to compare them side by side, which is the main point of the prototype.

## Algorithms included

| Algorithm | How it works |
|---|---|
| Top Popular | Recommends the most-rated movies overall |
| Bayesian Rating | Like Top Popular, but adjusts for movies with very few ratings |
| User-based CF | Finds users with similar taste and recommends what they liked |
| Item-based CF | Finds movies similar to ones you've already rated highly |
| Content-based | Matches movies based on genres and tags |
| Matrix Factorisation | Learns hidden patterns in the ratings data (SVD) |
| Hybrid | Combines Matrix Factorisation + Content-based |

## Dataset

Uses the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset — about 100,000 ratings from 600 users across 9,000 movies.

## How to run it locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Add your TMDB API key (for movie posters) to `.streamlit/secrets.toml`:
   ```
   TMDB_API_KEY = "your_key_here"
   ```
   Get a free key at [themoviedb.org](https://www.themoviedb.org/settings/api). The app still works without it — posters just won't load.

3. Start the app:
   ```
   streamlit run app.py
   ```

## Project structure

```
app.py              # Main Streamlit app
src/
  recommenders/     # One file per algorithm
  evaluation/       # Metrics (NDCG, precision, diversity, novelty, etc.)
  data/             # Data loading and preprocessing
DATA/raw/           # MovieLens CSV files
results/            # Pre-computed evaluation results (JSON)
```
