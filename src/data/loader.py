import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "DATA" / "raw"


def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    return ratings, movies, tags


def temporal_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Per-user temporal split: last `test_frac` of each user's ratings go to test."""
    train_rows, test_rows = [], []
    for _, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        cutoff = max(1, int(len(group) * (1 - test_frac)))
        train_rows.append(group.iloc[:cutoff])
        if len(group) > cutoff:
            test_rows.append(group.iloc[cutoff:])
    train = pd.concat(train_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, test


def get_movie_display(movie_ids: list, movies_df: pd.DataFrame) -> pd.DataFrame:
    sub = movies_df[movies_df["movieId"].isin(movie_ids)].copy()
    sub["year"] = sub["title"].str.extract(r"\((\d{4})\)$").squeeze()
    sub["clean_title"] = sub["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    return sub.set_index("movieId")[["clean_title", "year", "genres"]]
