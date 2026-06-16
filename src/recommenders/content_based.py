import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    name = "Content-Based Filtering"

    def fit(self, ratings_df, movies_df=None, tags_df=None):
        self.global_mean = ratings_df["rating"].mean()
        self.user_means = ratings_df.groupby("userId")["rating"].mean()
        self.seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()

        # Build item feature matrix for all rated movies
        rated_movies = set(ratings_df["movieId"])
        items = movies_df[movies_df["movieId"].isin(rated_movies)].copy()
        items = items.reset_index(drop=True)

        # Genre: one-hot binary
        genre_dummies = items["genres"].str.get_dummies(sep="|")
        genre_dummies.index = items["movieId"]

        # Tags: TF-IDF (max 150 features)
        if tags_df is not None and len(tags_df) > 0:
            movie_tags = (
                tags_df.groupby("movieId")["tag"]
                .apply(lambda x: " ".join(x.astype(str).str.lower()))
                .reset_index()
            )
            items = items.merge(movie_tags, on="movieId", how="left")
            items["tag"] = items["tag"].fillna("")
        else:
            items["tag"] = ""

        tfidf = TfidfVectorizer(max_features=150, sublinear_tf=True)
        tag_matrix = tfidf.fit_transform(items["tag"].values).toarray()
        tag_df = pd.DataFrame(
            tag_matrix,
            index=items["movieId"].values,
            columns=[f"tag_{t}" for t in tfidf.get_feature_names_out()],
        )

        # Combine genre + tag features
        feature_df = pd.concat([genre_dummies, tag_df], axis=1).fillna(0)
        self.movie_ids_ordered = list(feature_df.index)
        self.feature_idx = {mid: i for i, mid in enumerate(self.movie_ids_ordered)}

        raw = feature_df.values.astype(float)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_matrix = raw / norms  # (n_items, n_features), L2-normalised

        # Build user profiles: weighted sum of item vectors (weight = rating - user mean)
        self._build_user_profiles(ratings_df)

    def _build_user_profiles(self, ratings_df):
        self.user_profiles: dict[int, np.ndarray] = {}
        n_features = self.item_matrix.shape[1]

        for uid, group in ratings_df.groupby("userId"):
            r_bar = float(self.user_means[uid])
            profile = np.zeros(n_features)
            total_weight = 0.0
            for _, row in group.iterrows():
                mid = row["movieId"]
                if mid not in self.feature_idx:
                    continue
                weight = row["rating"] - r_bar
                profile += weight * self.item_matrix[self.feature_idx[mid]]
                total_weight += abs(weight)
            if total_weight > 0:
                profile /= total_weight
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
            self.user_profiles[uid] = profile

    def predict(self, user_id, movie_id):
        if user_id not in self.user_profiles or movie_id not in self.feature_idx:
            return self.global_mean
        sim = float(
            np.dot(self.user_profiles[user_id], self.item_matrix[self.feature_idx[movie_id]])
        )
        # cosine in [-1, 1] → rating scale [0.5, 5.0]
        return 0.5 + (sim + 1) / 2 * 4.5

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]
        unseen_mask = np.array([mid not in seen for mid in self.movie_ids_ordered])
        sims = self.item_matrix @ profile  # cosine similarity, all items
        sims[~unseen_mask] = -np.inf

        top_idx = np.argsort(-sims)[:n]
        return [
            (self.movie_ids_ordered[i], float(sims[i]))
            for i in top_idx
            if sims[i] > -np.inf
        ]
