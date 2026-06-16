from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    name: str = "BaseRecommender"

    @abstractmethod
    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame = None,
        tags_df: pd.DataFrame = None,
    ) -> None:
        pass

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Returns list of (movie_id, predicted_score) sorted descending, excluding seen items."""
        pass

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Returns predicted rating for a single user-item pair."""
        pass
