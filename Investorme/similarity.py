from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors


class SimilarityEngine:
    """
    Finds historical states similar to the current market/company state.

    Similarity is computed on standardized numerical features. The returned
    neighbors are historical rows, which can then be associated with their
    future realized returns.

    IMPORTANT: callers must only fit/search against rows that existed before
    the prediction timestamp.
    """

    def __init__(self, n_neighbors: int = 100):
        self.n_neighbors = n_neighbors
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean",
        )
        self.fitted = False

    def fit(self, history: pd.DataFrame, features: list[str]) -> "SimilarityEngine":
        self.features = features
        matrix = self.imputer.fit_transform(history[features])
        matrix = self.scaler.fit_transform(matrix)
        self.nn.fit(matrix)
        self.history_index = history.index.to_numpy()
        self.fitted = True
        return self

    def search(self, current_row: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("SimilarityEngine must be fitted first.")

        matrix = self.imputer.transform(current_row[self.features])
        matrix = self.scaler.transform(matrix)
        distances, positions = self.nn.kneighbors(matrix)

        result = history.iloc[positions[0]].copy()
        result["distance"] = distances[0]
        result["similarity"] = 1.0 / (1.0 + result["distance"])
        return result.sort_values("distance")
