from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class HiddenFactorEngine:
    """
    Discovers nonlinear feature interactions using a tree ensemble.

    This is intentionally a discovery component, not proof of causality.
    A factor is considered useful only after it survives out-of-sample and
    walk-forward validation.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=7,
            min_samples_leaf=15,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HiddenFactorEngine":
        self.features = list(X.columns)
        self.model.fit(X, y)
        return self

    def importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": self.features,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def interaction_candidates(self, X: pd.DataFrame, top_n: int = 10) -> list[tuple[str, str]]:
        imp = self.importance().head(top_n)["feature"].tolist()
        return [(a, b) for i, a in enumerate(imp) for b in imp[i + 1:]]
