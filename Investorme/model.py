from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from .features import FeatureBuilder
from .similarity import SimilarityEngine
from .hidden_factors import HiddenFactorEngine


@dataclass
class Prediction:
    ticker: str
    as_of: str
    horizon_days: int
    expected_return: float
    probability_up: float
    lower_quantile: float
    upper_quantile: float
    similar_cases: int
    top_factors: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class InvestorMeModel:
    """
    Research-grade prototype for InvestorMe's prediction core.

    It combines:
      1. point-in-time feature engineering;
      2. nonlinear tabular prediction;
      3. historical-state similarity;
      4. hidden-factor discovery;
      5. empirical uncertainty from historical residuals/similar cases.

    This is a research model, not an automated trading system.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.features = FeatureBuilder()
        self.imputer = SimpleImputer(strategy="median")
        self.predictor = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        )
        self.hidden = HiddenFactorEngine(random_state=random_state)
        self.similarity = SimilarityEngine(n_neighbors=100)
        self.fitted = False

    def fit(self, training_df: pd.DataFrame, horizon_days: int = 7) -> "InvestorMeModel":
        x = self.features.transform(training_df)
        target = f"target_return_{horizon_days}d"
        cols = self.features.feature_columns()

        train = x.dropna(subset=[target]).copy()
        X = self.imputer.fit_transform(train[cols])
        y = train[target].astype(float)

        self.predictor.fit(X, y)
        self.hidden.fit(pd.DataFrame(X, columns=cols), y)

        # Similarity is fitted only on historical training states.
        self.similarity.fit(train, cols)

        residuals = y.to_numpy() - self.predictor.predict(X)
        self.residual_std = float(np.nanstd(residuals))
        self.horizon_days = horizon_days
        self.training = train
        self.fitted = True
        return self

    def predict(self, row: pd.DataFrame) -> Prediction:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        x = self.features.transform(row)
        cols = self.features.feature_columns()
        X = self.imputer.transform(x[cols])

        expected = float(self.predictor.predict(X)[0])

        # Similar historical states add an empirical distribution.
        comparable = self.similarity.search(x, self.training)
        comparable_returns = comparable[f"target_return_{self.horizon_days}d"].dropna()

        if len(comparable_returns) >= 10:
            empirical = comparable_returns.to_numpy()
            lower, upper = np.quantile(empirical, [0.10, 0.90])
            prob_up = float(np.mean(empirical > 0))
        else:
            sigma = max(self.residual_std, 1e-6)
            lower, upper = expected - 1.28 * sigma, expected + 1.28 * sigma
            prob_up = float(0.5 * (1.0 + math.erf(expected / (sigma * np.sqrt(2)))))

        # Blend model expectation with historical analogues, conservatively.
        if len(comparable_returns) >= 20:
            empirical_mean = float(comparable_returns.mean())
            expected = 0.70 * expected + 0.30 * empirical_mean

        top = self.hidden.importance().head(8)
        factors = [
            {"feature": r.feature, "importance": float(r.importance)}
            for r in top.itertuples()
        ]

        return Prediction(
            ticker=str(row.iloc[0]["ticker"]),
            as_of=str(pd.Timestamp(row.iloc[0]["date"]).date()),
            horizon_days=self.horizon_days,
            expected_return=expected,
            probability_up=prob_up,
            lower_quantile=float(lower),
            upper_quantile=float(upper),
            similar_cases=int(len(comparable_returns)),
            top_factors=factors,
        )
