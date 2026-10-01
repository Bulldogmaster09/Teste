from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "volume_zscore_20d",
    "revenue_growth", "earnings_growth", "gross_margin",
    "debt_to_ebitda", "pe_ratio", "market_cap",
    "news_sentiment", "news_volume_zscore",
]


class FeatureBuilder:
    """
    Builds point-in-time features.

    Expected columns:
      date, ticker, close, volume, revenue_growth, earnings_growth,
      gross_margin, debt_to_ebitda, pe_ratio, market_cap,
      news_sentiment, news_volume

    The implementation deliberately uses only rolling/past observations
    when calculating market features, reducing look-ahead leakage.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values(["ticker", "date"]).reset_index(drop=True)

        g = x.groupby("ticker", group_keys=False)

        x["return_1d"] = g["close"].pct_change(1)
        x["return_5d"] = g["close"].pct_change(5)
        x["return_20d"] = g["close"].pct_change(20)

        x["volatility_20d"] = (
            g["close"].pct_change().rolling(20).std()
            .reset_index(level=0, drop=True)
        )

        rolling_volume = (
            g["volume"].rolling(20).mean()
            .reset_index(level=0, drop=True)
        )
        rolling_volume_std = (
            g["volume"].rolling(20).std()
            .reset_index(level=0, drop=True)
        )
        x["volume_zscore_20d"] = (
            (x["volume"] - rolling_volume) /
            rolling_volume_std.replace(0, np.nan)
        )

        rolling_news = (
            g["news_volume"].rolling(20).mean()
            .reset_index(level=0, drop=True)
        )
        rolling_news_std = (
            g["news_volume"].rolling(20).std()
            .reset_index(level=0, drop=True)
        )
        x["news_volume_zscore"] = (
            (x["news_volume"] - rolling_news) /
            rolling_news_std.replace(0, np.nan)
        )

        # Prediction targets are future returns, never used as input features.
        x["target_return_1d"] = g["close"].shift(-1) / x["close"] - 1.0
        x["target_return_3d"] = g["close"].shift(-3) / x["close"] - 1.0
        x["target_return_7d"] = g["close"].shift(-7) / x["close"] - 1.0

        return x

    @staticmethod
    def feature_columns() -> list[str]:
        return BASE_FEATURES.copy()
