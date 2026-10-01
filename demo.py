"""
Synthetic end-to-end demonstration.

Run:
    python demo.py

Replace the generated dataset with point-in-time market/fundamental/news
data before using the model for research.
"""

import numpy as np
import pandas as pd

from investorme.model import InvestorMeModel


def make_demo_data(seed=42):
    rng = np.random.default_rng(seed)
    tickers = [f"STK{i:03d}" for i in range(30)]
    dates = pd.bdate_range("2019-01-01", "2025-12-31")

    rows = []
    for ticker in tickers:
        price = 30 + rng.random() * 120
        base_growth = rng.normal(0.08, 0.04)
        for date in dates:
            hidden_regime = rng.normal(0, 1)
            volume = abs(rng.normal(1e6, 2e5))
            news_volume = max(0, rng.poisson(4 + 1.5 * abs(hidden_regime)))

            daily = (
                0.0002
                + 0.0015 * hidden_regime
                + rng.normal(0, 0.018)
            )
            price *= max(0.85, 1 + daily)

            rows.append({
                "date": date,
                "ticker": ticker,
                "close": price,
                "volume": volume,
                "revenue_growth": base_growth + rng.normal(0, .02),
                "earnings_growth": base_growth + rng.normal(0, .04),
                "gross_margin": 0.25 + rng.normal(0, .03),
                "debt_to_ebitda": max(0.1, 1.8 + rng.normal(0, .5)),
                "pe_ratio": max(3, 18 + rng.normal(0, 4)),
                "market_cap": price * 1e7,
                "news_sentiment": np.tanh(hidden_regime / 2 + rng.normal(0, .5)),
                "news_volume": news_volume,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = make_demo_data()

    cutoff = pd.Timestamp("2025-06-30")
    train = data[data["date"] <= cutoff].copy()
    current = data[data["date"] == data["date"].max()].head(1).copy()

    model = InvestorMeModel()
    model.fit(train, horizon_days=7)

    # For a real system, the current row must have all point-in-time inputs
    # available at the prediction timestamp.
    prediction = model.predict(current)

    print(prediction.to_dict())
