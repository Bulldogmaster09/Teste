from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass
class Event:
    ticker: str
    timestamp: str
    event_type: str
    sentiment: float
    importance: float
    novelty: float
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class EventEngine:
    """
    Lightweight event layer. In production, the text fields should be
    produced by a financial LM/NLP service. This class turns event records
    into numerical features that the prediction model can consume.
    """

    def aggregate(self, events: Iterable[Event], as_of: str) -> pd.DataFrame:
        as_of_ts = pd.Timestamp(as_of)
        rows = []

        for e in events:
            ts = pd.Timestamp(e.timestamp)
            if ts > as_of_ts:
                continue
            age_days = max((as_of_ts - ts).total_seconds() / 86400.0, 0.0)
            decay = np.exp(-age_days / 7.0)
            rows.append({
                "ticker": e.ticker,
                "event_weight": e.importance * e.novelty * decay,
                "event_sentiment_weighted": (
                    e.sentiment * e.importance * e.novelty * decay
                ),
                "event_count": 1.0,
            })

        if not rows:
            return pd.DataFrame(columns=[
                "ticker", "event_weight", "event_sentiment",
                "event_count"
            ])

        x = pd.DataFrame(rows)
        out = x.groupby("ticker", as_index=False).agg(
            event_weight=("event_weight", "sum"),
            event_sentiment=("event_sentiment_weighted", "sum"),
            event_count=("event_count", "sum"),
        )
        return out
