"""InvestorMe hidden-factor financial intelligence prototype."""

from .model import InvestorMeModel, Prediction
from .similarity import SimilarityEngine
from .events import Event, EventEngine
from .features import FeatureBuilder

__all__ = [
    "InvestorMeModel",
    "Prediction",
    "SimilarityEngine",
    "Event",
    "EventEngine",
    "FeatureBuilder",
]
