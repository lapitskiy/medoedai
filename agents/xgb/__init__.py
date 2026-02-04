"""XGB (XGBoost) supervised agent for MedoedAI."""

from .config import XgbConfig  # noqa: F401
from .trainer import XgbTrainer  # noqa: F401
from .predictor import XgbPredictor  # noqa: F401

__all__ = ["XgbConfig", "XgbTrainer", "XgbPredictor"]

