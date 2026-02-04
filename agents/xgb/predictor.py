from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class XgbPredictor:
    def __init__(self, model_path: str) -> None:
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError("xgboost is not installed in the runtime.") from e

        self._xgb = xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_action(self, X: np.ndarray) -> np.ndarray:
        # 0=hold, 1=buy, 2=sell
        return self.model.predict(X).astype(np.int64)

    @staticmethod
    def decode_action(a: int) -> str:
        return {0: "hold", 1: "buy", 2: "sell"}.get(int(a), "hold")

