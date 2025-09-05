import os
import threading
from typing import Dict, List, Tuple

import torch
import numpy as np


class EnsemblePredictor:
    def __init__(self, model_base_dir: str | None = None):
        self.model_base_dir = model_base_dir or os.environ.get("MODEL_BASE_DIR", "/workspace/models")
        self._model_cache_lock = threading.Lock()
        self._model_cache: Dict[str, torch.nn.Module] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.model_base_dir, path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        resolved = self._resolve_path(model_path)
        key = os.path.abspath(resolved)
        with self._model_cache_lock:
            if key in self._model_cache:
                return self._model_cache[key]
            # Загружаем веса; ожидаем torch.save(state_dict)
            state = torch.load(resolved, map_location=self._device)
            # Простой каркас сети должен совпадать с проектным DQNN; здесь минимальный заглушечный head
            # В реальном проекте импортируйте DQNN и создайте по manifest/gym_snapshot
            from agents.vdqn.dqnn import DQNN  # предполагается доступен через volume
            # NOTE: параметры сети должны соответствовать тренировочным; здесь полагаемся на веса и совместимость
            model = DQNN(input_dim=state.get('_input_dim', 128), action_space=3)
            missing, unexpected = model.load_state_dict(state, strict=False)
            model.to(self._device)
            model.eval()
            self._model_cache[key] = model
            return model

    @torch.inference_mode()
    def predict_single(self, model_path: str, state: List[float]) -> Tuple[str, float, List[float]]:
        model = self._load_model(model_path)
        x = torch.tensor(np.array(state, dtype=np.float32), device=self._device).unsqueeze(0)
        q_values = model(x).detach().cpu().numpy()[0].tolist()
        action_idx = int(np.argmax(q_values))
        # Простейшая уверенность как softmax max prob
        exps = np.exp(q_values - np.max(q_values))
        probs = exps / exps.sum()
        confidence = float(np.max(probs))
        action = ["hold", "buy", "sell"][action_idx] if len(q_values) >= 3 else str(action_idx)
        return action, confidence, q_values

    def vote(self, predictions: List[str], consensus_pct: int) -> Tuple[str, Dict[str, int], int]:
        counts: Dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        for p in predictions:
            counts[p] = counts.get(p, 0) + 1
        total = max(1, sum(counts.values()))
        threshold_votes = int(np.ceil(total * (consensus_pct / 100.0)))
        # Приоритет: buy/sell, иначе hold
        if counts["buy"] >= threshold_votes:
            return "buy", counts, threshold_votes
        if counts["sell"] >= threshold_votes:
            return "sell", counts, threshold_votes
        return "hold", counts, threshold_votes
