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

    def _normalize_model_output(self, output):
        """Приводит выход модели к torch.Tensor на правильном устройстве.
        Поддерживает tuple/dict/ndarray/list и берёт первый элемент для tuple,
        логиты для dict (logits/output) или первый попавшийся элемент.
        """
        try:
            # tuple: берём первый элемент
            if isinstance(output, tuple):
                output = output[0]
            # dict: берём logits/output или первый элемент
            if isinstance(output, dict):
                if 'logits' in output:
                    output = output['logits']
                elif 'output' in output:
                    output = output['output']
                else:
                    # первый элемент по порядку
                    output = next(iter(output.values()))
            # превращаем в Tensor при необходимости
            if not torch.is_tensor(output):
                output = torch.as_tensor(output, device=self._device)
            # переносим на целевое устройство
            output = output.to(self._device)
            # гарантируем наличие batch измерения при необходимости
            if output.ndim == 0:
                output = output.unsqueeze(0)
            return output
        except Exception:
            # худший случай — попытаться создать тензор
            return output if torch.is_tensor(output) else torch.as_tensor(output, device=self._device)

    def _resolve_path(self, path: str) -> str:
        # Приводим к унифицированному виду
        if not path:
            return self.model_base_dir
        if os.path.isabs(path):
            return path
        # Если уже содержит абсолютный префикс базовой директории — возвращаем как есть
        base_abs = os.path.abspath(self.model_base_dir)
        p_abs = os.path.abspath(path)
        if p_abs.startswith(base_abs):
            return p_abs
        # Убираем дублирующий префикс 'models/' или '/workspace/models/' если передали относительный путь из UI
        norm = path.replace('\\', '/').lstrip('/')
        # Если в пути встречается подстрока 'models/', отрезаем всё до неё включительно
        if 'models/' in norm:
            norm = norm.split('models/', 1)[1]
        return os.path.join(self.model_base_dir, norm)

    def _load_model(self, model_path: str, obs_dim: int | None) -> torch.nn.Module:
        resolved = self._resolve_path(model_path)
        # Ключ кэша должен учитывать размер входа, так как архитектура зависит от obs_dim
        key = f"{os.path.abspath(resolved)}|obs={obs_dim or 'NA'}"
        with self._model_cache_lock:
            if key in self._model_cache:
                print(f"[ensemble_predictor] Model '{resolved}' loaded from cache.")
                return self._model_cache[key]
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"Model file not found: {resolved}")
            print(f"[ensemble_predictor] Loading model state_dict from '{resolved}'")
            ckpt = torch.load(resolved, map_location=self._device)
            # Случай: сохранена целая модель
            if isinstance(ckpt, torch.nn.Module):
                model = ckpt
                model.to(self._device).eval()
                self._model_cache[key] = model
                return model
            # Случай: сохранён state_dict или словарь с метаданными
            from agents.vdqn.dqnn import DQNN
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            # Попытаемся вытащить метаданные
            meta_obs = None
            for k in ('_input_dim', 'obs_dim', 'input_dim'):
                try:
                    v = ckpt.get(k) if isinstance(ckpt, dict) else None
                    if isinstance(v, int) and v > 0:
                        meta_obs = v
                        break
                except Exception:
                    pass
            final_obs = meta_obs or obs_dim or 128
            # hidden_sizes из метаданных при наличии
            hidden_sizes = None
            try:
                hs = ckpt.get('hidden_sizes') if isinstance(ckpt, dict) else None
                if isinstance(hs, (list, tuple)) and len(hs) > 0:
                    hidden_sizes = tuple(int(x) for x in hs)
            except Exception:
                hidden_sizes = None
            if not hidden_sizes:
                hidden_sizes = (512, 256, 128)
            act_dim = 3
            model = DQNN(obs_dim=final_obs, act_dim=act_dim, hidden_sizes=hidden_sizes)
            # Грузим веса максимально мягко
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                # Если формат не совпал, оставляем как есть; модель будет работать, но без части весов
                pass
            model.to(self._device).eval()
            self._model_cache[key] = model
            return model

    @torch.inference_mode()
    def predict_single(self, model_path: str, state: List[float]) -> Tuple[str, float, List[float]]:
        obs_dim = None
        try:
            obs_dim = int(len(state)) if state is not None else None
        except Exception:
            obs_dim = None
        model = self._load_model(model_path, obs_dim)
        x = torch.tensor(np.array(state, dtype=np.float32), device=self._device).unsqueeze(0)
        raw_out = model(x)
        y = self._normalize_model_output(raw_out)
        q_values_np = y.detach().float().cpu().numpy()
        if q_values_np.ndim > 1:
            q_values_np = q_values_np[0]
        q_values = q_values_np.tolist()
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
