from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from .config import XgbConfig
from .features import build_xgb_dataset


def _symbol_code(sym: str) -> str:
    s = (sym or "").strip().upper().replace("/", "")
    for suf in ("USDT", "USD", "USDC", "BUSD", "USDP"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s or "UNKNOWN"


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class XgbTrainer:
    def __init__(self, cfg: Optional[XgbConfig] = None) -> None:
        self.cfg = cfg or XgbConfig()

    def train(self, symbol: str, dfs: Dict[str, Any], result_root: str = "result") -> Dict[str, Any]:
        started = time.time()
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "xgboost is not installed in the runtime. Install it inside the container (pip install xgboost)."
            ) from e

        X, y, meta, aux = build_xgb_dataset(dfs, self.cfg)
        task = (self.cfg.task or "directional").strip().lower()
        is_binary = task.startswith("entry") or task.startswith("exit")
        n = len(y)
        val_n = max(1, int(n * float(self.cfg.val_ratio)))
        train_n = max(1, n - val_n)
        X_tr, y_tr = X[:train_n], y[:train_n]
        X_va, y_va = X[train_n:], y[train_n:]

        # Class weights to reduce imbalance
        num_classes = 2 if is_binary else 3
        counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
        w = inv[y_tr]
        w = (w / np.mean(w)) if np.mean(w) > 0 else None

        # scale_pos_weight: auto (-1) = count_neg/count_pos
        spw = float(self.cfg.scale_pos_weight)
        if spw == 0:
            raise ValueError("scale_pos_weight=0 zeroes positive class. Use >=0.1 or -1 (auto).")
        if spw < 0 and is_binary:
            spw = float(counts[0]) / max(float(counts[1]), 1.0)

        common_params = dict(
            n_estimators=int(self.cfg.n_estimators),
            max_depth=int(self.cfg.max_depth),
            learning_rate=float(self.cfg.learning_rate),
            subsample=float(self.cfg.subsample),
            colsample_bytree=float(self.cfg.colsample_bytree),
            reg_lambda=float(self.cfg.reg_lambda),
            min_child_weight=float(self.cfg.min_child_weight),
            gamma=float(self.cfg.gamma),
            random_state=int(self.cfg.random_state),
            n_jobs=int(self.cfg.n_jobs),
            tree_method="hist",
        )
        es = int(self.cfg.early_stopping_rounds)
        if es > 0:
            common_params["early_stopping_rounds"] = es

        if is_binary:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=spw,
                **common_params,
            )
        else:
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                **common_params,
            )

        model.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)], verbose=False)

        # Metrics
        pred = model.predict(X_va)
        acc = float(np.mean(pred == y_va)) if len(y_va) else 0.0
        pred = pred.astype(np.int64, copy=False) if hasattr(pred, "astype") else pred

        # ------------------------------------------------------------------
        # Proxy-PnL (cheap, deterministic) on validation slice
        # ------------------------------------------------------------------
        proxy_pnl: Dict[str, Any] = {"enabled": True, "task": task}
        try:
            fee_frac = float(getattr(self.cfg, "fee_bps", 0.0) or 0.0) / 10000.0
        except Exception:
            fee_frac = 0.0
        try:
            if len(y_va) == 0:
                proxy_pnl.update({"enabled": False, "reason": "empty_val"})
            elif task == "directional":
                closes = aux.get("closes")
                H = int(getattr(self.cfg, "horizon_steps", 0) or 0)
                if not isinstance(closes, np.ndarray) or closes.ndim != 1 or H <= 0:
                    proxy_pnl.update({"enabled": False, "reason": "no_closes_or_bad_horizon"})
                else:
                    # val slice indices in original time-series
                    idx = np.arange(train_n, n, dtype=np.int64)
                    # need future close available
                    ok = (idx + H) < len(closes)
                    idx = idx[ok]
                    if len(idx) == 0:
                        proxy_pnl.update({"enabled": False, "reason": "no_future_in_val"})
                    else:
                        c0 = closes[idx]
                        c1 = closes[idx + H]
                        ret = (c1 - c0) / np.maximum(c0, 1e-12)
                        pv = pred[: len(idx)]
                        pos = np.where(pv == 1, 1.0, np.where(pv == 2, -1.0, 0.0))
                        traded = np.abs(pos) > 0
                        pnl = pos * ret - (fee_frac * traded.astype(np.float64))
                        trades = int(np.sum(traded))
                        pnl_sum = float(np.sum(pnl))
                        proxy_pnl.update(
                            {
                                "trades": trades,
                                "pnl_sum": pnl_sum,
                                "pnl_mean_per_bar": float(np.mean(pnl)) if len(pnl) else 0.0,
                                "pnl_mean_per_trade": float(pnl_sum / trades) if trades else 0.0,
                                "fee_frac": fee_frac,
                                "horizon_steps": H,
                            }
                        )
            elif task.startswith("entry"):
                closes = aux.get("closes")
                mh = int(getattr(self.cfg, "max_hold_steps", 0) or 0)
                side = "long" if task.endswith("long") else "short"
                if not isinstance(closes, np.ndarray) or closes.ndim != 1 or mh <= 0:
                    proxy_pnl.update({"enabled": False, "reason": "no_closes_or_bad_max_hold"})
                else:
                    # val indices in original series
                    idx_all = np.arange(train_n, n, dtype=np.int64)
                    # only where model predicts enter
                    idx = idx_all[pred == 1]
                    # match labeling window: require full [t+1 .. t+mh] to exist
                    idx = idx[(idx + mh + 1) < len(closes)]
                    pnls: list[float] = []
                    for t in idx.tolist():
                        end = min(len(closes) - 1, int(t) + mh)
                        if end <= t:
                            continue
                        entry = float(closes[int(t)])
                        if entry <= 0:
                            continue
                        window = closes[int(t) + 1 : end + 1]
                        if len(window) == 0:
                            continue
                        if side == "long":
                            best_exit = float(np.max(window))
                            pnl = (best_exit - entry) / entry - fee_frac
                        else:
                            best_exit = float(np.min(window))
                            pnl = (entry - best_exit) / entry - fee_frac
                        pnls.append(float(pnl))
                    trades = int(len(pnls))
                    pnl_sum = float(np.sum(pnls)) if pnls else 0.0
                    proxy_pnl.update(
                        {
                            "trades": trades,
                            "pnl_sum": pnl_sum,
                            "pnl_mean_per_trade": float(pnl_sum / trades) if trades else 0.0,
                            "fee_frac": fee_frac,
                            "max_hold_steps": mh,
                            "position_side": side,
                        }
                    )
            elif task.startswith("exit"):
                # For exit rows, we don't have per-trade grouping; use pnl_fee feature as proxy.
                # Last 4 features are [age_norm, pnl_fee, mfe, mae].
                if X_va.shape[1] < 4:
                    proxy_pnl.update({"enabled": False, "reason": "bad_feature_dim"})
                else:
                    pnl_fee = X_va[:, -3].astype(np.float64, copy=False)
                    traded = pred == 1
                    pnls = pnl_fee[traded]
                    trades = int(pnls.shape[0])
                    pnl_sum = float(np.sum(pnls)) if trades else 0.0
                    proxy_pnl.update(
                        {
                            "trades": trades,
                            "pnl_sum": pnl_sum,
                            "pnl_mean_per_trade": float(pnl_sum / trades) if trades else 0.0,
                            "fee_frac": fee_frac,
                            "note": "exit_task_proxy_uses_pnl_fee_feature_no_trade_grouping",
                        }
                    )
            else:
                proxy_pnl.update({"enabled": False, "reason": "unknown_task"})
        except Exception as _e:
            proxy_pnl.update({"enabled": False, "reason": f"error:{_e}"})

        # Class distribution (train/val)
        y_counts_train = np.bincount(y_tr.astype(np.int64, copy=False), minlength=num_classes).astype(int).tolist()
        y_counts_val = np.bincount(y_va.astype(np.int64, copy=False), minlength=num_classes).astype(int).tolist()

        # Confusion matrix on val (rows=true, cols=pred)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        if len(y_va):
            for t, p in zip(y_va, pred):
                ti = int(t)
                pi = int(p)
                if 0 <= ti < num_classes and 0 <= pi < num_classes:
                    cm[ti, pi] += 1

        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / np.maximum(tp + fn, 1e-12)
        f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
        f1_buy_sell = None
        if not is_binary:
            f1_buy_sell = float((f1[1] + f1[2]) / 2.0)

        # Coverage / non-hold rates
        y_non_hold_rate_val = float(np.mean((y_va != 0))) if len(y_va) else 0.0
        pred_non_hold_rate_val = float(np.mean((pred != 0))) if len(y_va) else 0.0

        run_name = f"xgb-{str(uuid.uuid4())[:6].lower()}"
        sym = _symbol_code(symbol)
        run_dir = os.path.join(result_root, "xgb", sym, "runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        model_path = os.path.join(run_dir, "model.json")
        model.save_model(model_path)

        metrics = {
            "val_acc": acc,
            "train_rows": int(len(y_tr)),
            "val_rows": int(len(y_va)),
            "y_counts_train": y_counts_train,
            "y_counts_val": y_counts_val,
            "cm_val": cm.astype(int).tolist(),
            "precision_val": precision.tolist(),
            "recall_val": recall.tolist(),
            "f1_val": f1.tolist(),
            "f1_buy_sell_val": f1_buy_sell,
            "task": task,
            "y_non_hold_rate_val": y_non_hold_rate_val,
            "pred_non_hold_rate_val": pred_non_hold_rate_val,
            "proxy_pnl_val": proxy_pnl,
        }
        manifest = {
            "symbol": symbol,
            "symbol_code": sym,
            "run_name": run_name,
            "direction": self.cfg.direction,
            "task": task,
            "model_type": "xgb",
            "model_path": model_path,
            "metrics_path": os.path.join(run_dir, "metrics.json"),
            "meta_path": os.path.join(run_dir, "meta.json"),
            "proxy_pnl_val": {
                "enabled": bool(proxy_pnl.get("enabled")),
                "pnl_sum": proxy_pnl.get("pnl_sum"),
                "trades": proxy_pnl.get("trades"),
            },
        }
        _atomic_write_json(manifest["metrics_path"], metrics)
        _atomic_write_json(manifest["meta_path"], meta)
        _atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)

        return {
            "success": True,
            "symbol": symbol,
            "run_name": run_name,
            "result_dir": run_dir,
            "model_path": model_path,
            "metrics": metrics,
            "cfg_snapshot": asdict(self.cfg),
            "total_training_time": float(time.time() - started),
        }

