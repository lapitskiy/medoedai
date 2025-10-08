"""Цикл обучения SAC, основанный на DQN тренере."""

from __future__ import annotations

import json
import os
import time
import pickle
import random
import hashlib
import platform
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def _architecture_summary(model: Optional[torch.nn.Module]) -> Dict:
    if model is None:
        return {}
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sample_keys = []
        try:
            sd = model.state_dict()
            for i, k in enumerate(sd.keys()):
                if i >= 10:
                    break
                sample_keys.append(k)
        except Exception:
            pass
        return {
            "model_class": model.__class__.__name__,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "state_dict_keys_sample": sample_keys,
        }
    except Exception:
        return {}


def _sha256_of_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

from envs.dqn_model.gym.gconfig import GymConfig

from .config import SacConfig
from .sac_agent import SacAgent


@dataclass
class SacTrainingStats:
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    losses: list[Dict[str, float]] = field(default_factory=list)
    episode_pl_ratios: list[float] = field(default_factory=list)


class SacTrainer:
    def __init__(
        self,
        cfg: Optional[SacConfig] = None,
        episode_callback=None,
        progress_callback=None,
    ) -> None:
        self.cfg = cfg or SacConfig()
        self.episode_callback = episode_callback
        self.progress_callback = progress_callback
        
        # Early stopping и валидация
        self.best_winrate = 0.0
        self.best_winrate_episode = None
        self.patience_counter = 0
        self.episode_winrates = []
        self.episode_pl_ratios = [] # Добавлено
        self.validation_rewards = []
        self.best_pl_ratio = -np.inf # Инициализируем отрицательной бесконечностью
        self.best_pl_ratio_episode = None

    def train(
        self,
        dfs: Dict,
        gym_cfg: Optional[GymConfig] = None,
        env_factory=None,
        progress_callback=None,
    ) -> Dict:
        if progress_callback is not None:
            self.progress_callback = progress_callback
        from envs.sac_model.gym.crypto_trading_env_sac import CryptoTradingEnvOptimized as make_trading_env # Изменено на новую среду SAC

        env_factory = env_factory or make_trading_env
        env = env_factory(dfs, gym_cfg)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self._set_global_seed(self.cfg.seed)
        try:
            env.seed(self.cfg.seed)
        except Exception:
            pass

        self._training_started_at = time.time()

        if self.progress_callback:
            self.progress_callback(f"[Trainer] Training for {self.cfg.train_episodes} episodes.")

        agent = SacAgent(obs_dim, action_dim, cfg=self.cfg)
        stats = SacTrainingStats()
        alpha_values: list[float] = []
        entropy_values: list[float] = []

        # Тайминг обучения и чекпоинты
        start_time = time.time()
        checkpoint_interval = getattr(self.cfg, "checkpoint_interval", 50)

        for episode in range(self.cfg.train_episodes):
            if self.progress_callback:
                self.progress_callback(f"[Trainer] Starting episode {episode + 1}/{self.cfg.train_episodes}")
            state = env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device)
            state_input = state_tensor.unsqueeze(0)

            # Проверка на NaN/Inf во входном состоянии
            if not torch.isfinite(state_input).all():
                warn_msg = f"[Trainer] WARNING: Invalid values in initial state_input at episode {episode + 1}. Skipping episode."
                print(warn_msg)
                if self.progress_callback:
                    self.progress_callback(warn_msg)
                continue

            done = False
            episode_reward = 0.0
            steps = 0
            episode_invalid = False
            while not done:
                if not torch.isfinite(state_input).all():
                    warn_msg = f"[Trainer] WARNING: Non-finite state_input detected at episode {episode + 1}, step {steps}. Aborting episode."
                    print(warn_msg)
                    if self.progress_callback:
                        self.progress_callback(warn_msg)
                    episode_invalid = True
                    break

                action_tensor = agent.act(state_input)
                discrete_action = int(action_tensor.item())
                next_state, reward, done, info = env.step(discrete_action)

                if np.isnan(next_state).any() or np.isinf(next_state).any():  # type: ignore[arg-type]
                    warn_msg = f"[Trainer] WARNING: Environment returned non-finite next_state at episode {episode + 1}, step {steps}. Aborting episode."
                    print(warn_msg)
                    if self.progress_callback:
                        self.progress_callback(warn_msg)
                    episode_invalid = True
                    break

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=agent.device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=agent.device)
                done_tensor = torch.tensor(done, dtype=torch.bool, device=agent.device)
                action_store = torch.tensor(discrete_action, dtype=torch.long, device=agent.device)

                if not torch.isfinite(next_state_tensor).all() or not torch.isfinite(reward_tensor).all():
                    warn_msg = f"[Trainer] WARNING: Non-finite tensors detected at episode {episode + 1}, step {steps}. Aborting episode."
                    print(warn_msg)
                    if self.progress_callback:
                        self.progress_callback(warn_msg)
                    episode_invalid = True
                    break

                agent.store_transition(state_tensor, action_store, reward_tensor, next_state_tensor, done_tensor)

                if agent.ready_to_update():
                    batch = agent.sample_batch()
                    if batch:
                        losses = agent.update(batch)
                        if losses:
                            stats.losses.append(losses)
                            alpha_val = losses.get("alpha")
                            if isinstance(alpha_val, (int, float)):
                                alpha_values.append(float(alpha_val))
                            entropy_val = losses.get("entropy")
                            if isinstance(entropy_val, (int, float)):
                                entropy_values.append(float(entropy_val))

                state_tensor = next_state_tensor
                state_input = state_tensor.unsqueeze(0)
                episode_reward += reward
                steps += 1

            if episode_invalid:
                continue

            stats.episode_rewards.append(episode_reward)
            stats.episode_lengths.append(steps)

            # Оценка ETA и чекпоинт каждые N эпизодов
            elapsed = time.time() - start_time
            done_eps = episode + 1
            avg_ep_time = elapsed / max(done_eps, 1)
            rem_eps = max(self.cfg.train_episodes - done_eps, 0)
            eta_sec = int(avg_ep_time * rem_eps)
            eta_msg = f"[Trainer] Progress: {done_eps}/{self.cfg.train_episodes} eps, ETA ~ {eta_sec//3600:02d}:{(eta_sec%3600)//60:02d}:{eta_sec%60:02d}"
            print(eta_msg)
            if self.progress_callback:
                try:
                    self.progress_callback(eta_msg)
                except Exception:
                    pass

            if (done_eps % checkpoint_interval) == 0:
                # Сохраняем веса
                try:
                    agent.save(self.cfg.model_path)
                except Exception:
                    pass
                # Сохраняем буфер
                try:
                    with open(self.cfg.replay_buffer_path, "wb") as f:
                        pickle.dump(agent.replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass

            if self.episode_callback:
                try:
                    self.episode_callback(
                        episode=episode + 1,
                        reward=episode_reward,
                        steps=steps,
                    )
                except Exception:
                    pass

            # Валидация и early stopping
            should_validate = (episode + 1) % self.cfg.validation_interval == 0
            if should_validate:
                avg_reward = (
                    np.mean(stats.episode_rewards[-10:])
                    if len(stats.episode_rewards) >= 10
                    else episode_reward
                )
                
                # Выполняем валидационные эпизоды
                validation_reward = self._validate_agent(agent, env)
                self.validation_rewards.append(validation_reward)
                
                # Вычисляем winrate для текущего эпизода
                episode_winrate = self._compute_episode_winrate(env)
                self.episode_winrates.append(episode_winrate)
                
                # Вычисляем P/L Ratio для текущего эпизода
                episode_pl_ratio = self._compute_pl_ratio(env)
                self.episode_pl_ratios.append(episode_pl_ratio)
                
                # Проверяем улучшение по P/L Ratio
                improved = False
                if episode_pl_ratio > self.best_pl_ratio: # Используем P/L Ratio как основной критерий
                    self.best_pl_ratio = episode_pl_ratio
                    self.best_pl_ratio_episode = episode + 1
                    improved = True
                    self.patience_counter = 0
                    
                    # Сохраняем лучшую модель если включено
                    if self.cfg.save_only_on_improvement:
                        try:
                            best_model_path = Path(self.cfg.result_dir) / "best_model.pth"
                            agent.save(str(best_model_path))
                        except Exception:
                            pass
                else:
                    self.patience_counter += 1
                
                message = (
                    f"Episode {episode + 1}: reward={episode_reward:.2f}, "
                    f"avg_last_10={avg_reward:.2f}, validation={validation_reward:.2f}, "
                    f"winrate={episode_winrate:.3f}, pl_ratio={episode_pl_ratio:.3f}, " # Добавляем P/L Ratio
                    f"best_pl_ratio={self.best_pl_ratio:.3f}, patience={self.patience_counter}/{self.cfg.early_stopping_patience}"
                )
                if improved:
                    message += " ⭐ IMPROVED"
                print(message)
                if self.progress_callback:
                    try:
                        self.progress_callback(message)
                    except Exception:
                        pass
            
            # Early stopping
            if episode >= self.cfg.min_episodes_before_stopping:
                # Дополнительная защита от слишком раннего stopping
                if episode < self.cfg.train_episodes // 2:
                    patience_limit = self.cfg.early_stopping_patience // 4
                elif episode < self.cfg.train_episodes * 3 // 4:
                    patience_limit = self.cfg.early_stopping_patience // 2
                else:
                    patience_limit = self.cfg.early_stopping_patience
                
                if self.patience_counter >= patience_limit:
                    message = f"⚠️ Early stopping triggered after {episode + 1} episodes (patience limit reached)"
                    print(message)
                    if self.progress_callback:
                        try:
                            self.progress_callback(message)
                        except Exception:
                            pass
                    break

        self._save_results(env, stats, agent)

        alpha_stats = {}
        if alpha_values:
            alpha_arr = np.array(alpha_values, dtype=float)
            alpha_stats = {
                "avg": float(alpha_arr.mean()),
                "min": float(alpha_arr.min()),
                "max": float(alpha_arr.max()),
                "last": float(alpha_arr[-1]),
                "count": int(len(alpha_arr)),
            }

        entropy_stats = {}
        if entropy_values:
            entropy_arr = np.array(entropy_values, dtype=float)
            entropy_stats = {
                "avg": float(entropy_arr.mean()),
                "min": float(entropy_arr.min()),
                "max": float(entropy_arr.max()),
                "last": float(entropy_arr[-1]),
                "count": int(len(entropy_arr)),
            }

        return {
            "episode_rewards": stats.episode_rewards,
            "episode_lengths": stats.episode_lengths,
            "losses": stats.losses,
            "validation_rewards": self.validation_rewards,
            "episode_winrates": self.episode_winrates,
            "best_winrate": self.best_winrate,
            "best_winrate_episode": self.best_winrate_episode,
            "alpha_stats": alpha_stats,
            "entropy_stats": entropy_stats,
            "episode_pl_ratios": self.episode_pl_ratios, # Добавлено
            "best_pl_ratio": self.best_pl_ratio, # Добавлено
            "best_pl_ratio_episode": self.best_pl_ratio_episode, # Добавлено
        }

    def _save_results(self, env, stats: SacTrainingStats, agent: SacAgent) -> None:
        run_dir = Path(self.cfg.result_dir)
        last_model_path = run_dir / "last_model.pth"
        agent.save(self.cfg.model_path)
        if os.path.exists(self.cfg.model_path):
            try:
                import shutil as _sh
                _sh.copy2(self.cfg.model_path, last_model_path)
            except Exception:
                pass

        max_dd = self._compute_max_drawdown(getattr(env, "balance_history", []))

        run_dir_path = run_dir

        all_trades = getattr(env, "all_trades", [])
        trades_count = len(all_trades)

        roi_values: list[float] = []
        net_values: list[float] = []
        for trade in all_trades:
            roi = trade.get("roi")
            if isinstance(roi, (int, float)) and roi == roi:
                roi_values.append(float(roi))
            net = trade.get("net")
            if isinstance(net, (int, float)) and net == net:
                net_values.append(float(net))

        if roi_values:
            wins = sum(1 for roi in roi_values if roi > 0)
        elif net_values:
            wins = sum(1 for net in net_values if net > 0)
        else:
            wins = 0

        pnl_sum = float(sum(net_values)) if net_values else None

        profit_values = [v for v in net_values if v > 0]
        loss_values = [abs(v) for v in net_values if v < 0]
        if not profit_values and not loss_values and roi_values:
            profit_values = [v for v in roi_values if v > 0]
            loss_values = [abs(v) for v in roi_values if v < 0]

        pl_ratio = None
        if profit_values and loss_values:
            avg_gain = np.mean(profit_values)
            avg_loss = np.mean(loss_values)
            if avg_loss:
                pl_ratio = avg_gain / avg_loss

        final_stats = {
            "winrate": (wins / trades_count) if trades_count else None,
            "pl_ratio": pl_ratio,
            "trades_count": trades_count,
            "pnl_sum": pnl_sum,
        }

        raw_action_counts = getattr(env, "action_counts", {}) or {}
        action_counts_total = {}
        try:
            for key, value in raw_action_counts.items():
                try:
                    action_counts_total[str(key)] = int(value)
                except Exception:
                    action_counts_total[str(key)] = value
        except Exception:
            action_counts_total = {}

        buy_attempts_total = getattr(env, "buy_attempts", None)
        if buy_attempts_total is None:
            buy_attempts_total = action_counts_total.get("BUY", action_counts_total.get("1", 0))
        buy_attempts_total = int(buy_attempts_total or 0)

        buy_rejected_vol_total = getattr(env, "buy_rejected_vol", 0) or 0.0
        buy_rejected_roi_total = getattr(env, "buy_rejected_roi", 0) or 0.0
        episodes_with_trade_count = getattr(env, "episodes_with_trade_count", None)
        if episodes_with_trade_count is None:
            episodes_with_trade_count = sum(1 for t in all_trades if t.get("opened", False))
        episodes_with_trade_count = int(episodes_with_trade_count or 0)
        total_steps_processed = sum(stats.episode_lengths)
        episode_length = int(np.mean(stats.episode_lengths)) if stats.episode_lengths else None

        reward_scale = float(getattr(env.cfg, "reward_scale", 1.0)) if hasattr(env, "cfg") else 1.0

        git_commit = None
        try:
            git_commit = subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_commit = None

        train_metadata = {
            "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": os.environ.get("COMPUTERNAME") or platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": getattr(self.cfg, "seed", None),
            "torch_version": torch.__version__,
            "git_commit": git_commit,
        }

        cfg_snapshot = getattr(self.cfg, "to_dict", None)
        cfg_snapshot = cfg_snapshot() if callable(cfg_snapshot) else self.cfg.__dict__.copy()
        gym_snapshot = {
            "episode_length": getattr(env, "episode_length", None),
            "lookback_window": getattr(env, "lookback_window", None),
            "reward_scale": reward_scale,
            "symbol": getattr(env, "symbol", None),
        }

        weights_info = {
            "model_path": os.path.relpath(self.cfg.model_path, run_dir_path),
            "model_sha256": _sha256_of_file(self.cfg.model_path),
            "buffer_path": None,
            "buffer_sha256": None,
        }

        architecture = {
            "actor": _architecture_summary(getattr(agent, "actor", None)) if hasattr(agent, "actor") else {},
            "critic1": _architecture_summary(getattr(agent, "critic1", None)) if hasattr(agent, "critic1") else {},
            "critic2": _architecture_summary(getattr(agent, "critic2", None)) if hasattr(agent, "critic2") else {},
        }

        total_training_time = None
        started_at = getattr(self, "_training_started_at", None)
        if isinstance(started_at, (int, float)):
            total_training_time = time.time() - started_at
        elif isinstance(started_at, datetime):
            total_training_time = (datetime.utcnow() - started_at).total_seconds()

        episode_winrates = []
        per_episode_trades = getattr(env, "episode_trades", None)
        if isinstance(per_episode_trades, list) and per_episode_trades:
            for trades in per_episode_trades:
                try:
                    trades = trades or []
                    profitable = [t for t in trades if t.get("roi", 0) > 0]
                    episode_winrates.append(len(profitable) / len(trades) if trades else 0.0)
                except Exception:
                    episode_winrates.append(0.0)
        else:
            per_episode_trades = []

        replay_exists = False

        training_results = {
            "episodes": self.cfg.train_episodes,
            "actual_episodes": len(stats.episode_rewards),
            "total_training_time": total_training_time,
            "episode_winrates": episode_winrates,
            "all_trades": all_trades,
            "final_stats": final_stats,
            "train_metadata": train_metadata,
            "episode_length": episode_length,
            "action_counts_total": action_counts_total,
            "buy_attempts_total": buy_attempts_total,
            "buy_rejected_vol_total": buy_rejected_vol_total,
            "buy_rejected_roi_total": buy_rejected_roi_total,
            "episodes_with_trade_count": episodes_with_trade_count,
            "buy_reject_rate": (buy_rejected_vol_total / buy_attempts_total) if buy_attempts_total else None,
            "episodes_with_trade_ratio": (episodes_with_trade_count / float(len(stats.episode_rewards))) if stats.episode_rewards else 0.0,
            "total_steps_processed": total_steps_processed,
            "reward_scale": reward_scale,
            "cfg_snapshot": cfg_snapshot,
            "gym_snapshot": gym_snapshot,
            "architecture": architecture,
            "weights": weights_info,
            "metrics": self._gather_metrics(env, stats, max_dd),
            "per_episode_trades": per_episode_trades,
            "model_path": str(Path(self.cfg.model_path)),
            "buffer_path": None,
        }

        train_result_path = run_dir_path / "train_result.pkl"
        try:
            with open(train_result_path, "wb") as tf:
                pickle.dump(training_results, tf, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Не удалось записать train_result.pkl: {exc}")

        if getattr(agent, "replay_buffer", None):
            replay_path = run_dir_path / "replay.pkl"
            try:
                with open(replay_path, "wb") as rf:
                    pickle.dump(agent.replay_buffer, rf, protocol=pickle.HIGHEST_PROTOCOL)
                replay_exists = True
                training_results["weights"]["buffer_sha256"] = _sha256_of_file(str(replay_path))
                training_results["buffer_path"] = str(Path(self.cfg.replay_buffer_path))
            except Exception:
                replay_exists = False

        training_results["replay_exists"] = replay_exists
        training_results["weights"]["buffer_path"] = "replay.pkl" if replay_exists else None
        if replay_exists:
            training_results["buffer_path"] = str(Path(self.cfg.replay_buffer_path))

        metrics = training_results["metrics"]
        metrics_path = run_dir / "metrics.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Не удалось записать metrics.json: {exc}")

        symbol_value = None
        if isinstance(self.cfg.extra, dict):
            symbol_value = self.cfg.extra.get("symbol")

        manifest = {
            "run_id": self.cfg.run_name,
            "parent_run_id": getattr(self.cfg, "parent_run_id", None),
            "root_id": getattr(self.cfg, "root_id", None),
            "symbol": symbol_value,
            "seed": int(self.cfg.seed) if isinstance(self.cfg.seed, int) else None,
            "episodes_end": len(stats.episode_rewards),
            "episodes_planned": self.cfg.train_episodes,
            "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts": {
                "model": "model.pth",
                "replay": "replay.pkl" if replay_exists else None,
                "result": "train_result.pkl",
                "last_model": "last_model.pth" if os.path.exists(run_dir / "last_model.pth") else None,
            },
            "metrics": training_results["metrics"],
        }
        try:
            with open(run_dir / "manifest.json", "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Не удалось записать manifest.json: {exc}")

    def _validate_agent(self, agent: SacAgent, env) -> float:
        """Проводит валидационный прогон агента без обновления параметров."""
        try:
            from copy import deepcopy
        except ImportError:  # pragma: no cover
            deepcopy = None

        env_copy = deepcopy(env) if deepcopy is not None else env
        total_reward = 0.0
        episodes = max(1, getattr(self.cfg, "validation_episodes", 1))

        for _ in range(episodes):
            state = env_copy.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            max_steps = getattr(self.cfg, "max_episode_steps", None)
            if max_steps is None:
                max_steps = getattr(env_copy, "episode_length", 10_000) or 10_000
            while not done and steps < max_steps:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                if not torch.isfinite(state_tensor).all():
                    break
                action_tensor = agent.act(state_tensor, deterministic=True)
                action = int(action_tensor.item())
                next_state, reward, done, _ = env_copy.step(action)
                if np.isnan(next_state).any() or np.isinf(next_state).any():  # type: ignore[arg-type]
                    break
                state = next_state
                episode_reward += float(reward)
                steps += 1
            total_reward += episode_reward

        return total_reward / episodes if episodes else 0.0

    def _compute_episode_winrate(self, env) -> float:
        """Оценивает долю прибыльных сделок в текущем эпизоде."""
        trades = getattr(env, "all_trades", None)
        if trades:
            profitable = [t for t in trades if t.get("roi", 0) > 0]
            return len(profitable) / len(trades) if trades else 0.0
        trades = getattr(env, "trades", None)
        if trades:
            profitable = [t for t in trades if t.get("roi", 0) > 0]
            return len(profitable) / len(trades) if trades else 0.0
        return 0.0

    def _compute_pl_ratio(self, env) -> Optional[float]:
        """Вычисляет P/L Ratio для всех сделок в окружении."""
        trades = getattr(env, "all_trades", [])
        if not trades:
            return None

        net_values = []
        for trade in trades:
            net = trade.get("net")
            if isinstance(net, (int, float)) and net == net: # Проверка на NaN
                net_values.append(float(net))

        if not net_values:
            return None

        profit_values = [v for v in net_values if v > 0]
        loss_values = [abs(v) for v in net_values if v < 0]

        if profit_values and loss_values:
            avg_gain = np.mean(profit_values)
            avg_loss = np.mean(loss_values)
            if avg_loss:
                return avg_gain / avg_loss
        return None

    def _compute_best_winrate(self, env) -> float:
        trades = getattr(env, "all_trades", [])
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.get("roi", 0) > 0)
        return wins / len(trades)

    def _compute_max_drawdown(self, balance_history) -> float:
        if not balance_history:
            return 0.0
        peak = balance_history[0]
        max_dd = 0.0
        for balance in balance_history:
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak if peak else 0.0
            max_dd = max(max_dd, drawdown)
        return max_dd

    def _gather_metrics(self, env, stats: SacTrainingStats, max_dd: float) -> Dict[str, float]:
        trades = getattr(env, "all_trades", [])
        winrate = self._compute_best_winrate(env)
        roi = float(np.mean([t.get("roi", 0.0) for t in trades])) if trades else 0.0
        return {
            "episodes": len(stats.episode_rewards),
            "avg_reward": float(np.mean(stats.episode_rewards)) if stats.episode_rewards else 0.0,
            "winrate": float(winrate),
            "avg_roi": float(roi),
            "max_drawdown": float(max_dd),
        }

    def _set_global_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            random.seed(seed)
        except Exception:
            pass
        try:
            np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


