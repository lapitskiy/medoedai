import os
import re
import sys
import tempfile
import logging
import numpy as np
from collections import deque
import torch
import wandb
import time
import psutil
from typing import Any, Dict, List, Optional
import pickle
from pickle import HIGHEST_PROTOCOL
import hashlib
import platform
from datetime import datetime
import subprocess
from utils.adaptive_normalization import adaptive_normalizer
import json as _json

from train.infrastructure.logging.tee import TailTruncatingTee
# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.vdqn.dqnsolver import DQNSolver
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from envs.dqn_model.gym.crypto_trading_env_short import CryptoTradingEnvShort
from train.infrastructure.gym.position_aware_wrapper import PositionAwareEpisodeWrapper
from train.domain.episode.extension_policy import EpisodeExtensionPolicy
from envs.dqn_model.gym.gconfig import GymConfig
from agents.vdqn.hyperparameter.symbol_overrides import get_symbol_override
from agents.vdqn.hyperparameter.global_overrides import GLOBAL_OVERRIDES
from utils.settings_store import get_setting_value as _get_setting_value


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_bytes(num_bytes: int) -> str:
    try:
        gb = num_bytes / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = num_bytes / (1024 ** 2)
        return f"{mb:.0f} MB"
    except Exception:
        return str(num_bytes)


def log_resource_usage(tag: str = "train", extra: str = "") -> None:
    try:
        process = psutil.Process(os.getpid())
        total_cpus = os.cpu_count() or 1
        cpu_total_pct = psutil.cpu_percent(interval=0.0)
        cpu_proc_pct = process.cpu_percent(interval=0.0)
        mem = psutil.virtual_memory()
        mem_total_pct = mem.percent
        mem_proc = process.memory_info().rss
        try:
            load1, _load5, _load15 = os.getloadavg()
            load_str = f"{load1:.1f}/{total_cpus}"
        except Exception:
            load_str = "n/a"
        omp = os.environ.get('OMP_NUM_THREADS')
        mkl = os.environ.get('MKL_NUM_THREADS')
        tor = os.environ.get('TORCH_NUM_THREADS')
        frac = os.environ.get('TRAIN_CPU_FRACTION')
        line = (
            f"[RES-{tag}] CPU {cpu_total_pct:.0f}% (proc {cpu_proc_pct:.0f}%), load {load_str}, "
            f"mem {mem_total_pct:.0f}% (proc {_format_bytes(mem_proc)}), "
            f"OMP/MKL/TORCH={omp}/{mkl}/{tor}"
            + (f", FRACTION={frac}" if frac else "")
            + (f" | {extra}" if extra else "")
        )
        print(line, flush=True)
    except Exception:
        pass


def _sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _atomic_write_json(path: str, data: Dict) -> None:
    try:
        dir_path = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
    except Exception:
        # Безопасно игнорируем сбой записи (не ломаем обучение)
        pass


def _safe_read_json(path: str) -> Any:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return _json.load(f)
    except Exception:
        return None
    return None


def _safe_cfg_snapshot(cfg_obj) -> Dict:
    try:
        snap = {}
        for k, v in (cfg_obj.__dict__.items() if hasattr(cfg_obj, '__dict__') else []):
            try:
                if str(type(v)).endswith("torch.device'>"):
                    snap[k] = str(v)
                elif hasattr(v, 'tolist'):
                    snap[k] = v.tolist()
                else:
                    snap[k] = v
            except Exception:
                snap[k] = str(v)
        return snap
    except Exception:
        return {}


def _architecture_summary(model: torch.nn.Module) -> Dict:
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
            'model_class': model.__class__.__name__,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'state_dict_keys_sample': sample_keys,
        }
    except Exception:
        return {}


def get_env_attr_safe(env, name: str, default=None):
    """Безопасно читает атрибут среды без варнингов Gym.
    1) Пытается через get_wrapper_attr (для обёрток)
    2) Затем из env.unwrapped
    3) Затем напрямую из env
    4) Затем пробует пройти по цепочке .env (обёртки)
    """
    try:
        if hasattr(env, 'get_wrapper_attr'):
            try:
                val = env.get_wrapper_attr(name)
                if val is not None:
                    return val
            except Exception:
                pass
    except Exception:
        pass
    try:
        base = getattr(env, 'unwrapped', None)
        if base is not None and hasattr(base, name):
            return getattr(base, name)
    except Exception:
        pass
    # direct attr disabled: gymnasium warns on wrapper access
    # wrapper chain: env.env.env...
    try:
        cur = env
        for _ in range(8):
            nxt = getattr(cur, 'env', None)
            if nxt is None:
                break
            cur = nxt
            if hasattr(cur, name):
                return getattr(cur, name)
    except Exception:
        pass
    return default


def set_env_attr_safe(env, name: str, value) -> bool:
    """Безопасно устанавливает атрибут базовой среды (unwrapped),
    при неудаче — прямо на env. Возвращает True при успехе.
    """
    try:
        base = getattr(env, 'unwrapped', None)
        if base is not None:
            setattr(base, name, value)
            return True
    except Exception:
        pass
    try:
        setattr(env, name, value)
        return True
    except Exception:
        return False

def prepare_data_for_training(dfs: Dict) -> Dict:
    """
    Подготавливает данные для тренировки, конвертируя DataFrame в numpy массивы
    
    Args:
        dfs: словарь с DataFrame для разных таймфреймов
        
    Returns:
        Dict: словарь с numpy массивами для разных таймфреймов
    """
    print(f"📊 Подготавливаю данные для тренировки")
    
    # Проверяем наличие необходимых данных
    required_keys = ['df_5min', 'df_15min', 'df_1h']
    for key in required_keys:
        if key not in dfs:
            raise ValueError(f"Отсутствует {key} в переданных данных")
        if dfs[key] is None or dfs[key].empty:
            raise ValueError(f"{key} пустой или None")
    
    print(f"✅ Данные готовы: 5min={len(dfs['df_5min'])}, 15min={len(dfs['df_15min'])}, 1h={len(dfs['df_1h'])}")
    
    return dfs

def _save_training_results(
    run_dir: str,
    cfg, # vDqnConfig
    training_name: str,
    current_episode: int, # Фактически завершенный эпизод или текущий для промежуточного сохранения
    total_episodes_planned: int, # Общее количество запланированных эпизодов
    all_trades: list,
    episode_winrates: list,
    best_winrate: float,
    best_episode_idx: int,
    action_counts_total: dict,
    buy_attempts_total: int,
    buy_rejected_vol_total: int,
    buy_rejected_roi_total: int,
    episodes_with_trade_count: int,
    total_steps_processed: int,
    episode_length: Optional[int],
    seed: Optional[int],
    dqn_solver, # DQNSolver instance
    env, # CryptoTradingEnvOptimized or MultiCryptoTradingEnv instance
    is_multi_crypto: bool,
    parent_run_id: Optional[str],
    root_id: Optional[str],
    training_start_time: float,
    current_total_training_time: float, # Текущее время обучения для промежуточных сохранений
    episode_epsilons: list | None = None,
    eps_threshold: float | None = None,
    eval_summary: dict | None = None,
    dfs: Optional[Dict] = None,
    direction: str = 'long',
    winrate_trend: dict | None = None,
):
    try:
        total_training_time = current_total_training_time

        # Определяем список плохих сделок (убыточные сделки)
        bad_trades_list = []
        try:
            if all_trades:
                bad_trades_list = [t for t in all_trades if t.get('roi', 0) < 0]
        except Exception:
            bad_trades_list = []

        bad_trades_count = len(bad_trades_list)
        total_trades_count = len(all_trades) if all_trades else 0
        bad_trades_percentage = (bad_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

        # Получаем статистику, если есть сделки, иначе пустой словарь
        stats_all = dqn_solver.print_trade_stats(all_trades) if all_trades and dqn_solver is not None else {}

        # Компактная поддержка winrate без необходимости хранить весь список
        try:
            ew_count = int(len(episode_winrates) if episode_winrates else 0)
            ew_last = episode_winrates[-1] if (episode_winrates and len(episode_winrates) > 0) else None
            ew_best = max(episode_winrates) if (episode_winrates and len(episode_winrates) > 0) else None
            ew_avg = (sum(episode_winrates) / float(len(episode_winrates))) if (episode_winrates and len(episode_winrates) > 0) else None
        except Exception:
            ew_count, ew_last, ew_best, ew_avg = 0, None, None, None
        store_full = bool(getattr(cfg, 'store_episode_winrates_full', True)) if cfg is not None else True
        tail_n = int(getattr(cfg, 'store_episode_winrates_tail', getattr(cfg, 'winrate_trend_window', 200))) if cfg is not None else 200
        ew_tail = []
        try:
            if episode_winrates and tail_n > 0:
                ew_tail = list(episode_winrates[-tail_n:])
        except Exception:
            ew_tail = []

        training_results = {
            'episodes': total_episodes_planned,  # Планируемое количество эпизодов
            'actual_episodes': current_episode,  # Реальное количество завершенных эпизодов (текущий эпизод)
            'total_training_time': total_training_time,
            'episode_winrates': (episode_winrates if store_full else None),
            'episode_winrates_tail': ew_tail,
            'episode_winrates_count': ew_count,
            'episode_winrates_last': ew_last,
            'episode_winrates_best': ew_best,
            'episode_winrates_avg': ew_avg,
            'episode_epsilons': episode_epsilons or [],
            'all_trades': all_trades,
            'bad_trades': bad_trades_list,
            'bad_trades_count': bad_trades_count,
            'bad_trades_percentage': bad_trades_percentage,
            'best_winrate': best_winrate,
            'final_stats': stats_all,
            'training_date': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'model_path': cfg.model_path,
            'buffer_path': getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None)),
            'symbol': training_name,
            'model_id': getattr(cfg, 'run_id', None) or (run_dir.split(os.sep)[-1] if run_dir else None), # Используем run_id из cfg или из run_dir
            'early_stopping_triggered': current_episode < total_episodes_planned,  # True если early stopping сработал
            'reward_scale': float(getattr(get_env_attr_safe(env, 'cfg'), 'reward_scale', 1.0)),
            'winrates': None,
            # --- Новые агрегаты для анализа поведения ---
            'action_counts_total': action_counts_total,
            'buy_attempts_total': buy_attempts_total,
            'buy_rejected_vol_total': buy_rejected_vol_total,
            'buy_rejected_roi_total': buy_rejected_roi_total,
            'buy_accept_rate': ( (action_counts_total.get(1, 0) or 0) / float(buy_attempts_total) ) if buy_attempts_total > 0 else 0.0,
            'episodes_with_trade_count': episodes_with_trade_count,
            'episodes_with_trade_ratio': (episodes_with_trade_count / float(current_episode)) if current_episode > 0 else 0.0,
            'avg_minutes_between_buys': ( (total_steps_processed * 5.0) / float(action_counts_total.get(1, 0) or 1) ) if (action_counts_total.get(1, 0) or 0) > 0 else None,
            'total_steps_processed': total_steps_processed,
            'episode_length': episode_length, # Добавляем длину эпизода
            # Продажи по причинам (кумулятивно из env)
            'sell_types_total': (get_env_attr_safe(env, 'cumulative_sell_types', {})),
            # Дополнительные метрики из dqn_solver
            'epsilon_final_value': dqn_solver.epsilon if dqn_solver is not None else None,
            'learning_rate_final_value': dqn_solver.optimizer.param_groups[0]['lr'] if dqn_solver and dqn_solver.optimizer and dqn_solver.optimizer.param_groups else None,
             # BUY/HOLD агрегаты
             'buy_stats_total': (get_env_attr_safe(env, 'buy_stats_total', {})),
             'hold_stats_total': (get_env_attr_safe(env, 'hold_stats_total', {})),
             # Market STATE counters (decision-time distribution)
             'market_state_counts_total': (get_env_attr_safe(env, 'market_state_counts_total', {})),
             'market_state_counts_episode': (get_env_attr_safe(env, 'market_state_counts_episode', {})),
            # Компактная история тренда winrate (снэпшоты + rolling median/EMA)
            'winrate_trend': winrate_trend or None,
        }
        
        # Печать BUY/HOLD статистики (если есть)
        try:
            buy_total = get_env_attr_safe(env, 'buy_stats_total', {})
            hold_total = get_env_attr_safe(env, 'hold_stats_total', {})
            if isinstance(buy_total, dict) and buy_total:
                print("\n📊 Детализация BUY:")
                for k, v in buy_total.items():
                    print(f"  • {k}: {int(v)}")
            if isinstance(hold_total, dict) and hold_total:
                print("\n📊 Детализация HOLD:")
                for k, v in hold_total.items():
                    print(f"  • {k}: {int(v)}")
        except Exception:
            pass
        # Печать Market STATE счётчиков (если есть)
        try:
            ms_total = get_env_attr_safe(env, 'market_state_counts_total', {})
            if isinstance(ms_total, dict) and ms_total:
                print("\n🧭 Market STATE counts (total):")
                for k, v in ms_total.items():
                    print(f"  • {k}: {int(v)}")
        except Exception:
            pass
        # Агрегация винрейтов (общий, эксплуатационный) и eval, если передан
        try:
            wr_all = float(np.mean(episode_winrates)) if episode_winrates else None
        except Exception:
            wr_all = None
        wr_exploit = None
        try:
            thr = eps_threshold if eps_threshold is not None else float(getattr(cfg, 'winrate_eps_threshold', 0.2))
            if episode_epsilons and episode_winrates and len(episode_epsilons) == len(episode_winrates):
                exploit_indices = [i for i, e in enumerate(episode_epsilons) if isinstance(e, (int, float)) and e <= thr]
                if exploit_indices:
                    wr_exploit = float(np.mean([episode_winrates[i] for i in exploit_indices]))
        except Exception:
            wr_exploit = None
        try:
            training_results['winrates'] = {
                'train_all': wr_all,
                'train_exploit': wr_exploit,
                'eps_threshold': float(eps_threshold) if eps_threshold is not None else float(getattr(cfg, 'winrate_eps_threshold', 0.2)),
                'counts': {
                    'episodes_total': int(len(episode_winrates) if episode_winrates else 0),
                    'episodes_exploit': int(sum(1 for e in (episode_epsilons or []) if isinstance(e, (int, float)) and e <= (eps_threshold if eps_threshold is not None else getattr(cfg, 'winrate_eps_threshold', 0.2))))
                }
            }
        except Exception:
            training_results['winrates'] = None
        if isinstance(eval_summary, dict) and eval_summary:
            training_results['eval'] = eval_summary
        # Метаданные окружения и запуска
        git_commit = None
        try:
            git_commit = subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        train_seed = seed if isinstance(seed, int) else None
        train_metadata = {
            'created_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            'omp_threads': os.environ.get('OMP_NUM_THREADS'),
            'mkl_threads': os.environ.get('MKL_NUM_THREADS'),
            'torch_threads': os.environ.get('TORCH_NUM_THREADS'),
            'train_cpu_fraction': os.environ.get('TRAIN_CPU_FRACTION'),
            'git_commit': git_commit,
            'script': os.path.basename(__file__),
            'seed': train_seed,
        }

        # Снимок конфигурации и архитектуры
        cfg_snapshot = _safe_cfg_snapshot(cfg)
        arch_main = _architecture_summary(dqn_solver.model) if dqn_solver and hasattr(dqn_solver, 'model') else {}
        arch_target = _architecture_summary(dqn_solver.target_model) if dqn_solver and hasattr(dqn_solver, 'target_model') else {}
        
        # Снимок параметров окружения и данных (для воспроизводимости) ---
        gym_snapshot = {}
        try:
            cfg_obj = get_env_attr_safe(env, 'cfg')
            gym_snapshot = {
                'symbol': get_env_attr_safe(env, 'symbol'),
                'lookback_window': get_env_attr_safe(env, 'lookback_window'),
                'indicators_config': get_env_attr_safe(env, 'indicators_config'),
                'reward_scale': getattr(cfg_obj, 'reward_scale', 1.0),
                'episode_length': get_env_attr_safe(env, 'episode_length'),
                'funding_features': {
                    'present_in_input_df': [], # Not directly available from env, needs to be passed
                    'included': False,
                },
                'risk_management': {
                    'STOP_LOSS_PCT': get_env_attr_safe(env, 'STOP_LOSS_PCT'),
                    'TAKE_PROFIT_PCT': get_env_attr_safe(env, 'TAKE_PROFIT_PCT'),
                    'min_hold_steps': get_env_attr_safe(env, 'min_hold_steps'),
                    'volume_threshold': get_env_attr_safe(env, 'volume_threshold'),
                    'base_stop_loss': get_env_attr_safe(env, 'base_stop_loss'),
                    'base_take_profit': get_env_attr_safe(env, 'base_take_profit'),
                    'base_min_hold': get_env_attr_safe(env, 'base_min_hold'),
                    'atr_trail_mult': getattr(cfg_obj, 'atr_trail_mult', None),
                    'atr_sl_mult': getattr(cfg_obj, 'atr_sl_mult', None),
                    # buy filters / gates (to reproduce throttling behavior)
                    'entry_confidence_gate': getattr(cfg_obj, 'entry_confidence_gate', None),
                    'buy_roi_thr': get_env_attr_safe(env, 'buy_roi_thr'),
                    'buy_trend_thr': get_env_attr_safe(env, 'buy_trend_thr'),
                    'buy_volat_thr': get_env_attr_safe(env, 'buy_volat_thr'),
                    'buy_strictness_floor': get_env_attr_safe(env, 'buy_strictness_floor'),
                    'buy_vol_min_lenient': get_env_attr_safe(env, 'buy_vol_min_lenient'),
                    'buy_vol_min_strict': get_env_attr_safe(env, 'buy_vol_min_strict'),
                    'buy_vol_floor_mult': get_env_attr_safe(env, 'buy_vol_floor_mult'),
                },
                'position_sizing': {
                    'base_position_fraction': get_env_attr_safe(env, 'base_position_fraction'),
                    'position_fraction': get_env_attr_safe(env, 'position_fraction'),
                    'position_confidence_threshold': get_env_attr_safe(env, 'position_confidence_threshold'),
                },
                'observation_space_shape': get_env_attr_safe(env, 'observation_space_shape'),
                'step_minutes': getattr(cfg_obj, 'step_minutes', 5),
            }
        except Exception as e:
            logger.warning(f"Ошибка при создании gym_snapshot: {e}")
            gym_snapshot = {}

        # --- Снимок адаптивной нормализации (по символам) ---
        adaptive_snapshot = {}
        try:
            if is_multi_crypto:
                # Мульти-режим: сохраняем компактно по символам (без тяжёлых массивов).
                per_symbol = {}
                try:
                    if isinstance(dfs, dict):
                        for sym, data in dfs.items():
                            try:
                                df5 = (data.get('df_5min') if isinstance(data, dict) else None)
                                if df5 is None:
                                    continue
                                train_split_point = None
                                try:
                                    train_split_point = int(len(df5) * 0.8)
                                except Exception:
                                    train_split_point = None
                                tp = adaptive_normalizer.get_trading_params(
                                    str(sym or ''),
                                    df5,
                                    train_split_point=train_split_point,
                                )
                                if isinstance(tp, dict):
                                    tp.pop('regime_precomputed', None)
                                mc = adaptive_normalizer.analyze_market_conditions(df5, train_split_point=train_split_point)
                                ap = adaptive_normalizer.adapt_parameters(str(sym or ''), df5, train_split_point=train_split_point)
                                if isinstance(ap, dict):
                                    ap.pop('regime_precomputed', None)
                                per_symbol[str(sym)] = {
                                    'trading_params': tp,
                                    'market_conditions': mc,
                                    'adapted_params': ap,
                                }
                            except Exception:
                                continue
                except Exception:
                    per_symbol = {}
                adaptive_snapshot = {'per_symbol': per_symbol}
            else:
                # Single-режим: фиксируем символ + динамические параметры (compact) + базовую инфу по фреймам.
                sym = None
                try:
                    sym = get_env_attr_safe(env, 'symbol', None)
                except Exception:
                    sym = None
                if not sym:
                    # training_name для single обычно равен crypto_symbol (например TONUSDT)
                    sym = training_name
                df5 = dfs.get('df_5min') if isinstance(dfs, dict) else None
                frames = {}
                try:
                    if isinstance(dfs, dict):
                        for key, value in dfs.items():
                            try:
                                if key in ('df_5min', 'df_15min', 'df_1h'):
                                    frames[key] = {'rows': (len(value) if hasattr(value, '__len__') else None)}
                            except Exception:
                                continue
                except Exception:
                    frames = {}
                adaptive_snapshot = {'symbol': sym, 'frames': frames}
                try:
                    if df5 is not None and isinstance(sym, str) and sym:
                        train_split_point = None
                        try:
                            train_split_point = int(len(df5) * 0.8)
                        except Exception:
                            train_split_point = None
                        tp = adaptive_normalizer.get_trading_params(sym, df5, train_split_point=train_split_point)
                        if isinstance(tp, dict):
                            tp.pop('regime_precomputed', None)
                        mc = adaptive_normalizer.analyze_market_conditions(df5, train_split_point=train_split_point)
                        ap = adaptive_normalizer.adapt_parameters(sym, df5, train_split_point=train_split_point)
                        if isinstance(ap, dict):
                            ap.pop('regime_precomputed', None)
                        adaptive_snapshot['trading_params'] = tp
                        adaptive_snapshot['market_conditions'] = mc
                        adaptive_snapshot['adapted_params'] = ap
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Ошибка при создании adaptive_snapshot: {e}")
            adaptive_snapshot = {}
            
        # Информация о весах (пути и хэши)
        weights_info = {
            'model_path': cfg.model_path,
            'buffer_path': getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None)),
            'model_sha256': _sha256_of_file(cfg.model_path) if cfg and getattr(cfg, 'model_path', None) and os.path.exists(cfg.model_path) else None,
            'buffer_sha256': _sha256_of_file(getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None))) if cfg and getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None)) and os.path.exists(getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None))) else None,
            'encoder_path': getattr(cfg, 'encoder_path', None),
            'encoder_sha256': _sha256_of_file(getattr(cfg, 'encoder_path', None)) if cfg and getattr(cfg, 'encoder_path', None) and os.path.exists(getattr(cfg, 'encoder_path', None)) else None,
        }

        # Объединяем
        # Сохраняем all_trades отдельно, чтобы не раздувать train_result.pkl
        all_trades_path = None
        all_trades_count = len(all_trades) if isinstance(all_trades, list) else 0
        # Компактный ряд ROI по сделкам (для /analitika), чтобы не зависеть от JSON сериализации all_trades
        trades_roi = []
        try:
            if isinstance(all_trades, list) and all_trades:
                for t in all_trades:
                    if not isinstance(t, dict):
                        continue
                    v = t.get('roi', None)
                    try:
                        fv = float(v)
                        if fv == fv:  # not NaN
                            trades_roi.append(fv)
                    except Exception:
                        continue
        except Exception:
            trades_roi = []
        try:
            if isinstance(all_trades, list) and all_trades:
                trades_json_path = os.path.join(run_dir, 'all_trades.json')
                def _jsonable(v):
                    if v is None or isinstance(v, (int, float, str, bool)):
                        return v
                    try:
                        import numpy as _np  # type: ignore
                        if isinstance(v, (_np.integer,)):
                            return int(v)
                        if isinstance(v, (_np.floating,)):
                            return float(v)
                    except Exception:
                        pass
                    try:
                        import datetime as _dt
                        if isinstance(v, (_dt.datetime, _dt.date)):
                            return v.isoformat()
                    except Exception:
                        pass
                    return str(v)
                safe_trades = []
                for t in all_trades:
                    if isinstance(t, dict):
                        safe_trades.append({str(k): _jsonable(v) for k, v in t.items()})
                    else:
                        safe_trades.append({'_raw': str(t)})
                with open(trades_json_path, 'w', encoding='utf-8') as tf:
                    json.dump(safe_trades, tf, ensure_ascii=False)
                all_trades_path = trades_json_path
        except Exception:
            all_trades_path = None

        store_trades_inline = False
        try:
            v = str(get_config_value('TS_STORE_ALL_TRADES_IN_PKL', '0'))
            store_trades_inline = v.lower() in ('1', 'true', 'yes', 'y')
        except Exception:
            store_trades_inline = False

        enriched_results = {
            **training_results,
            'train_metadata': train_metadata,
            'cfg_snapshot': cfg_snapshot,
            'gym_snapshot': gym_snapshot,
            'adaptive_normalization': adaptive_snapshot,
            'architecture': {
                'main': arch_main,
                'target': arch_target,
            },
            'weights': weights_info,
            'trades_roi': trades_roi,
            'all_trades': (all_trades if store_trades_inline else []),
            'all_trades_path': all_trades_path,
            'all_trades_count': all_trades_count,
        }

        # Создаем папку, если не существует
        os.makedirs(run_dir, exist_ok=True)
        results_file = os.path.join(run_dir, 'train_result.pkl')

        with open(results_file, 'wb') as f:
            pickle.dump(enriched_results, f, protocol=HIGHEST_PROTOCOL)
        
        logger.info(f"📊 Детальные результаты сохранены в: {results_file}")

        # === Структурированное сохранение в result/<SYMBOL>/runs/<run_id>/ ===
        # Копируем артефакты в папку запуска с фиксированными именами
        try:
            import shutil as _sh
            # Модель
            if cfg and getattr(cfg, 'model_path', None) and os.path.exists(cfg.model_path):
                _dst_m = os.path.join(run_dir, 'model.pth')
                if os.path.abspath(cfg.model_path) != os.path.abspath(_dst_m):
                    _sh.copy2(cfg.model_path, _dst_m)
            # Буфер
            buffer_path = getattr(cfg, 'replay_buffer_path', getattr(cfg, 'buffer_path', None))
            if cfg and buffer_path and os.path.exists(buffer_path):
                _dst_b = os.path.join(run_dir, 'replay.pkl')
                if os.path.abspath(buffer_path) != os.path.abspath(_dst_b):
                    _sh.copy2(buffer_path, _dst_b)
            # Результаты
            if os.path.exists(results_file):
                _dst_r = os.path.join(run_dir, 'train_result.pkl')
                if os.path.abspath(results_file) != os.path.abspath(_dst_r):
                    _sh.copy2(results_file, _dst_r)
        except Exception as _copy_err:
            logger.warning(f"⚠️ Не удалось скопировать артефакты в {run_dir}: {_copy_err}")

        # Пишем манифест (минимум метаданных; подробности уже в train_result.pkl)
        symbol_dir_name = training_name # Используем training_name, который уже обработан
        manifest = {
            'run_id': getattr(cfg, 'run_id', None) or (run_dir.split(os.sep)[-1] if run_dir else None),
            'parent_run_id': parent_run_id,
            'root_id': root_id,
            'symbol': symbol_dir_name,
            'seed': int(seed) if isinstance(seed, int) else None,
            'episodes_start': 0 if not (getattr(cfg, 'load_model_path', None) or getattr(cfg, 'load_buffer_path', None)) else None, # Если модель или буфер загружались
            'episodes_end': int(current_episode),
            'episodes_added': int(current_episode) if not (getattr(cfg, 'load_model_path', None) or getattr(cfg, 'load_buffer_path', None)) else int(current_episode - (getattr(cfg, 'start_episode', 0))), # Считаем добавленные эпизоды
            'episodes_last': int(current_episode),
            'episodes_best': int(best_episode_idx) if best_episode_idx is not None and best_episode_idx >= 0 else None,
            'training_time_sec': float(total_training_time) if isinstance(total_training_time, (int, float)) else None,
            'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'artifacts': {
                'model': 'model.pth',
                'replay': 'replay.pkl' if (dqn_solver and getattr(dqn_solver, 'cfg', None) and getattr(dqn_solver.cfg, 'buffer_path', None)) else None,
                'result': 'train_result.pkl',
                'best_model': 'best_model.pth' if os.path.exists(os.path.join(run_dir, 'best_model.pth')) else None,
                'last_model': 'last_model.pth' if os.path.exists(os.path.join(run_dir, 'last_model.pth')) else None,
                'encoder': getattr(cfg, 'encoder_path', None)
            },
            'best_metrics': {
                'winrate': float(best_winrate) if isinstance(best_winrate, (int, float)) else None
            },
            'trained_as': (direction or 'long'),
            'direction': (direction or 'long'),
        }
        # Добавляем компактный снэпшот adaptive params в manifest.json
        try:
            if isinstance(adaptive_snapshot, dict) and adaptive_snapshot.get('trading_params'):
                manifest['adaptive_params'] = adaptive_snapshot.get('trading_params')
        except Exception:
            pass
        try:
            with open(os.path.join(run_dir, 'manifest.json'), 'w', encoding='utf-8') as mf:
                _json.dump(manifest, mf, ensure_ascii=False, indent=2)
        except Exception as _mf_err:
            logger.warning(f"⚠️ Не удалось записать manifest.json: {_mf_err}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении результатов обучения: {e}", exc_info=True)


def train_model_optimized(
    dfs: Dict,
    cfg: Optional[vDqnConfig] = None,
    episodes: int = 10,
    patience_limit: int = 3000,  # Увеличено с 2000 до 3000 для более длительного обучения
    use_wandb: bool = False,
    load_model_path: Optional[str] = None,
    load_buffer_path: Optional[str] = None,
    seed: Optional[int] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    root_id: Optional[str] = None,
    episode_length: Optional[int] = None,
    direction: str = 'long',
    env_overrides: Optional[Dict] = None,
    env_class_override: Optional[str] = None,
) -> str:
    """
    Оптимизированная функция тренировки модели без pandas в hot-path
    
    Args:
        dfs: словарь с DataFrame для разных таймфреймов (df_5min, df_15min, df_1h)
        cfg: конфигурация модели
        episodes: количество эпизодов для тренировки
        patience_limit: лимит терпения для early stopping (по умолчанию 2000 эпизодов)
        use_wandb: использовать ли Weights & Biases
        
    Returns:
        str: сообщение о завершении тренировки
    """
    
    # Инициализация wandb
    wandb_run = None
    if use_wandb:
        try:
            run_name = getattr(cfg, 'run_name', 'default') if cfg else 'default'
            config_dict = cfg.__dict__ if cfg else {}
            
            wandb_run = wandb.init(
                project="medoedai-optimized",
                name=f"vDQN-optimized-{run_name}",
                config=config_dict
            )
        except Exception as e:
            logger.warning(f"Не удалось инициализировать wandb: {e}")
            use_wandb = False
    
    try:
        # Инициализируем файловый лог с триммингом (DDD infra)
        run_dir = None
        # Проверяем и создаем конфигурацию по умолчанию
        if cfg is None:
            cfg = vDqnConfig()
            print("⚠️ Конфигурация не передана, использую конфигурацию по умолчанию")
        
        # Проверяем тип данных: мультивалютные или одиночные
        is_multi_crypto = False
        if dfs and isinstance(dfs, dict):
            # Проверяем, есть ли ключи с названиями криптовалют
            first_key = list(dfs.keys())[0]
            if isinstance(first_key, str) and first_key.endswith('USDT'):
                # Это мультивалютные данные
                is_multi_crypto = True
                print(f"🌍 Обнаружены мультивалютные данные для {len(dfs)} криптовалют")
                for symbol, data in dfs.items():
                    print(f"  • {symbol}: {data.get('candle_count', 'N/A')} свечей")
        
        if is_multi_crypto:
            # Используем мультивалютное окружение
            from envs.dqn_model.gym.crypto_trading_env_multi import MultiCryptoTradingEnv
            env = MultiCryptoTradingEnv(dfs=dfs, cfg=cfg, episode_length=episode_length)
            print(f"✅ Создано мультивалютное окружение для {len(dfs)} криптовалют")
        else:
            # Используем обычное окружение для одной криптовалюты
            dfs = prepare_data_for_training(dfs)
            # --- SYMBOL-SPECIFIC OVERRIDES ---
            crypto_symbol = None
            try:
                if isinstance(dfs, dict):
                    crypto_symbol = dfs.get('symbol') or dfs.get('SYMBOL')
                if not crypto_symbol:
                    # попытка из данных
                    crypto_symbol = 'TONUSDT' if 'TON' in str(dfs).upper() else None
            except Exception:
                crypto_symbol = None

            override = get_symbol_override(crypto_symbol) if crypto_symbol else None
            # применяем training_params к cfg
            if override and 'training_params' in override:
                # GPU-owned параметры НЕ должны быть per-symbol.
                # Иначе сравнения между машинами/прогонами становятся непрозрачными,
                # а "ИИ-правки" могут случайно сломать hardware-профиль.
                GPU_OWNED_KEYS = {
                    'batch_size',
                    'memory_size',
                    'train_repeats',
                    'use_amp',
                    'use_gpu_storage',
                    'use_torch_compile',
                }
                for k, v in override['training_params'].items():
                    if hasattr(cfg, k):
                        try:
                            if k in GPU_OWNED_KEYS:
                                try:
                                    print(f"⚠️ SYMBOL OVERRIDE[{crypto_symbol}] игнорирует GPU-owned параметр: {k}={v}")
                                except Exception:
                                    pass
                                continue
                            setattr(cfg, k, v)
                        except Exception:
                            pass
                try:
                    print(f"🔧 SYMBOL OVERRIDE[{crypto_symbol}] | lr={getattr(cfg,'lr',None)} | eps=({getattr(cfg,'eps_start',None)}→{getattr(cfg,'eps_final',None)}) | decay={getattr(cfg,'eps_decay_steps',None)} | batch={getattr(cfg,'batch_size',None)} | mem={getattr(cfg,'memory_size',None)} | repeats={getattr(cfg,'train_repeats',None)} | soft_every={getattr(cfg,'soft_update_every',None)} | target_freq={getattr(cfg,'target_update_freq',None)}")
                except Exception:
                    pass

            # indicators_config для env
            indicators_config = None
            if override and 'indicators_config' in override:
                indicators_config = override['indicators_config']

            # Создаем GymConfig для получения значения по умолчанию
            gym_cfg = GymConfig()
            # Feature flag: state-based action masking (default OFF).
            # Allow enabling via symbol override gym_config.use_state_action_mask
            try:
                if override and isinstance(override.get('gym_config', None), dict):
                    og = override.get('gym_config', {}) or {}
                    # small allowlist of safe gym-config overrides
                    for k in ('use_state_action_mask', 'micro_profit_tau_mult', 'micro_profit_tau_min'):
                        if k in og and hasattr(gym_cfg, k):
                            try:
                                if k == 'use_state_action_mask':
                                    setattr(gym_cfg, k, bool(og.get(k)))
                                else:
                                    setattr(gym_cfg, k, float(og.get(k)))
                            except Exception:
                                pass
            except Exception:
                pass
            # Feature flag from Postgres (app_settings) via /settings.
            # Priority: per-symbol (group=<SYMBOL>) -> global (group=None) -> overrides/defaults.
            try:
                def _to_bool(v):
                    if v is None:
                        return None
                    s = str(v).strip().lower()
                    if s in ('1', 'true', 'yes', 'on'):
                        return True
                    if s in ('0', 'false', 'no', 'off'):
                        return False
                    return None
                v_sym = _to_bool(_get_setting_value('rl', (crypto_symbol or None), 'USE_STATE_ACTION_MASK'))
                v_glob = _to_bool(_get_setting_value('rl', None, 'USE_STATE_ACTION_MASK'))
                if v_sym is not None:
                    gym_cfg.use_state_action_mask = bool(v_sym)
                elif v_glob is not None:
                    gym_cfg.use_state_action_mask = bool(v_glob)
            except Exception:
                pass
            if env_class_override == 'stock':
                from envs.stock_trading_env import StockTradingEnv, StockGymConfig
                stock_cfg = StockGymConfig(
                    episode_length=episode_length or 2000,
                    lookback_window=override.get('gym_config', {}).get('lookback_window', 144) if override else 144,
                )
                base_env = StockTradingEnv(
                    dfs=dfs, cfg=stock_cfg,
                    lookback_window=stock_cfg.lookback_window,
                    indicators_config=indicators_config,
                    episode_length=stock_cfg.episode_length,
                )
            elif (direction or 'long') == 'short':
                base_env = CryptoTradingEnvShort(
                    dfs=dfs,
                    cfg=gym_cfg,
                    lookback_window=override.get('gym_config', {}).get('lookback_window', gym_cfg.lookback_window) if override else gym_cfg.lookback_window,
                    indicators_config=indicators_config,
                    episode_length=episode_length or gym_cfg.episode_length
                )
            else:
                base_env = CryptoTradingEnvOptimized(
                    dfs=dfs,
                    cfg=gym_cfg,
                    lookback_window=override.get('gym_config', {}).get('lookback_window', gym_cfg.lookback_window) if override else gym_cfg.lookback_window,
                    indicators_config=indicators_config,
                    episode_length=episode_length or gym_cfg.episode_length
                )
            # Оборачиваем env продлением эпизода на +100 шагов при открытой позиции
            policy = EpisodeExtensionPolicy(max_extension=20, extension_steps=100)
            env = PositionAwareEpisodeWrapper(base_env, policy=policy)

            # --- Global overrides (shared for all symbols) ---
            # Apply AFTER env init (overrides adaptive_normalization / env defaults).
            try:
                gov = GLOBAL_OVERRIDES if isinstance(GLOBAL_OVERRIDES, dict) else {}
                rm = gov.get('risk_management') if isinstance(gov.get('risk_management', None), dict) else {}
                ps = gov.get('position_sizing') if isinstance(gov.get('position_sizing', None), dict) else {}

                for k, v in rm.items():
                    set_env_attr_safe(env, str(k), v)
                for k, v in ps.items():
                    set_env_attr_safe(env, str(k), v)

                if rm:
                    print(
                        f"🧩 GLOBAL OVERRIDES | "
                        f"SL={get_env_attr_safe(env,'STOP_LOSS_PCT')} | "
                        f"TP={get_env_attr_safe(env,'TAKE_PROFIT_PCT')} | "
                        f"minHold={get_env_attr_safe(env,'min_hold_steps')} | "
                        f"volThr={get_env_attr_safe(env,'volume_threshold')}"
                    )
            except Exception:
                pass

            # risk_management per-symbol (optional override)
            if override and 'risk_management' in override:
                rm = override['risk_management']
                for field_name, env_attr in [
                    ('STOP_LOSS_PCT', 'STOP_LOSS_PCT'),
                    ('TAKE_PROFIT_PCT', 'TAKE_PROFIT_PCT'),
                    ('min_hold_steps', 'min_hold_steps'),
                    ('volume_threshold', 'volume_threshold'),
                ]:
                    if field_name in rm:
                        set_env_attr_safe(env, env_attr, rm[field_name])
                try:
                    print(f"🔧 RISK OVERRIDE[{crypto_symbol}] | SL={get_env_attr_safe(env,'STOP_LOSS_PCT')} | TP={get_env_attr_safe(env,'TAKE_PROFIT_PCT')} | minHold={get_env_attr_safe(env,'min_hold_steps')} | volThr={get_env_attr_safe(env,'volume_threshold')}")
                except Exception:
                    pass

            # --- Per-run overrides (grid etc), applied OVER global + per-symbol ---
            try:
                if isinstance(env_overrides, dict):
                    rm2 = env_overrides.get('risk_management')
                    if isinstance(rm2, dict):
                        for field_name, env_attr in [
                            ('STOP_LOSS_PCT', 'STOP_LOSS_PCT'),
                            ('TAKE_PROFIT_PCT', 'TAKE_PROFIT_PCT'),
                            ('min_hold_steps', 'min_hold_steps'),
                            ('volume_threshold', 'volume_threshold'),
                            # Buy-filters knobs (env attributes)
                            ('buy_roi_thr', 'buy_roi_thr'),
                            ('buy_trend_thr', 'buy_trend_thr'),
                            ('buy_volat_thr', 'buy_volat_thr'),
                            ('buy_strictness_floor', 'buy_strictness_floor'),
                            ('buy_vol_min_lenient', 'buy_vol_min_lenient'),
                            ('buy_vol_min_strict', 'buy_vol_min_strict'),
                            ('buy_vol_floor_mult', 'buy_vol_floor_mult'),
                        ]:
                            if field_name in rm2:
                                set_env_attr_safe(env, env_attr, rm2[field_name])
                        # atr_trail_mult / atr_sl_mult → пишем в cfg объект env
                        try:
                            cfg_obj = get_env_attr_safe(env, 'cfg')
                            if cfg_obj is not None:
                                if 'atr_trail_mult' in rm2:
                                    cfg_obj.atr_trail_mult = float(rm2['atr_trail_mult'])
                                if 'atr_sl_mult' in rm2:
                                    cfg_obj.atr_sl_mult = float(rm2['atr_sl_mult'])
                                if 'entry_confidence_gate' in rm2:
                                    cfg_obj.entry_confidence_gate = float(rm2['entry_confidence_gate'])
                        except Exception:
                            pass
                        print(
                            f"🧪 GRID OVERRIDE | SL={get_env_attr_safe(env,'STOP_LOSS_PCT')} | "
                            f"TP={get_env_attr_safe(env,'TAKE_PROFIT_PCT')} | "
                            f"minHold={get_env_attr_safe(env,'min_hold_steps')} | "
                            f"volThr={get_env_attr_safe(env,'volume_threshold')} | "
                            f"trail_mult={rm2.get('atr_trail_mult', '—')} | "
                            f"atr_sl_mult={rm2.get('atr_sl_mult', '—')} | "
                            f"gate={rm2.get('entry_confidence_gate', '—')} | "
                            f"roi_thr={rm2.get('buy_roi_thr', '—')} | "
                            f"trend_thr={rm2.get('buy_trend_thr', '—')} | "
                            f"volat_thr={rm2.get('buy_volat_thr', '—')} | "
                            f"strict_floor={rm2.get('buy_strictness_floor', '—')}"
                        )
            except Exception:
                pass
            print(f"✅ Создано обычное окружение для одной криптовалюты")
        
        # Начинаем отсчет времени тренировки
        training_start_time = time.time()

        # --- Снимок параметров окружения и данных (для воспроизводимости) ---
        gym_snapshot = {}
        try:
            df5 = dfs.get('df_5min') if isinstance(dfs, dict) else None
            funding_cols = ['funding_rate_bp', 'funding_rate_ema', 'funding_rate_change', 'funding_sign']
            funding_present = []
            try:
                if df5 is not None and hasattr(df5, 'columns'):
                    funding_present = [c for c in funding_cols if c in df5.columns]
            except Exception:
                funding_present = []

            # Базовые параметры окружения и риск-менеджмента
            cfg_obj2 = get_env_attr_safe(env, 'cfg')
            gym_snapshot = {
                'symbol': get_env_attr_safe(env, 'symbol'),
                'lookback_window': get_env_attr_safe(env, 'lookback_window'),
                'indicators_config': get_env_attr_safe(env, 'indicators_config'),
                'reward_scale': getattr(cfg_obj2, 'reward_scale', 1.0),
                'episode_length': get_env_attr_safe(env, 'episode_length'),
                'funding_features': {
                    'present_in_input_df': funding_present,
                    'included': bool(funding_present),
                },
                'risk_management': {
                    'STOP_LOSS_PCT': get_env_attr_safe(env, 'STOP_LOSS_PCT'),
                    'TAKE_PROFIT_PCT': get_env_attr_safe(env, 'TAKE_PROFIT_PCT'),
                    'min_hold_steps': get_env_attr_safe(env, 'min_hold_steps'),
                    'volume_threshold': get_env_attr_safe(env, 'volume_threshold'),
                    'base_stop_loss': get_env_attr_safe(env, 'base_stop_loss'),
                    'base_take_profit': get_env_attr_safe(env, 'base_take_profit'),
                    'base_min_hold': get_env_attr_safe(env, 'base_min_hold'),
                    'atr_trail_mult': getattr(cfg_obj2, 'atr_trail_mult', None),
                    'atr_sl_mult': getattr(cfg_obj2, 'atr_sl_mult', None),
                    # buy filters / gates (to reproduce throttling behavior)
                    'entry_confidence_gate': getattr(cfg_obj2, 'entry_confidence_gate', None),
                    'buy_roi_thr': get_env_attr_safe(env, 'buy_roi_thr'),
                    'buy_trend_thr': get_env_attr_safe(env, 'buy_trend_thr'),
                    'buy_volat_thr': get_env_attr_safe(env, 'buy_volat_thr'),
                    'buy_strictness_floor': get_env_attr_safe(env, 'buy_strictness_floor'),
                    'buy_vol_min_lenient': get_env_attr_safe(env, 'buy_vol_min_lenient'),
                    'buy_vol_min_strict': get_env_attr_safe(env, 'buy_vol_min_strict'),
                    'buy_vol_floor_mult': get_env_attr_safe(env, 'buy_vol_floor_mult'),
                },
                'position_sizing': {
                    'base_position_fraction': get_env_attr_safe(env, 'base_position_fraction'),
                    'position_fraction': get_env_attr_safe(env, 'position_fraction'),
                    'position_confidence_threshold': get_env_attr_safe(env, 'position_confidence_threshold'),
                },
                'observation_space_shape': get_env_attr_safe(env, 'observation_space_shape'),
                'step_minutes': getattr(cfg_obj2, 'step_minutes', 5),
            }
        except Exception:
            gym_snapshot = {}

        # Примечание: снапшот adaptive_normalization сохраняется централизованно в _save_training_results()
        # (там же обеспечена compact-форма без regime_precomputed). Здесь не считаем повторно.
        
        # Проверяем, что окружение правильно инициализировано
        if not hasattr(getattr(env, 'unwrapped', env), 'observation_space_shape'):
            # Попробуем вычислить размер состояния из observation_space
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
                set_env_attr_safe(env, 'observation_space_shape', env.observation_space.shape[0])
                print(f"⚠️ Вычислен размер состояния из observation_space: {get_env_attr_safe(env,'observation_space_shape')}")
            else:
                raise ValueError("Окружение не имеет observation_space_shape и не может быть вычислен")
        
        # Получаем символ криптовалюты для логирования
        if is_multi_crypto:
            crypto_symbol = "МУЛЬТИВАЛЮТА"  # Для мультивалютного окружения
            print(f"✅ Мультивалютное окружение создано, размер состояния: {get_env_attr_safe(env,'observation_space_shape')}")
        else:
            crypto_symbol = get_env_attr_safe(env, 'symbol', 'UNKNOWN')
            print(f"✅ Окружение создано для {crypto_symbol}, размер состояния: {get_env_attr_safe(env,'observation_space_shape')}")

        # Настраиваем директорию вывода и имена файлов под символ
        def _symbol_code(sym: str) -> str:
            if not isinstance(sym, str) or not sym:
                return "model"
            s = sym.upper().replace('/', '')
            for suffix in ["USDT", "USD", "USDC", "BUSD", "USDP"]:
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            s = s.lower() if s else "model"
            if s in ("мультивалюта", "multi", "multicrypto"):
                s = "multi"
            return s

        result_dir = os.path.join("result", "dqn")
        os.makedirs(result_dir, exist_ok=True)
        symbol_code = _symbol_code(crypto_symbol)
        # Короткий UUID для версионирования (по умолчанию)
        import uuid
        short_id = str(uuid.uuid4())[:4].lower()

        # Структурированный каталог результата: result/dqn/<SYMBOL>/runs/<run_id>/
        # Папка символа без суффикса (TON, BTC, BNB...) в верхнем регистре
        symbol_dir_name = _symbol_code(crypto_symbol).upper() if crypto_symbol else "UNKNOWN"
        # Короткий run_id (4 символа) если не передан
        this_run_id = run_id or str(uuid.uuid4())[:4]
        this_root_id = root_id or this_run_id
        run_dir = os.path.join("result", "dqn", symbol_dir_name, "runs", this_run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Перенаправляем stdout в файл train.log с ограничением 100 МБ (оставляем хвост)
        try:
            log_path = os.path.join(run_dir, 'train.log')
            sys.stdout = TailTruncatingTee(log_path, max_bytes=100*1024*1024)
        except Exception:
            pass

        # Код для артефактов по умолчанию
        artifacts_code = f"{symbol_code}_{short_id}"

        # Если это дообучение (есть исходный путь к весам) — добавим суффикс _update[<n>]
        try:
            if load_model_path and isinstance(load_model_path, str):
                base_name = os.path.basename(load_model_path)
                m = re.match(r"^dqn_model_(.+)\.pth$", base_name)
                if m:
                    base_code = m.group(1)
                    # Вычисляем корректный next-суффикс: _update, _update_2, _update_3, ...
                    # Избегаем _update_update и формата без подчёркивания
                    mm = re.match(r"^(.*?)(?:_update(?:_(\d+))?)?$", base_code)
                    if mm:
                        root = mm.group(1) or base_code
                        n_str = mm.group(2)
                        if base_code.endswith('_update') and not n_str:
                            # Было ..._update → станет ..._update_2
                            next_code = f"{root}_update_2"
                        elif n_str is not None:
                            next_code = f"{root}_update_{int(n_str)+1}"
                        else:
                            # Первый дообученный артефакт
                            next_code = f"{base_code}_update"
                    else:
                        next_code = f"{base_code}_update"

                    # Гарантируем уникальность, при коллизии инкрементируем номер
                    candidate = next_code
                    while (
                        os.path.exists(os.path.join(result_dir, f"dqn_model_{candidate}.pth")) or
                        os.path.exists(os.path.join(result_dir, f"replay_buffer_{candidate}.pkl")) or
                        os.path.exists(os.path.join(result_dir, f"train_result_{candidate}.pkl"))
                    ):
                        mm2 = re.match(r"^(.*?)(?:_update(?:_(\d+))?)?$", candidate)
                        if mm2:
                            root2 = mm2.group(1) or candidate
                            n2 = mm2.group(2)
                            if n2 is None:
                                candidate = f"{root2}_update_2"
                            else:
                                candidate = f"{root2}_update_{int(n2)+1}"
                        else:
                            candidate = candidate + "_update_2"

                    artifacts_code = candidate
        except Exception:
            pass

        # Подготавливаем НОВЫЕ пути сохранения: сразу в run_dir
        new_model_path = os.path.join(run_dir, 'model.pth')
        new_buffer_path = os.path.join(run_dir, 'replay.pkl')

        # --- Обработка выбора энкодера из dfs ---
        encoder_cfg = dfs.get('encoder') if isinstance(dfs, dict) else {}
        selected_encoder_id = None
        train_encoder_flag = True
        try:
            if isinstance(encoder_cfg, dict):
                val = encoder_cfg.get('id')
                selected_encoder_id = str(val).strip() if val else None
                train_encoder_flag = bool(encoder_cfg.get('train_encoder', True))
        except Exception:
            selected_encoder_id = None
            train_encoder_flag = True

        # freeze_encoder = not train_encoder
        try:
            setattr(cfg, 'freeze_encoder', not train_encoder_flag)
        except Exception:
            pass

        # Где искать выбранный энкодер (если указан):
        base_encoder_path = None
        base_encoder_root = None
        if selected_encoder_id:
            try:
                # result/dqn/<SYMBOL>/encoder/unfrozen/vN/encoder_only.pth
                cand1 = os.path.join("result", "dqn", symbol_dir_name, "encoder", "unfrozen", selected_encoder_id, "encoder_only.pth")
                # models/<symbol>/encoder/unfrozen/vN/encoder_only.pth
                cand2 = os.path.join("models", symbol_dir_name.lower(), "encoder", "unfrozen", selected_encoder_id, "encoder_only.pth")
                for pth in (cand1, cand2):
                    if os.path.exists(pth):
                        base_encoder_path = pth
                        base_encoder_root = 'result' if pth.startswith(os.path.join('result', 'dqn')) else 'models'
                        break
            except Exception:
                base_encoder_path = None
                base_encoder_root = None

        # Тип энкодера для новой версии/артефактов
        encoder_base = os.path.join("result", "dqn", symbol_dir_name, "encoder")
        encoder_type = "frozen" if getattr(cfg, 'freeze_encoder', False) else "unfrozen"
        encoder_type_dir = os.path.join(encoder_base, encoder_type)
        try:
            os.makedirs(encoder_type_dir, exist_ok=True)
        except Exception:
            pass

        # Создаём новую версию, только если тренируем энкодер или энкодер не выбран (новый)
        create_new_encoder = bool(train_encoder_flag or not selected_encoder_id)
        created_encoder_version = None
        new_encoder_path = None
        version_dir = None
        if create_new_encoder:
            # Определяем следующую версию vN (один раз на запуск)
            version = 1
            try:
                existing_versions = []
                for _d in os.listdir(encoder_type_dir):
                    _p = os.path.join(encoder_type_dir, _d)
                    if os.path.isdir(_p) and _d.startswith('v'):
                        try:
                            existing_versions.append(int(_d[1:]))
                        except Exception:
                            continue
                version = (max(existing_versions) + 1) if existing_versions else 1
            except Exception:
                version = 1
            version_dir = os.path.join(encoder_type_dir, f"v{version}")
            try:
                os.makedirs(version_dir, exist_ok=True)
            except Exception:
                pass
            new_encoder_path = os.path.join(version_dir, 'encoder_only.pth')
            created_encoder_version = version
            # Пробрасываем метаданные
            try:
                setattr(cfg, 'encoder_path', new_encoder_path)
                setattr(cfg, 'encoder_version', version)
                setattr(cfg, 'encoder_type', encoder_type)
            except Exception:
                pass
            try:
                print(f"🎯 Encoder target path: {new_encoder_path}")
            except Exception:
                pass
            # Если есть базовый энкодер (fine-tune) — скопируем как стартовую точку
            try:
                if base_encoder_path and os.path.exists(base_encoder_path):
                    import shutil as _sh
                    _sh.copy2(base_encoder_path, new_encoder_path)
            except Exception:
                pass
            # Предварительно создаём манифест-заглушку (будет перезаписан атомарно после сохранения весов)
            try:
                _atomic_write_json(
                    os.path.join(version_dir, 'encoder_manifest.json'),
                    {
                        'status': 'pending',
                        'symbol': symbol_dir_name,
                        'direction': (direction or 'long'),
                        'trained_as': (direction or 'long'),
                        'encoder_type': encoder_type,
                        'version': int(getattr(cfg, 'encoder_version', 0) or 0),
                        'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    }
                )
            except Exception:
                pass
        else:
            # Не создаём новую версию — используем выбранный энкодер как есть (frozen head)
            try:
                if base_encoder_path:
                    setattr(cfg, 'encoder_path', base_encoder_path)
                    setattr(cfg, 'encoder_version', None)
                    setattr(cfg, 'encoder_type', 'unfrozen')
            except Exception:
                pass

        # Если дообучаем из структурированного пути runs/... и parent/root не переданы — авто‑детект
        try:
            if (not parent_run_id) and load_model_path and isinstance(load_model_path, str):
                norm_path = load_model_path.replace('\\', '/')
                parts = norm_path.split('/')
                if len(parts) >= 4 and parts[-1] == 'model.pth' and 'runs' in parts:
                    runs_idx = parts.index('runs')
                    if runs_idx + 1 < len(parts):
                        parent_run_id = parts[runs_idx + 1]
                        # Прочитаем root из manifest.json родителя, если есть
                        try:
                            parent_dir = os.path.dirname(load_model_path)
                            mf_path = os.path.join(parent_dir, 'manifest.json')
                            if os.path.exists(mf_path):
                                import json as _json
                                with open(mf_path, 'r', encoding='utf-8') as mf:
                                    mf_data = _json.load(mf)
                                root_id = root_id or mf_data.get('root_id') or parent_run_id
                            else:
                                root_id = root_id or parent_run_id
                        except Exception:
                            root_id = root_id or parent_run_id
        except Exception:
            pass

        # Если символ BNB — мягкие оверрайды обучения для стабильности
        try:
            if not is_multi_crypto and isinstance(crypto_symbol, str) and 'BNB' in crypto_symbol.upper():
                # Снижаем exploration и lr, увеличиваем batch
                cfg.eps_start = min(getattr(cfg, 'eps_start', 1.0), 0.20)
                cfg.eps_final = max(getattr(cfg, 'eps_final', 0.01), 0.02)
                cfg.eps_decay_steps = int(getattr(cfg, 'eps_decay_steps', 1_000_000) * 0.75)
                cfg.batch_size = max(192, getattr(cfg, 'batch_size', 128))
                cfg.lr = min(getattr(cfg, 'lr', 1e-3), 2e-4)
                # Чуть реже сохраняем буфер, чтобы не тормозить I/O
                cfg.buffer_save_frequency = max(
                    400,
                    int(getattr(cfg, 'buffer_save_frequency', 800))
                )
                print(
                    f"🔧 BNB overrides: eps_start={cfg.eps_start}, eps_final={cfg.eps_final}, "
                    f"eps_decay_steps={cfg.eps_decay_steps}, batch_size={cfg.batch_size}, lr={cfg.lr}, "
                    f"buffer_save_frequency={getattr(cfg, 'buffer_save_frequency', 'n/a')}"
                )
        except Exception as _e:
            print(f"⚠️ Не удалось применить BNB-оверрайды: {_e}")

        # Создаем DQN solver
        print(f"🚀 Создаю DQN solver")
        
        dqn_solver = DQNSolver(
            observation_space=get_env_attr_safe(env, 'observation_space_shape'),
            action_space=env.action_space.n
        )
        # Если указан путь загрузки существующей модели/буфера — реально загружаем (continue-training)
        if load_model_path and isinstance(load_model_path, str):
            try:
                dqn_solver.cfg.model_path = load_model_path
            except Exception:
                pass
        if load_buffer_path and isinstance(load_buffer_path, str):
            try:
                # DQNSolver.load_state() читает cfg.buffer_path
                dqn_solver.cfg.replay_buffer_path = load_buffer_path
                dqn_solver.cfg.buffer_path = load_buffer_path
            except Exception:
                pass
        # Если внешняя cfg не передана — используем конфиг из dqn_solver
        if cfg is None:
            cfg = dqn_solver.cfg
        
        # 🚀 Дополнительная оптимизация PyTorch 2.x
        if torch.cuda.is_available():
            # Включаем cudnn benchmark для максимального ускорения
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Включаем TF32 для ускорения на Ampere+ GPU
            if hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                
            print("🚀 CUDA оптимизации включены: cudnn.benchmark, TF32")
        
        # Continue-training: загрузим веса/буфер ДО переназначения путей сохранения на новый run_dir
        try:
            did_load_any = False
            if load_model_path and isinstance(load_model_path, str):
                try:
                    print(f"🔁 Continue: loading model from {load_model_path}")
                except Exception:
                    pass
                try:
                    dqn_solver.load_model()
                    did_load_any = True
                except Exception as _e:
                    try:
                        print(f"⚠️ Continue: failed to load model: {_e}")
                    except Exception:
                        pass
            if load_buffer_path and isinstance(load_buffer_path, str):
                try:
                    print(f"🔁 Continue: loading replay buffer from {load_buffer_path}")
                except Exception:
                    pass
                try:
                    dqn_solver.load_state()
                    did_load_any = True
                except Exception as _e:
                    try:
                        print(f"⚠️ Continue: failed to load buffer: {_e}")
                    except Exception:
                        pass
            if not did_load_any:
                try:
                    print("🆕 Training: start from scratch (no continue paths provided)")
                except Exception:
                    pass
        except Exception:
            pass

        # После загрузки переназначаем пути сохранения на НОВЫЕ в result/<symbol>_<id>
        # Обновляем пути сохранения на структурированные независимо от наличия внешней cfg
        try:
            dqn_solver.cfg.model_path = new_model_path
            # ВАЖНО: DQNSolver.save() сохраняет по cfg.buffer_path, поэтому задаём оба пути
            dqn_solver.cfg.replay_buffer_path = new_buffer_path
            dqn_solver.cfg.buffer_path = new_buffer_path
            if hasattr(dqn_solver.cfg, 'encoder_path'):
                dqn_solver.cfg.encoder_path = new_encoder_path
        except Exception:
            pass
        try:
            if cfg is not None:
                cfg.model_path = dqn_solver.cfg.model_path
                cfg.replay_buffer_path = dqn_solver.cfg.replay_buffer_path
                # Для совместимости с потребляющим кодом используем одинаковый путь
                cfg.buffer_path = dqn_solver.cfg.replay_buffer_path
                if hasattr(cfg, 'encoder_path'):
                    cfg.encoder_path = dqn_solver.cfg.encoder_path
        except Exception:
            pass
        
        # Переменные для отслеживания прогресса
        all_trades = []
        episode_winrates = []
        episode_epsilons = []
        best_winrate = 0.0
        best_episode_idx = -1
        # Компактная история winrate (bounded memory)
        try:
            _wr_snapshot_every = int(getattr(cfg, 'winrate_snapshot_every', 50)) if cfg is not None else 50
        except Exception:
            _wr_snapshot_every = 50
        try:
            _wr_window_n = int(getattr(cfg, 'winrate_trend_window', 200)) if cfg is not None else 200
        except Exception:
            _wr_window_n = 200
        try:
            _wr_ema_alpha = float(getattr(cfg, 'winrate_ema_alpha', 0.05)) if cfg is not None else 0.05
        except Exception:
            _wr_ema_alpha = 0.05
        winrate_window = deque(maxlen=max(1, _wr_window_n))
        winrate_snapshots: list[dict] = []
        winrate_ema: float | None = None

        def _update_winrate_trend(ep: int, wr: float):
            nonlocal winrate_ema
            try:
                w = float(wr)
            except Exception:
                return
            winrate_window.append(w)
            if winrate_ema is None:
                winrate_ema = w
            else:
                a = float(_wr_ema_alpha)
                winrate_ema = (a * w) + ((1.0 - a) * float(winrate_ema))
            if _wr_snapshot_every > 0 and (ep % _wr_snapshot_every == 0):
                try:
                    arr = np.asarray(list(winrate_window), dtype=np.float32)
                    snap = {
                        'episode': int(ep),
                        'winrate': float(w),
                        'ema': float(winrate_ema) if winrate_ema is not None else None,
                        'median_window': float(np.median(arr)) if arr.size else None,
                        'p25_window': float(np.quantile(arr, 0.25)) if arr.size else None,
                        'p75_window': float(np.quantile(arr, 0.75)) if arr.size else None,
                        'window_n': int(arr.size),
                    }
                    winrate_snapshots.append(snap)
                except Exception:
                    pass
         # Reduce-on-plateau и warmup для best
        lr_plateau_patience = int(getattr(cfg, 'lr_plateau_patience', 1000))
        lr_min = float(getattr(cfg, 'lr_min', 1e-5))
        best_warmup_episodes = int(getattr(cfg, 'best_warmup_episodes', 1500))
        reduce_plateau_only_for_retrain = bool(getattr(cfg, 'reduce_on_plateau_only_for_retrain', True))
        episodes_since_best = 0
        patience_counter = 0
        global_step = 0
        grad_steps = 0
        actual_episodes = episodes  # ИСПРАВЛЕНИЕ: Переменная для отслеживания реального количества эпизодов
        
        # Принудительное exploration в начале для Noisy Networks
        if getattr(cfg, 'use_noisy_networks', True):
            dqn_solver.epsilon = 0.3  # Начинаем с 30% exploration
            #print(f"🔀 Noisy Networks: принудительное exploration с epsilon={dqn_solver.epsilon}")
        
        # Улучшенные переменные для early stopping (УЛУЧШЕНО)
        min_episodes_before_stopping = getattr(cfg, 'min_episodes_before_stopping', max(4000, episodes // 3))  # Увеличил с 3000 до 4000 и с 1/4 до 1/3
        winrate_history = []  # История winrate для анализа трендов
        recent_improvement_threshold = 0.002  # Увеличил с 0.001 до 0.002 для более стабильного обучения
        
        # --- Расширенные агрегаты для анализа поведения ---
        action_counts_total = {0: 0, 1: 0, 2: 0}
        buy_attempts_total = 0
        buy_rejected_vol_total = 0
        buy_rejected_roi_total = 0
        episodes_with_trade_count = 0
        total_steps_processed = 0
        # --- Метрики для freeze‑решений (лёгкие по ресурсу) ---
        q_loss_history: list[float] = []  # история Q‑loss из experience_replay
        probe_states: list[np.ndarray] = []  # фиксированный набор состояний для мониторинга
        probe_size: int = int(getattr(cfg, 'probe_size', 64))
        probe_collect_episodes: int = int(getattr(cfg, 'probe_collect_episodes', 3))
        probe_collect_stride: int = int(getattr(cfg, 'probe_collect_stride', 100))  # брать состояние раз в N шагов
        probe_interval_episodes: int = int(getattr(cfg, 'probe_interval_episodes', 10))  # как часто делать снэпшоты
        drift_cosine_history: list[float] = []  # средняя косинусная близость к базовой проекции
        drift_snapshot_episodes: list[int] = []
        q_values_history: list[np.ndarray] = []  # средние Q по действиям для probe‑состояний
        probe_embeddings_baseline: np.ndarray | None = None
        
        # Адаптивный patience_limit в зависимости от количества эпизодов
        if episodes >= 10000:
            patience_limit = max(patience_limit, episodes // 3)  # Для очень длинных тренировок - минимум 1/3 (было 1/2)
        elif episodes >= 5000:
            patience_limit = max(patience_limit, episodes // 4)  # Для длинных тренировок - минимум 1/4 (было 1/3)
        elif episodes >= 2000:
            patience_limit = max(patience_limit, episodes // 3)  # Для средних тренировок - минимум 1/3 (было 1/2)
        
        # Увеличиваем patience для длинных тренировок
        patience_limit = max(patience_limit, 8000)  # Минимум 8000 эпизодов (было 5000)
        
        long_term_patience = int(patience_limit * getattr(cfg, 'long_term_patience_multiplier', 2.5))
        trend_threshold = getattr(cfg, 'early_stopping_trend_threshold', 0.05)  # Увеличиваем порог тренда с 0.03 до 0.05
        
        # Определяем название для логирования
        training_name = "МУЛЬТИВАЛЮТА" if is_multi_crypto else crypto_symbol
        print(f"🎯 Начинаю тренировку на {episodes} эпизодов для {training_name}")
        print(f"📊 Параметры Early Stopping:")
        print(f"  • min_episodes_before_stopping: {min_episodes_before_stopping}")
        print(f"  • patience_limit: {patience_limit}")
        print(f"  • long_term_patience: {long_term_patience}")
        print(f"  • trend_threshold: {trend_threshold}")
        print(f"  • Самый ранний stopping: {min_episodes_before_stopping + patience_limit} эпизодов")            
        # Информация о настройках сохранения
        save_frequency = getattr(cfg, 'save_frequency', 50)
        save_only_on_improvement = getattr(cfg, 'save_only_on_improvement', False)

        # Частота логирования ресурсов (каждые N эпизодов)
        try:
            resource_log_every = int(os.getenv('TRAIN_LOG_EVERY_EPISODES', '100'))
        except Exception:
            resource_log_every = 100

        # Feature flag from Postgres (app_settings) via /settings for multi-crypto runs too.
        # MultiCryptoTradingEnv receives vDqnConfig (cfg) as env.cfg, env reads getattr(cfg,'use_state_action_mask', False).
        try:
            def _to_bool(v):
                if v is None:
                    return None
                s = str(v).strip().lower()
                if s in ('1', 'true', 'yes', 'on'):
                    return True
                if s in ('0', 'false', 'no', 'off'):
                    return False
                return None
            v_glob = _to_bool(_get_setting_value('rl', None, 'USE_STATE_ACTION_MASK'))
            if v_glob is not None:
                setattr(cfg, 'use_state_action_mask', bool(v_glob))
        except Exception:
            pass

        # Основной цикл тренировки
        for episode in range(episodes):
            # Логируем загрузку ресурсов по частоте
            if resource_log_every > 0 and (episode % resource_log_every == 0):
                log_resource_usage(tag="episode", extra=f"episode={episode}/{episodes}")

            state = env.reset()            
            # Убеждаемся, что state является numpy массивом
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            episode_reward = 0
            
            # Получаем текущую криптовалюту для мультивалютного окружения
            current_crypto = crypto_symbol
            if is_multi_crypto and hasattr(env, 'current_symbol'):
                current_crypto = env.current_symbol
            
            print(f"  🎯 Эпизод {episode} для {current_crypto} начат, reward={episode_reward}")
            # Фиксируем ε, использованный в этом эпизоде (до обновления)
            try:
                episode_epsilons.append(float(dqn_solver.epsilon))
            except Exception:
                try:
                    episode_epsilons.append(None)
                except Exception:
                    pass
            
            # Эпизод
            step_count = 0
            failed_train_attempts = 0
            while True:
                step_count += 1
                # Показываем прогресс каждые 100 шагов для ускорения
                if step_count % 10000 == 0:
                    print(f"    🔄 Step {step_count} в эпизоде {episode}")
                
                env.epsilon = dqn_solver.epsilon
                
                # Gymnasium: prefer wrapper-safe access to avoid deprecation warnings
                action_mask = None
                try:
                    if hasattr(env, 'get_wrapper_attr'):
                        fn = env.get_wrapper_attr('get_action_mask')
                        action_mask = fn() if callable(fn) else fn
                    elif hasattr(env, 'get_action_mask'):
                        action_mask = env.get_action_mask()
                except Exception:
                    action_mask = None
                action = dqn_solver.act(state, action_mask=action_mask)
                state_next, reward, terminal, info = env.step(action)
                
                # Проверяем next_state на NaN
                if isinstance(state_next, (list, tuple)):
                    state_next = np.array(state_next, dtype=np.float32)
                elif not isinstance(state_next, np.ndarray):
                    state_next = np.array(state_next, dtype=np.float32)
                
                # Безопасная проверка на NaN
                try:
                    if np.isnan(state_next).any():
                        state_next = np.nan_to_num(state_next, nan=0.0)
                except (TypeError, ValueError):
                    # Если не можем проверить на NaN, преобразуем в numpy и попробуем снова (без спама в лог)
                    state_next = np.array(state_next, dtype=np.float32)
                    if np.isnan(state_next).any():
                        state_next = np.nan_to_num(state_next, nan=0.0)
                
                # Сохраняем переход в replay buffer
                dqn_solver.store_transition(state, action, reward, state_next, terminal)
                
                # Получаем n-step transitions и добавляем их в replay buffer
                # Только если эпизод не завершен (не terminal)
                if not terminal:
                    n_step_transitions = env.get_n_step_return()
                    if n_step_transitions:
                        dqn_solver.memory.push_n_step(n_step_transitions)
                

                # Обновляем состояние
                state = state_next
                
                # Убеждаемся, что обновленный state является numpy массивом
                if isinstance(state, (list, tuple)):
                    state = np.array(state, dtype=np.float32)
                elif not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                
                episode_reward += reward
                global_step += 1
                total_steps_processed += 1
                
                # Обучаем модель чаще для лучшего обучения (УЛУЧШЕНО)
                soft_update_every = getattr(cfg, 'soft_update_every', 50)   # частота попыток обучения
                batch_size = getattr(cfg, 'batch_size', 128)
                target_update_freq = getattr(cfg, 'target_update_freq', 500)
                train_repeats = max(1, int(getattr(cfg, 'train_repeats', 1)))
                
                if global_step % soft_update_every == 0 and len(dqn_solver.memory) >= batch_size:
                    # Выполним несколько градиентных шагов подряд для лучшей загрузки CPU
                    for _ in range(train_repeats):
                        success, loss, abs_q, q_gap = dqn_solver.experience_replay(need_metrics=True)
                        if success:
                            grad_steps += 1
                            # Копим историю Q‑loss для оценки стабилизации обучения
                            try:
                                if loss is not None:
                                    q_loss_history.append(float(loss))
                            except Exception:
                                pass
                        else:
                            failed_train_attempts += 1
                            break
                    # Обновляем target network по расписанию
                    if global_step % target_update_freq == 0:
                        dqn_solver.update_target_model()

                if terminal:
                    break
                # Сбор probe‑состояний для дальнейшего мониторинга (лёгкий, редкий)
                try:
                    if (len(probe_states) < probe_size) and (episode < probe_collect_episodes) and (step_count % max(1, probe_collect_stride) == 1):
                        if isinstance(state, np.ndarray):
                            probe_states.append(state.copy())
                        else:
                            probe_states.append(np.array(state, dtype=np.float32))
                except Exception:
                    pass
            
            # Обновляем epsilon (только если не используем Noisy Networks)
            if not getattr(cfg, 'use_noisy_networks', True):
                eps_final = getattr(cfg, 'eps_final', 0.01)  # По умолчанию минимальный epsilon 0.01
                dqn_solver.epsilon = max(eps_final, dqn_solver.epsilon * dqn_solver._eps_decay_rate)
            else:
                # При использовании Noisy Networks оставляем небольшой epsilon для стабильности
                dqn_solver.epsilon = max(0.05, dqn_solver.epsilon * 0.999)  # Минимум 5%
            
            # Собираем статистику эпизода
            # РАДИКАЛЬНОЕ ИСПРАВЛЕНИЕ: Используем env.all_trades для расчета winrate
            trades_before = len(all_trades)
            
            # ИСПРАВЛЕНИЕ: Получаем сделки через безопасный доступ
            _all_trades = get_env_attr_safe(env, 'all_trades') or []
            if _all_trades:
                episode_trades = _all_trades
            else:
                # Fallback: используем env.trades
                episode_trades = get_env_attr_safe(env, 'trades', []) or []
            
            # ИСПРАВЛЕНИЕ: Инициализируем episode_winrate по умолчанию
            episode_winrate = 0.0
            
            if _all_trades:
                # Используем все сделки из окружения для расчета winrate
                all_profitable = [t for t in _all_trades if t.get('roi', 0) > 0]
                episode_winrate = len(all_profitable) / len(_all_trades) if _all_trades else 0
                episode_winrates.append(episode_winrate)
                _update_winrate_trend(episode, episode_winrate)
                
                # Детальная статистика эпизода
                episode_stats = dqn_solver.print_trade_stats(_all_trades, failed_attempts=failed_train_attempts)
                
                # Добавляем сделки в общий список если их там нет
                if len(all_trades) < len(_all_trades):
                    all_trades.extend(_all_trades[len(all_trades):])
                    
            elif episode_trades:
                # Fallback: используем env.trades
                all_trades.extend(episode_trades)
                
                # Вычисляем winrate для эпизода
                profitable_trades = [t for t in episode_trades if t.get('roi', 0) > 0]
                episode_winrate = len(profitable_trades) / len(episode_trades) if episode_trades else 0
                episode_winrates.append(episode_winrate)
                _update_winrate_trend(episode, episode_winrate)
                
                # Детальная статистика эпизода
                episode_stats = dqn_solver.print_trade_stats(episode_trades, failed_attempts=failed_train_attempts)
            else:
                # Если нет сделок вообще, используем последние сделки из all_trades
                if len(all_trades) > 0:
                    # Берем последние сделки для расчета winrate
                    recent_trades = all_trades[-min(10, len(all_trades)):]  # Последние 10 сделок
                    profitable_trades = [t for t in recent_trades if t.get('roi', 0) > 0]
                    episode_winrate = len(profitable_trades) / len(recent_trades) if recent_trades else 0
                    episode_winrates.append(episode_winrate)
                    _update_winrate_trend(episode, episode_winrate)
                    episode_stats = dqn_solver.print_trade_stats(recent_trades, failed_attempts=failed_train_attempts)
                else:
                    # Нет сделок вовсе — выводим агрегатную строку статистики с Failed train
                    episode_winrate = 0.0
                    episode_winrates.append(episode_winrate)
                    _update_winrate_trend(episode, episode_winrate)
                    episode_stats = dqn_solver.print_trade_stats([], failed_attempts=failed_train_attempts)
                
                # Объединяем всю статистику эпизода в одну строку
                action_stats = ""
                _ac = get_env_attr_safe(env, 'action_counts')
                if isinstance(_ac, dict):
                    action_stats = f" | HOLD={_ac.get(0, 0)}, BUY={_ac.get(1, 0)}, SELL={_ac.get(2, 0)}"
                
                # Добавляем информацию о времени выполнения
                time_stats = ""
                _est = get_env_attr_safe(env, 'episode_start_time')
                _esc = get_env_attr_safe(env, 'episode_step_count', 0)
                if _est is not None:
                    episode_duration = time.time() - _est
                    steps_per_second = _esc / episode_duration if episode_duration > 0 else 0
                    time_stats = f" | {episode_duration:.2f}с, {_esc} шагов, {steps_per_second:.1f} шаг/с"
                
                print(f"  🏁 Эпизод {episode} для {current_crypto} завершен | reward={episode_reward:.4f}{action_stats}{time_stats} | {episode_stats}")

                # Периодический снэпшот feature drift и стабильности Q на фиксированном probe‑наборе
                try:
                    if (len(probe_states) > 0) and (episode % max(1, probe_interval_episodes) == 0):
                        dqn_solver.model.eval()
                        with torch.no_grad():
                            st = torch.from_numpy(np.stack(probe_states)).float().to(dqn_solver.cfg.device)
                            # Q‑values snapshot
                            q_vals = dqn_solver.model(st)
                            if isinstance(q_vals, torch.Tensor):
                                q_avg = q_vals.mean(dim=1).detach().float().cpu().numpy()
                                q_values_history.append(q_avg)
                            # Feature embeddings snapshot (если есть extractor)
                            cos_sim_mean = None
                            if hasattr(dqn_solver.model, 'get_feature_extractor'):
                                fe = dqn_solver.model.get_feature_extractor()
                                if fe is not None:
                                    try:
                                        z = fe(st)
                                        if isinstance(z, torch.Tensor):
                                            z_np = z.detach().float().cpu().numpy()
                                            # Нормируем
                                            def _l2norm(x: np.ndarray) -> np.ndarray:
                                                n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
                                                return x / n
                                            z_np = _l2norm(z_np)
                                            if probe_embeddings_baseline is None:
                                                probe_embeddings_baseline = z_np.copy()
                                                cos_sim_mean = 1.0
                                            else:
                                                base = _l2norm(probe_embeddings_baseline)
                                                # Косинусная похожесть по батчу и среднее
                                                cos = np.sum(z_np * base, axis=1)
                                                cos_sim_mean = float(np.mean(cos))
                                    except Exception:
                                        pass
                            if cos_sim_mean is not None:
                                drift_cosine_history.append(cos_sim_mean)
                                drift_snapshot_episodes.append(int(episode))
                except Exception:
                    pass
                
                # Проверяем на улучшение с более умной логикой
                # Считаем улучшением только после warmup
                is_improvement = (episode >= best_warmup_episodes) and (episode_winrate > best_winrate)
                if is_improvement:
                    best_winrate = episode_winrate
                    patience_counter = 0
                    episodes_since_best = 0
                    
                    # Сохраняем модель, а по настройке — и replay buffer при улучшении
                    save_replay_on_improvement = getattr(cfg, 'save_replay_on_improvement', True)
                    if save_replay_on_improvement:
                        dqn_solver.save()
                        logger.info("[INFO] New best winrate: %.3f, saving model + replay buffer", best_winrate)
                    else:
                        dqn_solver.save_model()
                        logger.info("[INFO] New best winrate: %.3f, saving model", best_winrate)
                    # Дублируем текущую модель как best_model.pth и фиксируем эпизод
                    try:
                        import shutil as _sh
                        if os.path.exists(cfg.model_path):
                            _sh.copy2(cfg.model_path, os.path.join(run_dir, 'best_model.pth'))
                            best_episode_idx = int(episode)
                    except Exception:
                        pass
                else:
                    # Мягкая логика patience - увеличиваем только при явном ухудшении
                    if episode >= min_episodes_before_stopping:
                        # Анализируем тренд winrate
                        if len(episode_winrates) >= 30:  # Увеличено с 20 до 30 для более стабильного анализа
                            recent_avg = np.mean(episode_winrates[-30:])  # Увеличено окно анализа
                            older_avg = np.mean(episode_winrates[-60:-30]) if len(episode_winrates) >= 60 else recent_avg  # Увеличено окно
                            
                            # Если есть стабильный тренд улучшения, сбрасываем patience
                            if recent_avg > older_avg + recent_improvement_threshold:
                                patience_counter = max(0, patience_counter - 5)  # Уменьшаем patience сильнее (было -3)
                            elif recent_avg > older_avg:
                                patience_counter = max(0, patience_counter - 2)  # Небольшое улучшение (было -1)
                            elif recent_avg < older_avg - 0.05:  # Увеличиваем порог ухудшения с 0.03 до 0.05
                                patience_counter += 1
                            # Если изменения небольшие, не меняем patience
                        else:
                            patience_counter += 0  # Не увеличиваем patience в начале
                    else:
                        # В начале обучения не увеличиваем patience
                        patience_counter = 0
                # Reduce-on-plateau: только для дообучения по умолчанию
                try:
                    episodes_since_best += 1
                except Exception:
                    episodes_since_best = lr_plateau_patience + 1
                is_retrain = bool(load_model_path) or bool(parent_run_id)
                allow_reduce = (not reduce_plateau_only_for_retrain) or is_retrain
                if allow_reduce and episodes_since_best >= lr_plateau_patience:
                    try:
                        # текущее lr из оптимизатора
                        current_lr = None
                        for g in dqn_solver.optimizer.param_groups:
                            current_lr = g.get('lr', None)
                            break
                        if current_lr is None:
                            current_lr = float(getattr(cfg, 'lr', 1e-3))
                        new_lr = max(lr_min, float(current_lr) * 0.5)
                        if new_lr < current_lr:
                            for g in dqn_solver.optimizer.param_groups:
                                g['lr'] = new_lr
                            try:
                                cfg.lr = new_lr
                            except Exception:
                                pass
                            print(f"🔧 Reduce-on-plateau: lr {current_lr:.6f} → {new_lr:.6f}")
                        # Снижаем epsilon к eps_final мягко
                        try:
                            eps_final = float(getattr(cfg, 'eps_final', 0.01))
                            dqn_solver.epsilon = max(eps_final, dqn_solver.epsilon * 0.7)
                            print(f"🔧 Reduce-on-plateau: epsilon → {dqn_solver.epsilon:.4f}")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    finally:
                        episodes_since_best = 0

            # --- Агрегируем поведенческие метрики эпизода ---
            try:
                # Суммарные действия
                _ac2 = get_env_attr_safe(env, 'action_counts')
                if isinstance(_ac2, dict):
                    action_counts_total[0] = action_counts_total.get(0, 0) + int(_ac2.get(0, 0) or 0)
                    action_counts_total[1] = action_counts_total.get(1, 0) + int(_ac2.get(1, 0) or 0)
                    action_counts_total[2] = action_counts_total.get(2, 0) + int(_ac2.get(2, 0) or 0)
                # Попытки покупок и причины отказов
                buy_attempts_total += int(get_env_attr_safe(env, 'buy_attempts', 0) or 0)
                buy_rejected_vol_total += int(get_env_attr_safe(env, 'buy_rejected_vol', 0) or 0)
                buy_rejected_roi_total += int(get_env_attr_safe(env, 'buy_rejected_roi', 0) or 0)
                # Была ли сделка в эпизоде
                new_trades_added = len(all_trades) - trades_before
                if new_trades_added > 0:
                    episodes_with_trade_count += 1
            except Exception:
                pass

            # Логируем прогресс и периодически сохраняем модель
            if episode % 10 == 0:
                avg_winrate = np.mean(episode_winrates[-10:]) if episode_winrates else 0
                
                # Получаем текущую криптовалюту для логирования
                log_crypto = current_crypto
                
                logger.info(f"[INFO] Episode {episode}/{episodes} для {log_crypto}, Avg Winrate: {avg_winrate:.3f}, Epsilon: {dqn_solver.epsilon:.4f}")
                
                # Показываем информацию о early stopping
                if episode >= min_episodes_before_stopping:
                    remaining_patience = patience_limit - patience_counter
                    print(f"  📊 Early stopping для {log_crypto}: patience {patience_counter}/{patience_limit} (осталось {remaining_patience})")
                    if patience_counter > patience_limit * 0.8:  # Показываем предупреждение при приближении к лимиту
                        print(f"  ⚠️ ВНИМАНИЕ: patience_counter приближается к лимиту!")                    
                
                # Очищаем GPU память каждые 10 эпизодов
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Периодическое сохранение модели и буфера
            save_frequency = getattr(cfg, 'save_frequency', 50)  # По умолчанию каждые 50 эпизодов
            save_only_on_improvement = getattr(cfg, 'save_only_on_improvement', False)
            buffer_save_frequency = getattr(cfg, 'buffer_save_frequency', max(200, save_frequency * 4))
            
            if not save_only_on_improvement and episode > 0 and episode % save_frequency == 0:
                dqn_solver.save_model()
                logger.info("[INFO] Periodic save model at episode %d", episode)
            
            if episode > 0 and episode % buffer_save_frequency == 0:
                dqn_solver.save()
                logger.info("[INFO] Periodic save model + replay buffer at episode %d", episode)
                # Сохраняем результаты обучения также
                _save_training_results(
                    run_dir=run_dir,
                    dfs=dfs, # Передаем dfs
                    cfg=cfg,
                    training_name=training_name,
                    current_episode=episode,
                    total_episodes_planned=episodes,
                    all_trades=all_trades,
                    episode_winrates=episode_winrates,
                    episode_epsilons=episode_epsilons,
                    eps_threshold=getattr(cfg, 'winrate_eps_threshold', 0.2),
                    eval_summary=None,
                    best_winrate=best_winrate,
                    best_episode_idx=best_episode_idx,
                    action_counts_total=action_counts_total,
                    buy_attempts_total=buy_attempts_total,
                    buy_rejected_vol_total=buy_rejected_vol_total,
                    buy_rejected_roi_total=buy_rejected_roi_total,
                    episodes_with_trade_count=episodes_with_trade_count,
                    total_steps_processed=total_steps_processed,
                    episode_length=episode_length,
                    seed=seed,
                    dqn_solver=dqn_solver,
                    env=env,
                    is_multi_crypto=is_multi_crypto,
                    parent_run_id=parent_run_id,
                    root_id=root_id,
                    training_start_time=training_start_time,
                    current_total_training_time=time.time() - training_start_time,
                    direction=(direction or 'long'),
                    winrate_trend={
                        'snapshot_every': int(_wr_snapshot_every),
                        'window_size': int(_wr_window_n),
                        'ema_alpha': float(_wr_ema_alpha),
                        'snapshots': winrate_snapshots,
                    },
                )
                # Сохраняем последний снапшот last_model.pth (отдельно)
                try:
                    import shutil as _sh
                    if os.path.exists(cfg.model_path):
                        _sh.copy2(cfg.model_path, os.path.join(run_dir, 'last_model.pth'))
                except Exception:
                    pass
            
            # Улучшенный Early stopping с множественными критериями
            if episode >= min_episodes_before_stopping:
                # Дополнительная защита от слишком раннего stopping
                if episode < episodes // 2:  # Не останавливаемся в первой половине обучения (было 1/3)
                    patience_counter = min(patience_counter, patience_limit // 4)  # Ограничиваем patience сильнее (было 1/3)
                elif episode < episodes * 3 // 4:  # Дополнительная защита до 3/4 (было 1/2)
                    patience_counter = min(patience_counter, patience_limit // 2)
                
                # Основной критерий - patience
                if patience_counter >= patience_limit:
                    logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (patience limit reached)")
                    print(f"  ⚠️ Early stopping: достигнут лимит patience ({patience_limit})")
                    print(f"  🔍 Отладка: patience_counter={patience_counter}, patience_limit={patience_limit}")
                    # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                    actual_episodes = episode
                    break
                
                # Дополнительный критерий - анализ трендов (УЛУЧШЕНО)
                if len(episode_winrates) >= 400 and episode >= episodes * 4 // 5:  # Увеличил требования: 400 эпизодов и последняя 1/5
                    recent_winrate = np.mean(episode_winrates[-80:])   # Увеличил окно анализа с 50 до 80
                    mid_winrate = np.mean(episode_winrates[-160:-80])  # Увеличил окно с 100:-50 до 160:-80
                    early_winrate = np.mean(episode_winrates[-240:-160])  # Увеличил окно с 150:-100 до 240:-160
                    
                    # Если winrate стабильно падает на протяжении 240 эпизодов (более строгое условие)
                    if (recent_winrate < mid_winrate < early_winrate and 
                        mid_winrate - recent_winrate > trend_threshold * 2.5 and  # Увеличил порог с 2.0 до 2.5
                        early_winrate - mid_winrate > trend_threshold * 2.5):
                        
                        logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (declining trend)")
                        # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                        actual_episodes = episode
                        break
                
                                # Долгосрочный критерий - если модель стабильна, даем больше времени
                if patience_counter >= long_term_patience:
                    logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (long-term patience)")
                    # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                    actual_episodes = episode
                    break

        # Финальная статистика
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print("\n" + "="*60)
        print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ для {training_name}")
        print("="*60)
        
        print(f"⏱️ ВРЕМЯ ОБУЧЕНИЯ:")
        print(f"  • Общее время: {total_training_time:.2f} секунд ({total_training_time/60:.1f} минут)")
        # Исправление ZeroDivisionError: Проверяем, что episode > 0 перед делением
        if episode > 0:
            print(f"  • Время на эпизод: {total_training_time/episode:.2f} секунд")
            print(f"  • Эпизодов в минуту: {episode/(total_training_time/60):.1f}")
        else:
            print(f"  • Время на эпизод: Недоступно (эпизод = 0)")
            print(f"  • Эпизодов в минуту: Недоступно (эпизод = 0)")
        
        stats_all = dqn_solver.print_trade_stats(all_trades)
        
        # Дополнительная статистика
        if all_trades:
            total_profit = sum([t.get('roi', 0) for t in all_trades if t.get('roi', 0) > 0])
            total_loss = abs(sum([t.get('roi', 0) for t in all_trades if t.get('roi', 0) < 0]))
            avg_duration = np.mean([t.get('duration', 0) for t in all_trades])
            
            print(f"\n💰 Общая статистика:")
            print(f"  • Общая прибыль: {total_profit:.4f}")
            print(f"  • Общий убыток: {total_loss:.4f}")
            print(f"  • Средняя длительность сделки: {avg_duration:.1f} минут")
            print(f"  • Планируемые эпизоды: {episodes}")
            print(f"  • Реальные эпизоды: {episode}")
            if episode < episodes:
                print(f"  • Early Stopping: Сработал на {episode} эпизоде")
            else:
                print(f"  • Early Stopping: Не сработал")
            print(f"  • Средний winrate: {np.mean(episode_winrates):.3f}")
        else:
            print(f"\n⚠️ Нет сделок за все {episodes} эпизодов!")

        # Печать причин продаж (если есть)
        try:
            sell_types_total = get_env_attr_safe(env, 'cumulative_sell_types', {})
            if isinstance(sell_types_total, dict) and sell_types_total:
                print("\n🧾 Причины продаж (агрегат):")
                for k, v in sell_types_total.items():
                    print(f"  • {k}: {int(v)}")
        except Exception:
            pass
        
        if hasattr(cfg, 'use_wandb') and cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        # Финальное сохранение модели и replay buffer
        print("\n💾 Финальное сохранение модели и replay buffer")
        # Единый препроцессинг: сохраняем статистики нормализации env в чекпойнт
        norm_stats = None
        try:
            export_stats = get_env_attr_safe(env, 'export_normalization_stats')
            if callable(export_stats):
                norm_stats = export_stats()
        except Exception:
            norm_stats = None
        dqn_solver.save(normalization_stats=norm_stats)
        # После сохранения весов — финализируем encoder_manifest.json, encoder_index.json и current.json (атомарно)
        try:
            enc_path = getattr(cfg, 'encoder_path', None)
            enc_sha = _sha256_of_file(enc_path) if enc_path and os.path.exists(enc_path) else None
            enc_size = os.path.getsize(enc_path) if enc_path and os.path.exists(enc_path) else None
            # Попытка оценить число параметров энкодера (если модель предоставляет метод)
            try:
                num_params = None
                if hasattr(dqn_solver.model, 'get_feature_extractor'):
                    _fe = dqn_solver.model.get_feature_extractor()
                    if _fe is not None:
                        num_params = sum(p.numel() for p in _fe.parameters())
            except Exception:
                num_params = None

            # Подготовим сводки обучения и датасета для манифеста энкодера
            try:
                avg_winrate_val = float(np.mean(episode_winrates)) if episode_winrates else None
            except Exception:
                avg_winrate_val = None
            try:
                frames_info = {}
                if isinstance(dfs, dict):
                    for _k in ('df_5min','df_15min','df_1h'):
                        _dfx = dfs.get(_k)
                        frames_info[_k] = {
                            'rows': int(len(_dfx)) if _dfx is not None else None
                        }
                    # Диапазон дат для 5m
                    _df5 = dfs.get('df_5min')
                    if _df5 is not None and len(_df5) > 0:
                        try:
                            _start_ts = _df5.index.min()
                            _end_ts = _df5.index.max()
                            frames_info['df_5min']['date_range'] = {
                                'start': str(_start_ts),
                                'end': str(_end_ts),
                            }
                        except Exception:
                            pass
            except Exception:
                frames_info = {}

            # --- Итоговые индикаторы для freeze‑решения ---
            training_indicators = None
            try:
                # 1) Q‑loss stability (EMA/STD/наклон на последней части)
                q_loss_metrics = None
                if q_loss_history:
                    arr = np.array(q_loss_history, dtype=np.float32)
                    # EMA (alpha=0.1)
                    ema = float(arr[0])
                    alpha = 0.1
                    for v in arr[1:]:
                        ema = alpha * float(v) + (1.0 - alpha) * ema
                    # Последние 20% точек для STD и slope
                    w = max(10, int(len(arr) * 0.2))
                    tail = arr[-w:]
                    std_last = float(np.std(tail)) if tail.size > 1 else 0.0
                    # Линейный тренд (slope) по последнему окну
                    x = np.arange(tail.size, dtype=np.float32)
                    slope = float(np.polyfit(x, tail, deg=1)[0]) if tail.size >= 2 else 0.0
                    thr_std = float(getattr(cfg, 'q_loss_std_threshold', 2e-3))
                    thr_slope = float(getattr(cfg, 'q_loss_slope_threshold', 1e-4))
                    loss_stable = (abs(slope) < thr_slope) and (std_last < thr_std)
                    q_loss_metrics = {
                        'ema': ema,
                        'std_last_20pct': std_last,
                        'slope_last_20pct': slope,
                        'thresholds': { 'std': thr_std, 'abs_slope': thr_slope },
                        'stable': bool(loss_stable),
                        'count': int(len(arr)),
                    }

                # 2) Feature drift (косинусная схожесть эмбеддингов)
                drift_metrics = None
                if drift_cosine_history:
                    mean_cos_last = float(drift_cosine_history[-1])
                    # Считаем момент стабилизации: первый индекс, с которого все значения >= 0.99 подряд в последних 3 снэпшотах
                    stable_since = None
                    if len(drift_cosine_history) >= 3:
                        for i in range(len(drift_cosine_history) - 3, -1, -1):
                            if min(drift_cosine_history[i: i+3]) >= 0.99:
                                stable_since = int(drift_snapshot_episodes[i]) if i < len(drift_snapshot_episodes) else None
                                break
                    drift_metrics = {
                        'mean_cos_sim_last': mean_cos_last,
                        'stable_since_episode': stable_since,
                        'probe_size': int(len(probe_states)),
                        'interval_episodes': probe_interval_episodes,
                        'threshold': 0.99,
                    }

                # 3) Q‑value stability (коэфф. вариации и макс. осцилляция в последних 10%)
                q_value_metrics = None
                if q_values_history:
                    H = len(q_values_history)
                    m = max(2, int(np.ceil(H * 0.1)))
                    window = q_values_history[-m:]
                    # Матрица [m, N]
                    mat = np.stack(window, axis=0)
                    mean_per_state = np.mean(mat, axis=0)
                    std_per_state = np.std(mat, axis=0)
                    # Коэффициент вариации (стабильность)
                    cv = std_per_state / (np.abs(mean_per_state) + 1e-8)
                    mean_cv_last = float(np.mean(cv))
                    # Макс. относительная осцилляция
                    range_per_state = (np.max(mat, axis=0) - np.min(mat, axis=0)) / (np.abs(mean_per_state) + 1e-8)
                    max_osc_last = float(np.max(range_per_state))
                    thr_cv = float(getattr(cfg, 'q_value_cv_threshold', 0.10))
                    thr_osc = float(getattr(cfg, 'q_value_osc_threshold', 0.10))
                    q_value_metrics = {
                        'mean_cv_last': mean_cv_last,
                        'max_oscillation_last_10pct': max_osc_last,
                        'thresholds': { 'cv': thr_cv, 'osc': thr_osc },
                    }

                # Итоговая рекомендация freeze
                freeze_rec = None
                try:
                    cond_loss = (q_loss_metrics or {}).get('stable', False)
                    cond_drift = (drift_metrics or {}).get('mean_cos_sim_last', 0.0) >= 0.99
                    cond_q = ((q_value_metrics or {}).get('max_oscillation_last_10pct', 1.0) <= (q_value_metrics or {}).get('thresholds', {}).get('osc', 0.10))
                    should_freeze = bool(cond_loss and cond_drift and cond_q)
                    reasons = []
                    if cond_loss: reasons.append('loss_stable')
                    if cond_drift: reasons.append('drift_stopped')
                    if cond_q: reasons.append('q_values_stable')
                    freeze_rec = {
                        'should_freeze': should_freeze,
                        'reasons': reasons,
                        'evaluated_at_episode': int(actual_episodes) if isinstance(actual_episodes, (int, float)) else None,
                    }
                except Exception:
                    freeze_rec = None

                training_indicators = {
                    'q_loss_stability': q_loss_metrics,
                    'feature_drift': drift_metrics,
                    'q_value_stability': q_value_metrics,
                    'freeze_recommendation': freeze_rec,
                }
            except Exception:
                training_indicators = None

            # Суммируем опыт с прошлого манифеста базового энкодера (если дообучаем)
            prev_episodes_completed = None
            prev_time_sec = None
            prev_total_steps = None
            try:
                if base_encoder_path and os.path.exists(base_encoder_path):
                    _prev_manifest_path = os.path.join(os.path.dirname(base_encoder_path), 'encoder_manifest.json')
                    _prev = _safe_read_json(_prev_manifest_path) or {}
                    _tr = _prev.get('training') or {}
                    # Предпочитаем cumulative, иначе обычные значения
                    prev_episodes_completed = _tr.get('cumulative_episodes_completed')
                    if prev_episodes_completed is None:
                        prev_episodes_completed = _tr.get('episodes_completed')
                    prev_time_sec = _tr.get('cumulative_time_sec')
                    if prev_time_sec is None:
                        prev_time_sec = _tr.get('time_sec')
                    prev_total_steps = _tr.get('cumulative_total_steps')
                    if prev_total_steps is None:
                        prev_total_steps = _tr.get('total_steps')
            except Exception:
                prev_episodes_completed = None
                prev_time_sec = None
                prev_total_steps = None

            manifest_data = {
                'manifest_version': 1,
                'run_id': this_run_id,
                'training_run_dir': run_dir,
                'freeze_encoder': bool(getattr(cfg, 'freeze_encoder', False)),
                'encoder_type': getattr(cfg, 'encoder_type', None),
                'version': int(getattr(cfg, 'encoder_version', 0) or 0),
                'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'symbol': symbol_dir_name,
                'encoder': {
                    'path_abs': os.path.abspath(enc_path) if enc_path else None,
                    'path_rel': enc_path,
                    'sha256': enc_sha,
                    'size_bytes': enc_size,
                    'num_parameters': num_params,
                },
                'model': {
                    'path_abs': os.path.abspath(cfg.model_path) if getattr(cfg, 'model_path', None) else None,
                    'path_rel': getattr(cfg, 'model_path', None),
                },
                'parent_run_id': parent_run_id,
                'root_id': this_root_id,
                'base_encoder': {
                    'selected_id': selected_encoder_id,
                    'path': base_encoder_path,
                    'root': base_encoder_root,
                },
                'seed': int(seed) if isinstance(seed, int) else None,
                'training': {
                    'episodes_planned': int(episodes) if isinstance(episodes, (int, float)) else None,
                    'episodes_completed': int(actual_episodes) if isinstance(actual_episodes, (int, float)) else None,
                    'episode_length': int(episode_length) if isinstance(episode_length, (int, float)) else None,
                    'total_steps': int(total_steps_processed) if isinstance(total_steps_processed, (int, float)) else None,
                    'time_sec': float(total_training_time) if isinstance(total_training_time, (int, float)) else None,
                    'start_utc': datetime.utcfromtimestamp(training_start_time).strftime('%Y-%m-%dT%H:%M:%SZ') if isinstance(training_start_time, (int, float)) else None,
                    'end_utc': datetime.utcfromtimestamp(training_start_time + total_training_time).strftime('%Y-%m-%dT%H:%M:%SZ') if isinstance(training_start_time, (int, float)) and isinstance(total_training_time, (int, float)) else None,
                    'prev_episodes_completed': int(prev_episodes_completed) if isinstance(prev_episodes_completed, (int, float)) else None,
                    'prev_time_sec': float(prev_time_sec) if isinstance(prev_time_sec, (int, float)) else None,
                    'prev_total_steps': int(prev_total_steps) if isinstance(prev_total_steps, (int, float)) else None,
                    'cumulative_episodes_completed': (
                        int(prev_episodes_completed) + int(actual_episodes)
                    ) if isinstance(prev_episodes_completed, (int, float)) and isinstance(actual_episodes, (int, float)) else int(actual_episodes) if isinstance(actual_episodes, (int, float)) else None,
                    'cumulative_time_sec': (
                        float(prev_time_sec) + float(total_training_time)
                    ) if isinstance(prev_time_sec, (int, float)) and isinstance(total_training_time, (int, float)) else float(total_training_time) if isinstance(total_training_time, (int, float)) else None,
                    'cumulative_total_steps': (
                        int(prev_total_steps) + int(total_steps_processed)
                    ) if isinstance(prev_total_steps, (int, float)) and isinstance(total_steps_processed, (int, float)) else int(total_steps_processed) if isinstance(total_steps_processed, (int, float)) else None,
                },
                'performance': {
                    'avg_winrate': avg_winrate_val,
                    'best_winrate': float(best_winrate) if isinstance(best_winrate, (int, float)) else None,
                    'best_episode': int(best_episode_idx) if isinstance(best_episode_idx, (int, float)) else None,
                },
                'data': {
                    'frames': frames_info,
                },
                'direction': (direction or 'long'),
                'trained_as': (direction or 'long'),
                # Места для опциональных данных об обучении/датах/метриках можно заполнить позже
            }
            # Вкладываем training_indicators при наличии
            if training_indicators is not None:
                manifest_data['training_indicators'] = training_indicators
            # Записываем encoder_manifest.json ТОЛЬКО если создавали новую версию
            if create_new_encoder:
                enc_manifest_path = os.path.join(os.path.dirname(enc_path) if enc_path else run_dir, 'encoder_manifest.json')
                _atomic_write_json(enc_manifest_path, manifest_data)

                # Обновляем encoder_index.json списком версий
                encoder_base = os.path.join("result", "dqn", symbol_dir_name, "encoder")
                encoder_type = getattr(cfg, 'encoder_type', 'unfrozen') or 'unfrozen'
                encoder_type_dir = os.path.join(encoder_base, encoder_type)
                index_path = os.path.join(encoder_type_dir, 'encoder_index.json')
                index_data = _safe_read_json(index_path) or []
                index_entry = {
                    'version': int(getattr(cfg, 'encoder_version', 0) or 0),
                    'run_id': this_run_id,
                    'date': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'encoder_type': encoder_type,
                    'encoder_path': enc_path,
                    'model_path': getattr(cfg, 'model_path', None),
                    'symbol': symbol_dir_name,
                    'sha256': enc_sha,
                    'size_bytes': enc_size,
                    'direction': (direction or 'long'),
                }
                try:
                    index_entry['episodes_completed'] = int(actual_episodes) if isinstance(actual_episodes, (int, float)) else None
                    # Дополнительно фиксируем cumulative (если есть предыдущий опыт)
                    cum_eps = (
                        int(prev_episodes_completed) + int(actual_episodes)
                    ) if isinstance(prev_episodes_completed, (int, float)) and isinstance(actual_episodes, (int, float)) else int(actual_episodes) if isinstance(actual_episodes, (int, float)) else None
                    index_entry['cumulative_episodes_completed'] = cum_eps
                except Exception:
                    pass
                index_data.append(index_entry)
                _atomic_write_json(index_path, index_data)

                # Указатель на текущую версию
                current_path = os.path.join(encoder_type_dir, 'current.json')
                _atomic_write_json(current_path, {'version': f"v{int(getattr(cfg, 'encoder_version', 0) or 0)}", 'sha256': enc_sha})
        except Exception:
            pass
        # Гарантируем сохранение энкодера (fallback): если файл отсутствует — сохраним модель (включая encoder_only)
        try:
            enc_check = getattr(cfg, 'encoder_path', None)
            if enc_check and not os.path.exists(enc_check):
                dqn_solver.save_model()
        except Exception:
            pass
        # Финальный last_model снапшот
        try:
            import shutil as _sh
            if os.path.exists(cfg.model_path):
                _sh.copy2(cfg.model_path, os.path.join(run_dir, 'last_model.pth'))
            # Дополнительно сохраняем удобную копию энкодера в run_dir (если существует)
            try:
                enc_src = getattr(cfg, 'encoder_path', None)
                if enc_src and os.path.exists(enc_src):
                    _sh.copy2(enc_src, os.path.join(run_dir, 'last_encoder.pth'))
            except Exception:
                pass
        except Exception:
            pass
        
        # Greedy-оценка политики (ε=0) — отдельные eval-эпизоды без обучения
        eval_summary = None
        try:
            eval_episodes_count = int(getattr(cfg, 'eval_episodes', 5))
        except Exception:
            eval_episodes_count = 5
        try:
            if eval_episodes_count and eval_episodes_count > 0:
                eval_episode_winrates = []
                saved_eps = dqn_solver.epsilon
                try:
                    for _ in range(eval_episodes_count):
                        state_eval = env.reset()
                        # Отсечём текущее количество кумулятивных сделок ПОСЛЕ reset(),
                        # т.к. reset() может очищать trades/all_trades и иначе winrate может залипать в 0.0
                        try:
                            _before_count = len(get_env_attr_safe(env, 'all_trades') or [])
                        except Exception:
                            _before_count = None
                        # Убедимся, что состояние — numpy
                        if isinstance(state_eval, (list, tuple)):
                            state_eval = np.array(state_eval, dtype=np.float32)
                        elif not isinstance(state_eval, np.ndarray):
                            state_eval = np.array(state_eval, dtype=np.float32)
                        dqn_solver.epsilon = 0.0
                        while True:
                            action_eval = dqn_solver.act(state_eval)
                            state_eval, _, terminal_eval, _ = env.step(action_eval)
                            if isinstance(state_eval, (list, tuple)):
                                state_eval = np.array(state_eval, dtype=np.float32)
                            elif not isinstance(state_eval, np.ndarray):
                                state_eval = np.array(state_eval, dtype=np.float32)
                            if terminal_eval:
                                break
                        # Определяем сделки этого eval-эпизода
                        trades_after = get_env_attr_safe(env, 'all_trades') or []
                        if _before_count is not None and isinstance(trades_after, list) and len(trades_after) >= _before_count:
                            ep_trades = trades_after[_before_count:]
                        else:
                            ep_trades = get_env_attr_safe(env, 'trades', []) or []
                        if ep_trades:
                            wins = sum(1 for t in ep_trades if t.get('roi', 0) > 0)
                            wr_ep = (wins / len(ep_trades)) if len(ep_trades) > 0 else 0.0
                            eval_episode_winrates.append(float(wr_ep))
                        else:
                            eval_episode_winrates.append(0.0)
                finally:
                    dqn_solver.epsilon = saved_eps
                try:
                    eval_wr = float(np.mean(eval_episode_winrates)) if eval_episode_winrates else None
                except Exception:
                    eval_wr = None
                eval_summary = {
                    'episodes': int(eval_episodes_count),
                    'winrate': eval_wr,
                    'episode_winrates': eval_episode_winrates,
                }
        except Exception:
            eval_summary = None

        # Сохраняем детальные результаты обучения
        _save_training_results(
            run_dir=run_dir,
            dfs=dfs, # Передаем dfs
            cfg=cfg,
            training_name=training_name,
            current_episode=actual_episodes, # Используем actual_episodes для финального сохранения
            total_episodes_planned=episodes,
            all_trades=all_trades,
            episode_winrates=episode_winrates,
            episode_epsilons=episode_epsilons,
            eps_threshold=getattr(cfg, 'winrate_eps_threshold', 0.2),
            eval_summary=eval_summary,
            best_winrate=best_winrate,
            best_episode_idx=best_episode_idx,
            action_counts_total=action_counts_total,
            buy_attempts_total=buy_attempts_total,
            buy_rejected_vol_total=buy_rejected_vol_total,
            buy_rejected_roi_total=buy_rejected_roi_total,
            episodes_with_trade_count=episodes_with_trade_count,
            total_steps_processed=total_steps_processed,
            episode_length=episode_length,
            seed=seed,
            dqn_solver=dqn_solver,
            env=env,
            is_multi_crypto=is_multi_crypto,
            parent_run_id=parent_run_id,
            root_id=root_id,
            training_start_time=training_start_time,
            current_total_training_time=total_training_time, # Используем final total_training_time
            direction=(direction or 'long'),
            winrate_trend={
                'snapshot_every': int(_wr_snapshot_every),
                'window_size': int(_wr_window_n),
                'ema_alpha': float(_wr_ema_alpha),
                'snapshots': winrate_snapshots,
            },
        )

        # Добавим краткую информацию о выборе энкодера в отдельный JSON в run_dir
        try:
            selection = {
                'selected_id': selected_encoder_id,
                'train_encoder': bool(train_encoder_flag),
                'mode': 'unfrozen' if train_encoder_flag else 'frozen',
                'base_encoder_path': base_encoder_path,
                'created_new_version': bool(create_new_encoder),
                'new_version': f"v{created_encoder_version}" if create_new_encoder else None,
                'final_encoder_path': getattr(cfg, 'encoder_path', None),
            }
            _atomic_write_json(os.path.join(run_dir, 'encoder_selection.json'), selection)
        except Exception:
            pass

        # Вывод статистики продлений (если обёртка активна)
        try:
            if hasattr(env, 'episode_extensions_total'):
                print(f"  • Продления эпизодов: {int(env.episode_extensions_total)} раз (+{int(env.episode_extension_steps_total)} шагов)")
        except Exception:
            pass

        # Анализ трендов
        if len(episode_winrates) > 10:
            recent_winrate = np.mean(episode_winrates[-10:])
            overall_winrate = np.mean(episode_winrates)
            print(f"📈 Winrate тренд: последние 10 эпизодов: {recent_winrate:.3f}, общий: {overall_winrate:.3f}")
            
            if recent_winrate > overall_winrate:
                print("✅ Модель улучшается!")
            else:
                print("⚠️ Модель может переобучаться")
        
        return "Обучение завершено"    
    finally:
        # Закрываем wandb
        if wandb_run is not None:
            wandb_run.finish()
