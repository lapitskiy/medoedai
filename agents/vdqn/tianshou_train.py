import os
import uuid
import time
import json as _json
import hashlib
import platform
from datetime import datetime
import random
from collections import deque
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import gymnasium as gym
import signal
import io

from tianshou.env import SubprocVectorEnv
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy
try:
    from tianshou.trainer import offpolicy_trainer
except Exception:
    try:
        from tianshou.trainer import OffpolicyTrainer as _OffpolicyTrainer
        def offpolicy_trainer(*args, **kwargs):
            return _OffpolicyTrainer(*args, **kwargs).run()
    except Exception:
        raise
try:
    from tianshou.utils.logger import CSVLogger  # может отсутствовать в старых версиях
except Exception:
    CSVLogger = None

from agents.vdqn.dqnn import DQNN
from envs.dqn_model.gym.crypto_trading_env_tainshou import CryptoTradingEnvOptimized as CryptoTradingEnvOptimized
from envs.dqn_model.gym.crypto_trading_env_multi import MultiCryptoTradingEnv
from envs.dqn_model.gym.gconfig import GymConfig
from agents.vdqn.hyperparameter.symbol_overrides import get_symbol_override
from utils.config_loader import get_config_value
from utils.adaptive_normalization import adaptive_normalizer
from threading import Thread
from gymnasium.wrappers import TimeLimit
from agents.vdqn.tianshou.env_wrappers import PositionAwareEpisodeWrapper


class MaskedDQNPolicy(DQNPolicy):
    """DQNPolicy with action_mask support (mask only, no market logic).

    Mask is expected in batch.info as scalar keys: mask_0/mask_1/mask_2 (0/1).
    """

    def _extract_mask(self, batch: Batch):
        try:
            info = getattr(batch, 'info', None)
            if info is None:
                return None
            m0 = info.get('mask_0', None)
            m1 = info.get('mask_1', None)
            m2 = info.get('mask_2', None)
            if m0 is None or m1 is None or m2 is None:
                return None
            # convert to tensor shape [B, 3]
            m = np.stack([np.asarray(m0), np.asarray(m1), np.asarray(m2)], axis=-1).astype(np.int64)
            return torch.as_tensor(m, device=self.model_device)
        except Exception:
            return None

    @property
    def model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device('cpu')

    def forward(self, batch: Batch, state=None, **kwargs):
        obs = batch.obs
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(np.array(obs, dtype=np.float32), device=self.model_device)
        else:
            obs = obs.to(self.model_device)
        out = self.model(obs)
        q = out[0] if isinstance(out, tuple) else out

        mask = self._extract_mask(batch)
        if mask is not None:
            try:
                q = q.clone()
                q[mask == 0] = -1e9
            except Exception:
                pass

        act = torch.argmax(q, dim=1)

        # epsilon-greedy exploration with mask-aware random choice
        eps = float(getattr(self, 'eps', 0.0) or 0.0)
        if eps > 0:
            rnd = torch.rand(act.shape[0], device=act.device)
            do_rand = rnd < eps
            if bool(do_rand.any()):
                for i in range(int(act.shape[0])):
                    if not bool(do_rand[i]):
                        continue
                    if mask is None:
                        act[i] = int(np.random.randint(0, int(q.shape[1])))
                    else:
                        allowed = torch.where(mask[i] > 0)[0]
                        if allowed.numel() > 0:
                            j = int(torch.randint(0, int(allowed.numel()), (1,), device=act.device).item())
                            act[i] = int(allowed[j].item())
        return Batch(logits=q, act=act, state=state)
from agents.vdqn.cfg.extension_config import SYMBOL_EXTENSION_CONFIG, DEFAULT_EXTENSION_CONFIG


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


def _sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _format_bytes(num_bytes: int) -> str:
    try:
        gb = num_bytes / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = num_bytes / (1024 ** 2)
        return f"{mb:.0f} MB"
    except Exception:
        return str(num_bytes)


def _log_resource_usage(tag: str = "train", extra: str = "") -> None:
    try:
        import psutil
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
        line = (
            f"[RES-{tag}] CPU {cpu_total_pct:.0f}% (proc {cpu_proc_pct:.0f}%), load {load_str}, "
            f"mem {mem_total_pct:.0f}% (proc {_format_bytes(mem_proc)})"
            + (f" | {extra}" if extra else "")
        )
        print(line, flush=True)
    except Exception:
        pass


# Обёртка для env: прокидывает трейды/метрики в info при завершении эпизода
class TradingEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        # Проксируем необходимые атрибуты/пространства
        self.action_space = getattr(env, 'action_space', None)
        self.observation_space = getattr(env, 'observation_space', None)
        self.observation_space_shape = getattr(env, 'observation_space_shape', None)
        self.symbol = getattr(env, 'symbol', None)
        self.cfg = getattr(env, 'cfg', None)
        # Счетчики эпизода для лаконичного логирования
        self._episode_idx = -1
        self._episode_steps = 0

        # Снимок кумулятивных причин продаж на границах эпизодов
        self.cumulative_sell_stats = {}
        self._episode_reward = 0.0

    def reset(self, *args, **kwargs):
        # Форсируем длину эпизода на входе
        target_len = kwargs.get('episode_length') if 'episode_length' in kwargs else None
        if target_len is None and args:
            try:
                target_len = args[0]
            except Exception:
                target_len = None
        if target_len is None:
            target_len = getattr(self.env, 'episode_length', None)

        # Синхронизируем cfg и env длину эпизода
        try:
            if target_len is not None:
                target_len = int(target_len)
                if hasattr(self.env, 'cfg'):
                    setattr(self.env.cfg, 'episode_length', target_len)
                if hasattr(self.env, 'episode_length'):
                    self.env.episode_length = target_len
        except Exception:
            pass

        result = self.env.reset(*args, **kwargs)
        # Новый эпизод
        try:
            self._episode_idx += 1
            self._episode_steps = 0
            self._episode_reward = 0.0
            try:
                cfg_len = getattr(self.env.cfg, 'episode_length', None) if hasattr(self.env, 'cfg') else None
                env_len = getattr(self.env, 'episode_length', None)
                print(f"➡️ Reset env: cfg_episode_length={cfg_len} env_episode_length={env_len}")
            except Exception:
                pass
        except Exception:
            pass
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            state_next, reward, terminated, truncated, info = result
        else:
            # Совместимость со старым API (obs, reward, done, info)
            state_next, reward, done, info = result
            terminated = bool(done)
            truncated = False

        self._episode_steps += 1
        self._episode_reward = float(self._episode_reward) + float(reward)

        # Масштабирование награды по cfg.reward_scale (как в legacy DQN)
        try:
            rs = 1.0
            if hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'reward_scale'):
                rs = float(getattr(self.env.cfg, 'reward_scale', 1.0))
            reward = float(reward) * rs
        except Exception:
            pass

        # Длину эпизода контролирует базовое окружение/TimeLimit

        # Промежуточные логи убраны ради скорости; оставляем только терминальные
        try:
            if terminated or truncated:
                # Собираем трейды за эпизод, если env их хранит
                trades = []
                for attr in ('all_trades', 'trades'):
                    if hasattr(self.env, attr) and getattr(self.env, attr):
                        try:
                            trades = list(getattr(self.env, attr))
                            break
                        except Exception:
                            trades = []
                # Клонируем info и добавим ключ только на boundary шагах
                if not isinstance(info, dict):
                    info = {}
                info = dict(info)
                if trades:
                    info['trades_episode'] = trades
                # Снимем снапшот cumulative_sell_types до reset() со стороны Collector
                try:
                    base_env = self.env
                    for _ in range(10):
                        if hasattr(base_env, 'env'):
                            base_env = getattr(base_env, 'env')
                        else:
                            break
                    st = getattr(base_env, 'cumulative_sell_types', None)
                    if isinstance(st, dict):
                        self.cumulative_sell_stats = dict(st)
                except Exception:
                    pass
                # Проксируем buy-* метрики, если доступны в env
                for k_src, k_dst in (
                    ('buy_attempts', 'buy_attempts'),
                    ('buy_rejected_vol', 'buy_rejected_vol'),
                    ('buy_rejected_roi', 'buy_rejected_roi'),
                ):
                    if hasattr(self.env, k_src):
                        try:
                            info[k_dst] = int(getattr(self.env, k_src) or 0)
                        except Exception:
                            pass
                # Краткая строка лога по эпизоду «как в dqn»
                try:
                    act_stats = getattr(self.env, 'action_counts', {}) if hasattr(self.env, 'action_counts') else {}
                    eps_val = getattr(self.env, 'epsilon', None)
                    trades_cnt = len(trades) if isinstance(trades, list) else (len(getattr(self.env, 'trades', []) or []) if hasattr(self.env, 'trades') else 0)
                    print(
                        f"🏁 Ep {self._episode_idx}: steps={self._episode_steps} reward={self._episode_reward:.4f} "
                        f"actions={act_stats} trades={trades_cnt}"
                        + (f" eps={eps_val:.3f}" if isinstance(eps_val, (int, float)) else "")
                    )
                except Exception:
                    pass
        except Exception:
            pass
        try:
            info = _sanitize_info_for_tianshou(info)
        except Exception:
            pass
        return state_next, reward, terminated, truncated, info

    # Проксируем возможные вспомогательные методы
    def __getattr__(self, item):
        return getattr(self.env, item)


def _export_norm_stats_safe(dfs: Dict, episode_length: Optional[int]) -> Optional[dict]:
    try:
        env = make_env_fn(dfs, episode_length)()
        stats = None
        if hasattr(env, 'export_normalization_stats'):
            stats = env.export_normalization_stats()
        # Попытка аккуратно закрыть, если есть close
        try:
            if hasattr(env, 'close'):
                env.close()
        except Exception:
            pass
        return stats
    except Exception:
        return None


def _try_load_legacy_replay(path: str, buf) -> bool:
    """Попытка загрузить старый формат replay.pkl в буфер Tianshou.
    Поддерживаются простые словари: {states, actions, rewards, next_states, dones}.
    Возвращает True если удалось что-то импортировать.
    """
    try:
        import pickle
        if not os.path.isfile(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Вариант 1: словарь массивов
        if isinstance(data, dict):
            states = data.get('states') or data.get('state')
            actions = data.get('actions') or data.get('action')
            rewards = data.get('rewards') or data.get('reward')
            next_states = data.get('next_states') or data.get('next_state')
            dones = data.get('dones') or data.get('done')
            if states is not None and actions is not None and rewards is not None and next_states is not None and dones is not None:
                n = min(len(states), len(actions), len(rewards), len(next_states), len(dones))
                imported = 0
                for i in range(n):
                    try:
                        s = states[i]
                        a = int(actions[i])
                        r = float(rewards[i])
                        sn = next_states[i]
                        d = bool(dones[i])
                        buf.add(s, a, r, d, sn, {})
                        imported += 1
                    except Exception:
                        continue
                if imported > 0:
                    print(f"♻️  Импортировано переходов из legacy replay: {imported}")
                    return True
        # Вариант 2: список переходов (s,a,r,sn,d)
        if isinstance(data, list):
            imported = 0
            for item in data:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 5:
                        s, a, r, sn, d = item[:5]
                        buf.add(s, int(a), float(r), bool(d), sn, {})
                        imported += 1
                except Exception:
                    continue
            if imported > 0:
                print(f"♻️  Импортировано переходов из legacy списка: {imported}")
                return True
    except Exception as e:
        print(f"⚠️ Конвертер legacy replay: {e}")
    return False


def _is_multi_crypto(dfs: Dict) -> bool:
    try:
        if dfs and isinstance(dfs, dict):
            k = list(dfs.keys())[0]
            return isinstance(k, str) and k.endswith('USDT')
    except Exception:
        return False
    return False


def make_env_fn(
    dfs: Dict,
    episode_length: Optional[int],
    gym_override: Optional[GymConfig] = None,
    *,
    enable_extension: bool = True,
) -> Callable[[], CryptoTradingEnvOptimized]:
    # Определим символ (если одиночные данные)
    symbol = None
    try:
        if isinstance(dfs, dict):
            symbol = dfs.get('symbol') or dfs.get('SYMBOL')
    except Exception:
        symbol = None

    # Получим overrides по символу
    override = get_symbol_override(symbol) if symbol else None
    indicators_config = override.get('indicators_config') if (override and 'indicators_config' in override) else None

    def _thunk():
        gym_cfg = gym_override or GymConfig()
        # Выровняем длину эпизода в cfg с запрошенным значением, чтобы done срабатывал по ожидаемой длине
        try:
            if episode_length is not None:
                gym_cfg.episode_length = int(episode_length)
        except Exception:
            pass
        if _is_multi_crypto(dfs):
            # Multi-crypto env не принимает episode_length напрямую
            env = MultiCryptoTradingEnv(dfs=dfs, cfg=gym_cfg)
        else:
            env = CryptoTradingEnvOptimized(
                dfs=dfs,
                cfg=gym_cfg,
                lookback_window=(override.get('gym_config', {}).get('lookback_window', gym_cfg.lookback_window) if override else gym_cfg.lookback_window),
                indicators_config=indicators_config,
                episode_length=episode_length or gym_cfg.episode_length,
            )
        # Сначала совместимость API: наш wrapper всегда возвращает 5 значений
        try:
            env = TradingEnvWrapper(env)
        except Exception:
            pass
        # Подключаем продление эпизода для train-окружений; для test — оставляем TimeLimit
        if enable_extension:
            try:
                sym_key = None
                try:
                    if _is_multi_crypto(dfs):
                        sym_key = 'MULTI'
                    else:
                        sym_key = (symbol or '').upper() if isinstance(symbol, str) else None
                except Exception:
                    sym_key = None
                cfg_ext = DEFAULT_EXTENSION_CONFIG
                if isinstance(sym_key, str) and sym_key in SYMBOL_EXTENSION_CONFIG:
                    cfg_ext = SYMBOL_EXTENSION_CONFIG[sym_key]
                env = PositionAwareEpisodeWrapper(
                    env,
                    max_extension=int(cfg_ext.get('max_extension', 20)),
                    extension_steps=int(cfg_ext.get('extension_steps', 100)),
                )
            except Exception:
                # Фолбэк: если не удалось подключить враппер — используем TimeLimit
                try:
                    max_steps = int(episode_length or gym_cfg.episode_length or 0)
                    if max_steps and max_steps > 0:
                        env = TimeLimit(env, max_episode_steps=max_steps)
                except Exception:
                    pass
        else:
            # Для тестовых окружений всегда используем TimeLimit, чтобы гарантировать завершение
            try:
                max_steps = int(episode_length or gym_cfg.episode_length or 0)
                if max_steps and max_steps > 0:
                    env = TimeLimit(env, max_episode_steps=max_steps)
            except Exception:
                pass
        # Применим risk-management overrides в env
        try:
            if override and 'risk_management' in override:
                rm = override['risk_management']
                for field_name, env_attr in [
                    ('STOP_LOSS_PCT', 'STOP_LOSS_PCT'),
                    ('TAKE_PROFIT_PCT', 'TAKE_PROFIT_PCT'),
                    ('min_hold_steps', 'min_hold_steps'),
                    ('volume_threshold', 'volume_threshold'),
                ]:
                    if field_name in rm:
                        setattr(env, env_attr, rm[field_name])
        except Exception:
            pass
        return env

    return _thunk


def train_tianshou_dqn(
    dfs: Dict,
    episodes: int = 2000,
    n_envs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    n_step: int = 1,
    target_update_freq: int = 500,
    memory_size: int = 500_000,
    episode_length: Optional[int] = None,
    run_id: Optional[str] = None,
    symbol_hint: Optional[str] = None,
    # Доп. параметры для совместимости
    parent_run_id: Optional[str] = None,
    root_id: Optional[str] = None,
    load_model_path: Optional[str] = None,
    load_buffer_path: Optional[str] = None,
    save_frequency: int = 50,
    buffer_save_frequency: Optional[int] = None,
    save_replay_on_improvement: bool = True,
    seed: Optional[int] = None,
) -> str:
    # Жёстко ограничим потоки BLAS/Torch, чтобы не было оверсабскрипшна при n_envs > 1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Seed / детерминизм (насколько возможно)
    # Автогенерация seed, если не задан
    if not isinstance(seed, int):
        try:
            seed = int.from_bytes(os.urandom(4), 'little')
        except Exception:
            try:
                seed = int(time.time()) % (2**31)
            except Exception:
                seed = 42
        try:
            print(f"🎲 Используем seed: {seed}")
        except Exception:
            pass
    try:
        if isinstance(seed, int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Время старта обучения
    training_start_time = time.time()

    # Включаем подробный дамп исключений и глобальные хуки
    try:
        import faulthandler, sys, traceback, threading as _threading
        try:
            faulthandler.enable()
            # Регистрация сигналов для дампа трейсбеков
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGSEGV, signal.SIGABRT):
                try:
                    faulthandler.register(sig, chain=True)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            def _global_excepthook(exc_type, exc, tb):
                try:
                    print("UNCAUGHT:", "".join(traceback.format_exception(exc_type, exc, tb)), flush=True)
                except Exception:
                    pass
            sys.excepthook = _global_excepthook
        except Exception:
            pass
        try:
            # Для исключений в потоках Python>=3.8
            def _thread_excepthook(args):
                try:
                    print("THREAD EXC:", "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)), flush=True)
                except Exception:
                    pass
            if hasattr(_threading, 'excepthook'):
                _threading.excepthook = _thread_excepthook
        except Exception:
            pass
    except Exception:
        pass

    single_env = make_env_fn(dfs, episode_length)()
    obs_dim = getattr(single_env, 'observation_space_shape', None)
    if obs_dim is None and hasattr(single_env, 'observation_space') and hasattr(single_env.observation_space, 'shape'):
        obs_dim = int(single_env.observation_space.shape[0])
    if obs_dim is None:
        raise RuntimeError("Не удалось определить размер наблюдения (obs_dim)")

    act_dim = single_env.action_space.n
    action_space_gym = getattr(single_env, 'action_space', None)
    if _is_multi_crypto(dfs):
        symbol = 'МУЛЬТИВАЛЮТА'
    else:
        symbol = getattr(single_env, 'symbol', symbol_hint or 'UNKNOWN')
    symbol_dir = _symbol_code(symbol).upper()
    # Прикинем бюджет памяти для буфера переходов: ограничим по TS_REPLAY_MEM_MB
    try:
        replay_mem_mb_cfg = int(str(get_config_value('TS_REPLAY_MEM_MB', '1024')))
    except Exception:
        replay_mem_mb_cfg = 1024
    try:
        # Грубая оценка: два наблюдения по obs_dim float32 + запас 1 KB на метаданные
        bytes_per_sample = (int(obs_dim) * 4) * 2 + 1024
        max_samples_by_budget = max(10_000, (replay_mem_mb_cfg * 1024 * 1024) // max(1, bytes_per_sample))
        print(f"🧮 Replay budget: obs_dim={obs_dim} ~{bytes_per_sample/1024:.1f} KB/transition, budget={replay_mem_mb_cfg} MB → max_samples≈{max_samples_by_budget}")
    except Exception:
        max_samples_by_budget = None
    # Кэшируем статистики нормализации (если поддерживается)
    norm_stats_cached = None
    try:
        if hasattr(single_env, 'export_normalization_stats'):
            norm_stats_cached = single_env.export_normalization_stats()
    except Exception:
        norm_stats_cached = None
    min_valid_start_step = getattr(single_env, 'min_valid_start_step', None)
    start_step_snapshot = getattr(single_env, 'start_step', None)
    actual_episode_length = getattr(single_env, 'episode_length', episode_length)
    if actual_episode_length is None:
        actual_episode_length = episode_length
    # Снимок ключевых атрибутов single_env до удаления, чтобы использовать позже
    try:
        single_env_snapshot = {
            'symbol': getattr(single_env, 'symbol', None),
            'lookback_window': getattr(single_env, 'lookback_window', None),
            'indicators_config': getattr(single_env, 'indicators_config', None),
            'cfg_reward_scale': (getattr(single_env.cfg, 'reward_scale', 1.0) if hasattr(single_env, 'cfg') else 1.0),
            'episode_length': getattr(single_env, 'episode_length', None),
            'position_sizing': {
                'base_position_fraction': getattr(single_env, 'base_position_fraction', None),
                'position_fraction': getattr(single_env, 'position_fraction', None),
                'position_confidence_threshold': getattr(single_env, 'position_confidence_threshold', None),
            },
            'risk_management': {
                'STOP_LOSS_PCT': getattr(single_env, 'STOP_LOSS_PCT', None),
                'TAKE_PROFIT_PCT': getattr(single_env, 'TAKE_PROFIT_PCT', None),
                'min_hold_steps': getattr(single_env, 'min_hold_steps', None),
                'volume_threshold': getattr(single_env, 'volume_threshold', None),
            },
            'observation_space_shape': getattr(single_env, 'observation_space_shape', None),
            'step_minutes': (getattr(single_env.cfg, 'step_minutes', 5) if hasattr(single_env, 'cfg') else 5),
        }
    except Exception:
        single_env_snapshot = None
    del single_env

    this_run_id = run_id or str(uuid.uuid4())[:4].lower()
    run_dir = Path("result") / "dqn" / symbol_dir / "runs" / this_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pth"
    replay_path = run_dir / "replay.pkl"

    # Флаги окружения/отладки читаем ДО manifest.json
    force_dummy = False
    force_single = False
    debug_exploration = False
    debug_run = False
    try:
        v = str(get_config_value('TS_FORCE_DUMMY', '0'))
        force_dummy = v.lower() in ('1', 'true', 'yes', 'y')
    except Exception:
        force_dummy = False
    try:
        v = str(get_config_value('TS_FORCE_SINGLE', '0'))
        force_single = v.lower() in ('1', 'true', 'yes', 'y')
    except Exception:
        force_single = False
    try:
        v = str(get_config_value('TS_DEBUG_EXPLORATION', '0'))
        debug_exploration = v.lower() in ('1', 'true', 'yes', 'y')
    except Exception:
        debug_exploration = False
    try:
        v = str(get_config_value('TS_DEBUG_RUN', '0'))
        debug_run = v.lower() in ('1', 'true', 'yes', 'y')
    except Exception:
        debug_run = False

    # Initial manifest.json (безусловно, чтобы он существовал даже при очень коротких запусках)
    try:
        initial_manifest = {
            'run_id': this_run_id,
            'parent_run_id': parent_run_id,
            'root_id': root_id or this_run_id,
            'symbol': symbol_dir,
            'seed': (int(seed) if isinstance(seed, int) else None),
            'episodes_start': 0,
            'episodes_end': 0,
            'episodes_added': 0,
            'episodes_last': 0,
            'episodes_best': None,
            'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'debug': bool(debug_run or debug_exploration),
            'debug_flags': [
                f for f, on in (
                    ('TS_DEBUG_RUN', debug_run),
                    ('TS_DEBUG_EXPLORATION', debug_exploration),
                    ('TS_FORCE_SINGLE', force_single),
                    ('TS_FORCE_DUMMY', force_dummy),
                ) if on
            ],
            'artifacts': {
                'model': 'model.pth',
                'replay': None,
                'result': 'train_result.pkl',
                'best_model': 'best_model.pth' if (run_dir / 'best_model.pth').exists() else None,
                'last_model': 'last_model.pth' if (run_dir / 'last_model.pth').exists() else None,
                'encoder_only': 'encoder_only.pth' if (run_dir / 'encoder_only.pth').exists() else None,
                'all_trades': 'all_trades.json',
            },
            'best_metrics': {
                'winrate': None,
                'reward': None,
            }
        }
        print(f"💡[Init] Сохраняем manifest.json в {run_dir / 'manifest.json'}")
        with open(run_dir / 'manifest.json', 'w', encoding='utf-8') as mf:
            _json.dump(initial_manifest, mf, ensure_ascii=False, indent=2)
            try:
                mf.flush(); os.fsync(mf.fileno())
            except Exception:
                pass
    except Exception as me_init:
        print(f"⚠️ Не удалось сохранить initial manifest.json: {me_init}")
        import traceback; traceback.print_exc()

    # Tee все print в файл run_dir/train.log
    _orig_stdout = None
    _orig_stderr = None
    _log_fp = None
    try:
        _log_fp = open(run_dir / 'train.log', 'a', encoding='utf-8')
        class _TeeIO(io.TextIOBase):
            def __init__(self, a, b):
                self.a = a; self.b = b
            def write(self, s):
                try:
                    self.a.write(s); self.a.flush()
                except Exception:
                    pass
                try:
                    self.b.write(s); self.b.flush()
                except Exception:
                    pass
                return len(s)
            def flush(self):
                try:
                    self.a.flush()
                except Exception:
                    pass
                try:
                    self.b.flush()
                except Exception:
                    pass
        import sys as _sys
        _orig_stdout, _orig_stderr = _sys.stdout, _sys.stderr
        _sys.stdout = _TeeIO(_sys.stdout, _log_fp)
        _sys.stderr = _TeeIO(_sys.stderr, _log_fp)
    except Exception:
        pass

    # Env уже обёрнут в TradingEnvWrapper внутри make_env_fn
    def wrapped_env_fn_train():
        return make_env_fn(dfs, episode_length, enable_extension=True)()

    def wrapped_env_fn_test():
        return make_env_fn(dfs, episode_length, enable_extension=False)()
    # Флаги окружения уже прочитаны выше (force_dummy/force_single/debug_*)

    # Пытаемся создать subprocess-векторы; при ошибке или флаге откатываемся к DummyVectorEnv/одиночным env
    # Префлайт: проверим, что одиночное окружение способно reset/step (ранний фейл, если зависание)
    try:
        _pre_env = wrapped_env_fn_train()
        try:
            _obs, _info = _pre_env.reset()
            print(f"🧪 preflight: reset ok, obs type={type(_obs)}")
            _a = _pre_env.action_space.sample()
            _res = _pre_env.step(_a)
            print("🧪 preflight: step ok")
        except Exception as _pe:
            print(f"❌ preflight failed: {_pe}")
            raise
        finally:
            try:
                if hasattr(_pre_env, 'close'):
                    _pre_env.close()
            except Exception:
                pass
    except Exception as _fatal_pe:
        print("❌ Критическая ошибка инициализации окружения (preflight)")
        raise

    try:
        if force_single:
            train_envs = [wrapped_env_fn_train()]
            test_envs = [wrapped_env_fn_test()]
            print("ℹ️ Форсирован одиночный Env (TS_FORCE_SINGLE)")
        elif not force_dummy:
            train_envs = SubprocVectorEnv([wrapped_env_fn_train for _ in range(n_envs)])
            test_envs = SubprocVectorEnv([wrapped_env_fn_test for _ in range(max(1, n_envs // 2))])
        else:
            from tianshou.env import DummyVectorEnv
            train_envs = DummyVectorEnv([wrapped_env_fn_train for _ in range(max(1, n_envs))])
            test_envs = DummyVectorEnv([wrapped_env_fn_test for _ in range(max(1, n_envs // 2))])
            print("ℹ️ Форсирован DummyVectorEnv (TS_FORCE_DUMMY)")
    except Exception as e:
        print(f"⚠️ SubprocVectorEnv не удалось создать, откатываюсь к DummyVectorEnv: {e}")
        try:
            from tianshou.env import DummyVectorEnv
            train_envs = DummyVectorEnv([wrapped_env_fn_train for _ in range(max(1, n_envs))])
            test_envs = DummyVectorEnv([wrapped_env_fn_test for _ in range(max(1, n_envs // 2))])
        except Exception as e2:
            # Финальный шанс: одинарные окружения
            print(f"⚠️ DummyVectorEnv тоже не создался, пробую одиночные env: {e2}")
            train_envs = [wrapped_env_fn_train()]
            test_envs = [wrapped_env_fn_test()]
    try:
        # Вычисляем фактическое число env для корректной конфигурации буфера/сборки
        env_count = (len(train_envs) if isinstance(train_envs, (list, tuple)) else getattr(train_envs, 'env_num', n_envs))
    except Exception:
        env_count = max(1, n_envs)
    try:
        print(f"🧩 Tianshou envs: train={env_count}, test={len(test_envs) if isinstance(test_envs, (list, tuple)) else getattr(test_envs, 'env_num', max(1, env_count // 2))}")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Детерминизм для воспроизводимости (может снизить скорость)
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    # Применим training_params overrides (lr, batch_size, eps, repeats, target_freq, memory)
    # 1) Пытаемся взять оптимизированные конфиги по символу
    override = None
    try:
        if isinstance(symbol, str):
            symu = symbol.upper()
            if 'TON' in symu:
                from agents.vdqn.hyperparameter.ton_optimized_config import TON_OPTIMIZED_CONFIG as _SYM_CFG
                override = _SYM_CFG
                print("⚙️ Применяю оптимизированный конфиг для TON (hyperparameter/ton_optimized_config.py)")
            elif 'BNB' in symu:
                from agents.vdqn.hyperparameter.bnb_optimized_config import BNB_OPTIMIZED_CONFIG as _SYM_CFG
                override = _SYM_CFG
                print("⚙️ Применяю оптимизированный конфиг для BNB (hyperparameter/bnb_optimized_config.py)")
            elif 'BTC' in symu:
                from agents.vdqn.hyperparameter.btc_optimized_config import BTC_OPTIMIZED_CONFIG as _SYM_CFG
                override = _SYM_CFG
                print("⚙️ Применяю оптимизированный конфиг для BTC (hyperparameter/btc_optimized_config.py)")
    except Exception as _e:
        print(f"ℹ️ Не удалось загрузить оптимизированный конфиг по символу: {_e}")
        override = None
    # 2) Если есть внешний override из конфига символа — он приоритетнее
    try:
        ext_override = get_symbol_override(symbol if not _is_multi_crypto(dfs) else None) if isinstance(dfs, dict) else None
        if ext_override:
            override = ext_override
            print("⚙️ Применяю внешний override из конфигурации символа (get_symbol_override)")
    except Exception:
        pass
    if override and 'training_params' in override:
        tp = override['training_params']
        # GPU-owned параметры НЕ должны быть per-symbol.
        GPU_OWNED_KEYS = {'batch_size', 'memory_size', 'train_repeats'}
        try:
            lr = tp.get('lr', lr)
            gamma = tp.get('gamma', gamma)
        except Exception:
            pass
        try:
            if 'batch_size' in tp and 'batch_size' in GPU_OWNED_KEYS:
                print(f"⚠️ SYMBOL OVERRIDE игнорирует GPU-owned параметр: batch_size={tp.get('batch_size')}")
            else:
                batch_size = tp.get('batch_size', batch_size)
        except Exception:
            pass
        try:
            # epsilon схема
            eps_start_override = tp.get('eps_start', None)
            eps_final_override = tp.get('eps_final', None)
            eps_decay_steps_override = tp.get('eps_decay_steps', None)
        except Exception:
            eps_start_override = None; eps_final_override = None; eps_decay_steps_override = None
        try:
            target_update_freq = tp.get('target_update_freq', target_update_freq)
            try:
                if 'memory_size' in tp and 'memory_size' in GPU_OWNED_KEYS:
                    print(f"⚠️ SYMBOL OVERRIDE игнорирует GPU-owned параметр: memory_size={tp.get('memory_size')}")
                else:
                    memory_size = tp.get('memory_size', memory_size)
            except Exception:
                pass
            try:
                if 'train_repeats' in tp and 'train_repeats' in GPU_OWNED_KEYS:
                    print(f"⚠️ SYMBOL OVERRIDE игнорирует GPU-owned параметр: train_repeats={tp.get('train_repeats')}")
                    train_repeats = train_repeats
                else:
                    train_repeats = tp.get('train_repeats', 1)
            except Exception:
                train_repeats = 1
        except Exception:
            train_repeats = 1
    else:
        eps_start_override = None; eps_final_override = None; eps_decay_steps_override = None; train_repeats = 1

    # Спец-логика для BNB: мягкие оверрайды для стабильности
    try:
        if (not _is_multi_crypto(dfs)) and isinstance(symbol, str) and ('BNB' in symbol.upper()):
            # Снижаем exploration и lr, увеличиваем batch
            try:
                eps_start_override = eps_start_override if eps_start_override is not None else 0.20
                eps_final_override = eps_final_override if eps_final_override is not None else 0.02
                eps_decay_steps_override = eps_decay_steps_override if eps_decay_steps_override is not None else int(1_000_000 * 0.75)
            except Exception:
                pass
            try:
                batch_size = max(192, int(batch_size))
                lr = min(float(lr), 2e-4)
            except Exception:
                pass
            # Чуть чаще сохранять буфер
            try:
                buffer_save_frequency = max(400, int(buffer_save_frequency or 0) or 800)
            except Exception:
                pass
    except Exception:
        pass

    net = DQNN(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(512, 512, 256),
        dropout_rate=0.1,
        layer_norm=True,
        dueling=True,
        activation='relu',
        use_residual=True,
        use_swiglu=False,
    ).to(device)

    try:
        should_compile = False
        source = 'default'
        # 1) Пробуем взять из GPU-конфига
        try:
            from agents.vdqn.cfg.gpu_configs import get_optimal_config
            _gpu_cfg = get_optimal_config()
            if hasattr(_gpu_cfg, 'use_torch_compile'):
                should_compile = bool(getattr(_gpu_cfg, 'use_torch_compile'))
                source = 'gpu_config'
        except Exception:
            pass
        # 2) Проверим наличие компилятора (gcc/clang); если нет — отключим compile
        try:
            import shutil
            has_cc = bool(shutil.which('cc') or shutil.which('gcc') or shutil.which('clang'))
            if should_compile and not has_cc:
                print("⚠️ torch.compile отключен: не найден C-компилятор (cc/gcc/clang)")
                should_compile = False
                source = f"{source}+no_cc"
        except Exception:
            pass
        if should_compile and torch.cuda.is_available() and hasattr(torch, 'compile'):
            # используем глобально импортированный os
            # Снизим вероятность крэшей индуктора в сабпроцессах
            os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
            # Выберем backend: Ampere+ (cc>=8.0) → inductor, иначе безопасный aot_eager
            backend = 'inductor'
            try:
                cc_major, cc_minor = torch.cuda.get_device_capability(0)
                if int(cc_major) < 8:
                    backend = 'aot_eager'
            except Exception:
                backend = 'aot_eager'
            print(f"⚙️ torch.compile: enabled ({source}), backend={backend}, TORCHINDUCTOR_COMPILE_THREADS={os.environ.get('TORCHINDUCTOR_COMPILE_THREADS')}")
            compiled_net = torch.compile(net, backend=backend)
            # Тёплый прогон для раннего выявления проблем и безопасного фолбэка
            try:
                with torch.no_grad():
                    dummy = torch.zeros(1, int(obs_dim), dtype=torch.float32, device=device)
                    _ = compiled_net(dummy)
                net = compiled_net
            except Exception as ce:
                print(f"⚠️ torch.compile warmup failed: {ce}; falling back to eager")
                source = f"{source}+warmup_fail"
        else:
            print(f"⚙️ torch.compile: disabled ({source})")
    except Exception:
        pass

    # Оптимизатор как в legacy DQN: AdamW
    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    policy = MaskedDQNPolicy(
        model=net,
        optim=optim,
        action_space=action_space_gym,
        discount_factor=gamma,
        estimation_step=n_step,
        target_update_freq=target_update_freq,
        is_double=True,
    )
    try:
        print(f"🎯 target_update_freq={target_update_freq}")
    except Exception:
        pass
    # Начальный epsilon и плавное снижение — аналог твоей схемы
    try:
        # Можно передать извне значения, здесь — безопасные дефолты
        eps_start = eps_start_override if eps_start_override is not None else 0.3
        eps_final = eps_final_override if eps_final_override is not None else 0.05
        # Режим ускоренной диагностики исследования
        if debug_exploration:
            try:
                print("🧪 [DEBUG] Exploration override активен: eps_start=1.0, eps_final=0.05, eps_decay_steps=100000")
            except Exception:
                pass
            eps_start = 1.0
            eps_final = 0.05
            eps_decay_steps_override = 100_000
        # Установим стартовый eps
        policy.set_eps(eps_start)
    except Exception:
        pass

    try:
        effective_total = int(memory_size)
        if max_samples_by_budget is not None:
            effective_total = min(effective_total, int(max_samples_by_budget))
    except Exception:
        effective_total = int(memory_size)
    # Привязываем размер буфера к фактическому числу env (без умножения при одиночном env)
    try:
        env_count = (len(train_envs) if isinstance(train_envs, (list, tuple)) else getattr(train_envs, 'env_num', n_envs))
    except Exception:
        env_count = max(1, n_envs)
    buf = VectorReplayBuffer(total_size=int(effective_total), buffer_num=int(env_count))
    # исследование управляется policy.eps; шум коллектора отключаем
    train_collector = Collector(policy, train_envs, buf, exploration_noise=False)
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # Warmup буфера: ускоряет появление разнообразных переходов
    try:
        warmup_steps = 0
        try:
            warmup_steps = int(str(get_config_value('TS_WARMUP_STEPS', '0')))
        except Exception:
            warmup_steps = 0
        if debug_exploration and warmup_steps <= 0:
            warmup_steps = 10_000
        if warmup_steps and warmup_steps > 0:
            _prev_eps = getattr(policy, 'eps', None)
            try:
                policy.set_eps(1.0)
            except Exception:
                pass
            try:
                print(f"🧪 Warmup: collect {warmup_steps} steps (eps=1.0)")
                _res = train_collector.collect(n_step=warmup_steps)
                try:
                    _buf_size = getattr(train_collector.buffer, 'size', None)
                    print(f"🧪 Warmup завершён: buffer_size={_buf_size} result={_res}")
                except Exception:
                    pass
            finally:
                try:
                    if _prev_eps is not None:
                        policy.set_eps(_prev_eps)
                except Exception:
                    pass
    except Exception:
        pass
    # Warm-start буфера (поддержка Tianshou/legacy форматов)
    try:
        if load_buffer_path and os.path.isfile(load_buffer_path):
            import pickle
            with open(load_buffer_path, 'rb') as f:
                loaded_buf = pickle.load(f)
            if hasattr(loaded_buf, 'add') and hasattr(loaded_buf, '__len__'):
                train_collector.buffer = loaded_buf
                buf = loaded_buf
                print(f"♻️  Загружен Tianshou replay buffer из {load_buffer_path}")
            else:
                # Попробуем сконвертировать legacy формат в текущий buf
                if _try_load_legacy_replay(load_buffer_path, buf):
                    print(f"♻️  Конвертирован legacy replay в текущий буфер")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить буфер: {e}")

    steps_per_episode = episode_length or 2000
    # Используем ровно UI-настройки без жёстких минимумов
    train_steps = max(1, episodes * steps_per_episode)

    # Early stopping и best-модель по среднему тест-награждению
    best_reward = None
    best_epoch = -1
    recent_test_rewards: deque = deque(maxlen=60)
    stopped_by_trend = False
    # История для train_result (проксируем winrate как среднюю тест-награду по эпохам)
    epoch_test_rewards: list[float] = []

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int, **kwargs):
        nonlocal best_reward, best_epoch
        # Сохраняем best_model при улучшении
        try:
            cur_best = kwargs.get('best_reward', None)
            if cur_best is None:
                return
            if best_reward is None or cur_best > best_reward:
                best_reward = cur_best
                best_epoch = epoch
                import shutil as _sh
                _sh.copy2(model_path, run_dir / 'best_model.pth')
                # При улучшении — по флагу также сохраняем replay
                if save_replay_on_improvement:
                    import pickle
                    with open(replay_path, 'wb') as f:
                        pickle.dump(buf, f)
        except Exception:
            pass
        # Периодические чекпоинты model/last и replay
        try:
            if save_frequency and epoch > 0 and (epoch % save_frequency == 0):
                # Переэкспортируем нормализацию (на случай, если менялась)
                try:
                    latest_norm = _export_norm_stats_safe(dfs, episode_length)
                except Exception:
                    latest_norm = None
                torch.save({'model': net.state_dict(), 'normalization_stats': latest_norm or norm_stats_cached}, model_path)
                import shutil as _sh
                if model_path.exists():
                    _sh.copy2(model_path, run_dir / 'last_model.pth')
                # Сохраняем encoder_only
                if hasattr(net, 'get_feature_extractor'):
                    encoder = net.get_feature_extractor()
                    if encoder is not None:
                        enc_path = run_dir / 'encoder_only.pth'
                        torch.save({'encoder': encoder.state_dict()}, enc_path)
            freq_buf = buffer_save_frequency if buffer_save_frequency is not None else max(200, save_frequency * 4)
            if epoch > 0 and (epoch % freq_buf == 0):
                import pickle
                with open(replay_path, 'wb') as f:
                    pickle.dump(buf, f)
            # Обновляем manifest.json с прогрессом
            try:
                approx_episodes = int(env_step / float(steps_per_episode)) if steps_per_episode else None
                manifest = {
                    'run_id': this_run_id,
                    'parent_run_id': parent_run_id,
                    'root_id': root_id or this_run_id,
                    'symbol': symbol_dir,
                    'seed': (int(seed) if isinstance(seed, int) else None),
                    'episodes_start': 0,
                    'episodes_end': approx_episodes,
                    'episodes_added': approx_episodes,
                    'episodes_last': approx_episodes,
                    'episodes_best': (best_epoch if best_epoch >= 0 else None),
                    'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'debug': bool(debug_run or debug_exploration),
                    'debug_flags': [
                        f for f, on in (
                            ('TS_DEBUG_RUN', debug_run),
                            ('TS_DEBUG_EXPLORATION', debug_exploration),
                            ('TS_FORCE_SINGLE', force_single),
                            ('TS_FORCE_DUMMY', force_dummy),
                        ) if on
                    ],
                    'artifacts': {
                        'model': 'model.pth',
                        'replay': 'replay.pkl' if Path(replay_path).exists() else None,
                        'result': 'train_result.pkl',
                        'best_model': 'best_model.pth' if (run_dir / 'best_model.pth').exists() else None,
                        'last_model': 'last_model.pth' if (run_dir / 'last_model.pth').exists() else None,
                        'encoder_only': 'encoder_only.pth' if (run_dir / 'encoder_only.pth').exists() else None,
                        'all_trades': 'all_trades.json' if (run_dir / 'all_trades.json').exists() else None,
                    },
                    'best_metrics': {
                        'winrate': (winrate_from_trades if (winrate_from_trades is not None) else (max(epoch_test_rewards) if epoch_test_rewards else None)),
                        'reward': (best_reward if best_reward is not None else (max(epoch_test_rewards) if epoch_test_rewards else None))
                    }
                }
                # Логирование перед записью (progress)
                print(f"💡[Progress] Сохраняем manifest.json в {run_dir / 'manifest.json'}")
                print(manifest)
                with open(run_dir / 'manifest.json', 'w', encoding='utf-8') as mf:
                    _json.dump(manifest, mf, ensure_ascii=False, indent=2)
                    try:
                        mf.flush(); os.fsync(mf.fileno())
                    except Exception:
                        pass
            except Exception as me_prog:
                print(f"⚠️ Не удалось обновить manifest.json (progress): {me_prog}")
                import traceback; traceback.print_exc()
        except Exception:
            pass

# Трендовый stop_fn с минимумом "эпизодов"
    min_episodes_before_stopping = max(4000, episodes // 3)
    patience_limit = max(8000, episodes // 2)

    def stop_fn(mean_rewards: float) -> bool:
        # Собираем историю для тренда
        try:
            recent_test_rewards.append(mean_rewards)
            epoch_test_rewards.append(float(mean_rewards))
            # Для коротких тренировок (< 50 эпизодов) отключаем early stopping
            if episodes < 50:
                return False
            if len(recent_test_rewards) < 60:
                return False
            recent_avg = float(np.mean(list(recent_test_rewards)[-30:]))
            older_avg = float(np.mean(list(recent_test_rewards)[:-30]))
            declining = recent_avg + 0.05 < older_avg
            # Примерная оценка "эпизодов" по шагам
            approx_episodes = int((train_collector.buffer.size / max(1, steps_per_episode)))
            if approx_episodes < min_episodes_before_stopping:
                return False
            should_stop = declining and approx_episodes >= patience_limit
            if should_stop:
                nonlocal stopped_by_trend
                stopped_by_trend = True
            return should_stop
        except Exception:
            return False

    # Линейный шедулер epsilon по шагам (train_fn): строго как legacy DQN
    eps_decay_steps = int(eps_decay_steps_override) if (isinstance(eps_decay_steps_override, (int, float)) and eps_decay_steps_override) else train_steps

    # Heartbeat: периодический лог раз в N секунд
    try:
        heartbeat_interval_sec = int(get_config_value('TS_HEARTBEAT_SEC', '60'))
    except Exception:
        heartbeat_interval_sec = 60
    last_heartbeat_ts = time.time()
    last_heartbeat_env_step = 0

    def train_fn(epoch: int, env_step: int):
        # Лог подтверждения вызова функции train_fn
        #print(f"⚙️ train_fn: Вызван. epoch={epoch}, env_step={env_step}")
        try:
            frac = min(1.0, env_step / float(max(1, eps_decay_steps)))
            cur_eps = max(eps_final, eps_start + (eps_final - eps_start) * frac)
            policy.set_eps(cur_eps)
            # Периодический лог ресурсов
            if env_step % (n_envs * 1_000) == 0:
                _log_resource_usage(tag="ts-train", extra=f"env_step={env_step} eps={cur_eps:.3f}")
            # Временный отладочный прогресс каждые 50 шагов
            if env_step % 200 == 0:
                try:
                    buf_size = getattr(train_collector.buffer, 'size', None)
                    print(f"🧪 train_fn (debug): env_step={env_step} eps={cur_eps:.3f} buffer_size={buf_size}")
                except Exception:
                    pass
            # Heartbeat раз в N секунд с оценкой скорости
            nonlocal last_heartbeat_ts, last_heartbeat_env_step
            now = time.time()
            # Первый heartbeat сразу при старте эпохи
            if env_step == 0 and last_heartbeat_env_step == 0:
                print(f"🚀 Начало эпохи {epoch}/{num_epochs}. Текущий шаг: {env_step}, eps={cur_eps:.3f}")
                last_heartbeat_ts = now
                last_heartbeat_env_step = env_step
            if now - last_heartbeat_ts >= heartbeat_interval_sec:
                dt = max(1e-6, now - last_heartbeat_ts)
                dsteps = max(0, env_step - last_heartbeat_env_step)
                sps = dsteps / dt  # steps per second
                print(f"⏱️ heartbeat (time): epoch={epoch}/{num_epochs} env_step={env_step} eps={cur_eps:.3f} speed={sps:.1f} steps/s")
                last_heartbeat_ts = now
                last_heartbeat_env_step = env_step
            # Fallback: если секундный таймер не сработал, печатаем по шагам
            fallback_step_interval = max(50, n_envs * 50)
            if env_step - last_heartbeat_env_step >= fallback_step_interval:
                # обновим и выведем скорость по шагам
                dt = max(1e-6, now - last_heartbeat_ts)
                dsteps = max(0, env_step - last_heartbeat_env_step)
                sps = dsteps / dt if dt > 0 else 0.0
                #print(f"⏱️ heartbeat (step): epoch={epoch}/{num_epochs} env_step={env_step} eps={cur_eps:.3f} speed={sps:.1f} steps/s")
                last_heartbeat_ts = now
                last_heartbeat_env_step = env_step
            # Изредка чистим CUDA-кэш
            if torch.cuda.is_available() and (env_step % (n_envs * 50_000) == 0):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

    episodes_since_best = 0
    lr_min = 1e-5
    lr_plateau_patience = 1000

    def test_fn(epoch: int, env_step: int):
        # Лог подтверждения вызова функции test_fn
        print(f"🧪 test_fn: Вызван. epoch={epoch}, env_step={env_step}")
        try:
            policy.eval()
            collector = test_collector
            result = collector.collect(n_episode=1) # Changed from episode_per_test to 1
            try:
                # Tianshou 1.x: CollectStats имеет returns/lens (+ returns_stat/lens_stat)
                rets = getattr(result, 'returns', None)
                lens = getattr(result, 'lens', None)
                if rets is None:
                    # Старые версии: rews/lens в dict
                    rets = getattr(result, 'rews', None)
                if rets is None or lens is None:
                    # Fallback на dict-доступ
                    try:
                        rets = result["returns"] if "returns" in result else result["rews"]
                        lens = result["lens"]
                    except Exception:
                        # Если и это не удалось — пробрасываем для внешней обработки
                        raise
                # Статистики, если доступны (Tianshou 1.x)
                rew_mean = None; rew_std = None
                len_mean = None; len_std = None
                try:
                    rs = getattr(result, 'returns_stat', None)
                    if rs is not None:
                        rew_mean = float(getattr(rs, 'mean', None)) if getattr(rs, 'mean', None) is not None else None
                        rew_std = float(getattr(rs, 'std', None)) if getattr(rs, 'std', None) is not None else None
                except Exception:
                    pass
                try:
                    ls = getattr(result, 'lens_stat', None)
                    if ls is not None:
                        len_mean = float(getattr(ls, 'mean', None)) if getattr(ls, 'mean', None) is not None else None
                        len_std = float(getattr(ls, 'std', None)) if getattr(ls, 'std', None) is not None else None
                except Exception:
                    pass
                # Если нет готовых стат, считаем от массивов
                if rew_mean is None:
                    try:
                        rew_mean = float(rets.mean())
                    except Exception:
                        rew_mean = float(sum(rets) / max(1, len(rets)))
                if rew_std is None:
                    try:
                        rew_std = float(rets.std())
                    except Exception:
                        rew_std = 0.0
                if len_mean is None:
                    try:
                        len_mean = float(lens.mean())
                    except Exception:
                        len_mean = float(sum(lens) / max(1, len(lens)))
                if len_std is None:
                    try:
                        len_std = float(lens.std())
                    except Exception:
                        len_std = 0.0
                print(f"🎯 Test: epoch={epoch} env_step={env_step} avg_reward={rew_mean:.2f} (std={rew_std:.2f}) avg_len={len_mean:.1f} (std={len_std:.1f})")
            except Exception as se:
                print(f"❌ Ошибка в test_fn(stat): {se}; result={result}")
            finally:
                policy.train() # Вернуться в режим тренировки
        except Exception as e:
            print(f"❌ Ошибка в test_fn: {e}")

    # Разбиваем тренинг на более короткие эпохи
    progress_step_per_epoch = steps_per_episode
    num_epochs = max(1, int(train_steps / float(progress_step_per_epoch)))
    # Конфигурируем step_per_collect
    try:
        _spc = int(str(get_config_value('TS_STEP_PER_COLLECT', '0')))
    except Exception:
        _spc = 0
    step_per_collect_value = _spc if _spc > 0 else (n_envs * 32)
    if debug_run:
        step_per_collect_value = max(step_per_collect_value, 256)

    # Соответствие train_repeats → update_per_step
    try:
        update_per_step_value = float(train_repeats) if isinstance(train_repeats, (int, float)) and float(train_repeats) > 0 else 1.0
    except Exception:
        update_per_step_value = 1.0
    try:
        print(f"🔄 update_per_step={update_per_step_value} (train_repeats={train_repeats})")
    except Exception:
        pass

    # Глобальный heartbeat: печатаем раз в N секунд независимо от train_fn
    try:
        global_hb_interval = int(get_config_value('TS_HEARTBEAT_SEC', '60'))
    except Exception:
        global_hb_interval = 60
    _hb_stop = {'flag': False}

    def _global_heartbeat():
        last_ts = time.time()
        last_env_step = 0
        while not _hb_stop.get('flag', False):
            time.sleep(max(5, min( global_hb_interval, 60)))
            try:
                # Печатаем минимум системные ресурсы и примерный шаг
                _log_resource_usage(tag="ts-global")
                # Если есть train_collector, выведем размер буфера и env_step
                try:
                    buf_size = getattr(train_collector.buffer, 'size', None)
                    print(f"🫀 global-heartbeat: buffer_size={buf_size} (env_step={last_env_step})")
                except Exception:
                    print(f"🫀 global-heartbeat: alive")
            except Exception:
                pass

    _hb_thread = Thread(target=_global_heartbeat, daemon=True)
    _hb_thread.start()

    # ==========================================================================================
    #  Запуск тренировки Tianshou
    # ==========================================================================================
    print(f"🗓️ Trainer: epochs={num_epochs}, step_per_epoch={progress_step_per_epoch}, step_per_collect={step_per_collect_value}")
    print(f"ℹ️ CryptoTradingEnv reset snapshot: requested_episode_length={episode_length}, env_episode_length={actual_episode_length}, min_start={min_valid_start_step if min_valid_start_step is not None else 'n/a'}, start_step={start_step_snapshot if start_step_snapshot is not None else 'n/a'}")
    test_collector.reset()
    train_collector.reset()
    # print(f"📊 Старт тренера: env_step={train_collector.env_step} buffer_size={train_collector.buffer.size}")
    print(f"🏁 Готов к запуску Tianshou offpolicy_trainer для {num_epochs} эпох.")
    # Логгер тренера: используем CSVLogger если доступен, иначе без логгера (у нас есть train.log tee)
    try:
        logger = CSVLogger(output_dir=str(run_dir / 'logs')) if CSVLogger is not None else None
    except Exception:
        logger = None
    try:
        result = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=num_epochs,
            step_per_epoch=progress_step_per_epoch,
            step_per_collect=step_per_collect_value,
            episode_per_test=1,
            batch_size=batch_size,
            # выравниваем поведение: обновлений на шаг среды = train_repeats из legacy
            update_per_step=float(update_per_step_value),
            test_in_train=False,
            train_fn=train_fn,
            test_fn=test_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            stop_fn=stop_fn,
            logger=logger,
            show_progress=False,
        )
    except Exception as e:
        import traceback
        print("❌ Exception in offpolicy_trainer:")
        traceback.print_exc()
        raise

    # Остановим глобальный heartbeat
    try:
        _hb_stop['flag'] = True
    except Exception:
        pass

    # Сохранение модели (+ encoder_only при наличии)
    print(f"✅ Tianshou offpolicy_trainer завершил работу.")
    # Статус сохранения результатов
    saved_train_result = False
    # Подготовка данных
    all_trades = []
    collected_trades = []
    # Краткий итог времени и средних длительностей
    try:
        total_training_time = time.time() - training_start_time
        def _fmt_duration(sec: float) -> str:
            try:
                sec = float(sec)
            except Exception:
                return str(sec)
            if sec < 60:
                return f"{sec:.1f} сек"
            mins = sec / 60.0
            if mins < 60:
                return f"{mins:.1f} мин"
            hours = mins / 60.0
            return f"{hours:.1f} ч"
        try:
            env_steps_total = int(result.get('env_step', 0)) if isinstance(result, dict) else 0
        except Exception:
            env_steps_total = 0
        steps_per_ep = episode_length or 2000
        approx_env_episodes = (env_steps_total // max(1, steps_per_ep)) if env_steps_total > 0 else num_epochs
        avg_per_env_episode = total_training_time / max(1, approx_env_episodes)
        avg_per_epoch = total_training_time / max(1, num_epochs)
        print(
            f"⏱️ Итог: общее время={_fmt_duration(total_training_time)}, "
            f"ср. на эпизод≈{_fmt_duration(avg_per_env_episode)}, "
            f"ср. на эпоху≈{_fmt_duration(avg_per_epoch)}"
        )
    except Exception:
        pass
    payload = {'model': net.state_dict(), 'normalization_stats': norm_stats_cached}
    try:
        if hasattr(net, 'get_feature_extractor'):
            encoder = net.get_feature_extractor()
            if encoder is not None:
                enc_path = run_dir / 'encoder_only.pth'
                torch.save({'encoder': encoder.state_dict()}, enc_path)
    except Exception:
        pass
    torch.save(payload, model_path)
    print(f"💾 Сохранил итоговую модель в {model_path}")
    last_model_path = run_dir / 'last_model.pth'
    try:
        import shutil as _sh
        if model_path.exists():
            _sh.copy2(model_path, last_model_path)
    except Exception:
        pass

    try:
        import pickle
        with open(replay_path, 'wb') as f:
            pickle.dump(buf, f)
    except Exception:
        pass

    # Соберём агрегаты продления эпизодов из train_envs (если возможно локально)
    try:
        def _collect_extension_stats(env_container):
            total_ext = 0
            total_steps_ext = 0
            try:
                # Доступно только для локальных env (список или DummyVectorEnv)
                env_list = []
                if hasattr(env_container, 'envs'):
                    env_list = getattr(env_container, 'envs') or []
                elif isinstance(env_container, (list, tuple)):
                    env_list = env_container
                for _env in env_list:
                    try:
                        cur = _env
                        # Разворачиваем обёртки
                        for _ in range(6):
                            if hasattr(cur, 'env'):
                                cur = getattr(cur, 'env')
                            else:
                                break
                        te = getattr(cur, 'total_episode_extensions', 0)
                        ts = getattr(cur, 'total_extension_steps', 0)
                        total_ext += int(te or 0)
                        total_steps_ext += int(ts or 0)
                    except Exception:
                        pass
            except Exception:
                pass
            return total_ext, total_steps_ext

        total_ext_count, total_ext_steps = _collect_extension_stats(train_envs)
    except Exception:
        total_ext_count, total_ext_steps = 0, 0

    # Обновляем чекпоинт-метаданные (manifest, best_model и т.д.)
    try:
        save_checkpoint_fn(
            epoch=num_epochs,
            env_step=int(result.get('env_step', 0)) if isinstance(result, dict) else 0,
            gradient_step=int(result.get('gradient_step', 0)) if isinstance(result, dict) else 0,
            best_reward=result.get('best_reward', best_reward) if isinstance(result, dict) else best_reward,
        )
    except Exception:
        pass

    # Подготовка train_result.pkl в формате, совместимом с анализатором
    print("—> Начало общего блока подготовки train_result.pkl")
    try:
        total_training_time = time.time() - training_start_time
        print("—> total_training_time вычислено: ", total_training_time)

        # Метаданные окружения (снимок с single_env)
        try:
            # Funding features presence from dfs
            try:
                df5_cols = list(dfs.get('df_5min').columns) if isinstance(dfs, dict) and hasattr(dfs.get('df_5min'), 'columns') else []
            except Exception:
                df5_cols = []
            funding_cols = ['funding_rate_bp', 'funding_rate_ema', 'funding_rate_change', 'funding_sign']
            funding_present = [c for c in funding_cols if c in df5_cols]

            gym_snapshot = {}
            if single_env_snapshot is not None:
                gym_snapshot = {
                    'symbol': single_env_snapshot.get('symbol'),
                    'lookback_window': single_env_snapshot.get('lookback_window'),
                    'indicators_config': single_env_snapshot.get('indicators_config'),
                    'reward_scale': single_env_snapshot.get('cfg_reward_scale', 1.0),
                    'episode_length': single_env_snapshot.get('episode_length'),
                    'position_sizing': single_env_snapshot.get('position_sizing', {}),
                    'risk_management': single_env_snapshot.get('risk_management', {}),
                    'observation_space_shape': single_env_snapshot.get('observation_space_shape'),
                    'step_minutes': single_env_snapshot.get('step_minutes', 5),
                    'funding_features': {
                        'present_in_input_df': funding_present,
                        'included': bool(funding_present),
                    },
                }
            else:
                gym_snapshot = {
                    'symbol': None,
                    'lookback_window': None,
                    'indicators_config': None,
                    'reward_scale': 1.0,
                    'episode_length': None,
                    'position_sizing': {},
                    'risk_management': {},
                    'observation_space_shape': None,
                    'step_minutes': 5,
                    'funding_features': {
                        'present_in_input_df': funding_present,
                        'included': bool(funding_present),
                    },
                }
        except Exception:
            gym_snapshot = {}

        # Снимок системы
        train_metadata = {
            'created_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        }

        # Примерная оценка реальных эпизодов
        steps_per_episode = episode_length or 2000
        approx_actual_episodes = int(train_steps * n_envs / steps_per_episode)
        # Точные шаги если есть
        try:
            if isinstance(result, dict) and 'env_step' in result:
                env_steps_total = int(result['env_step'])
            else:
                env_steps_total = int(train_steps * n_envs)
        except Exception:
            env_steps_total = int(train_steps * n_envs)

        cfg_like = {
            'model_path': str(model_path),
            'replay_buffer_path': str(replay_path),
            'run_id': this_run_id,
        }

        weights_info = {
            'model_path': str(model_path),
            'buffer_path': str(replay_path),
            'model_sha256': _sha256_of_file(str(model_path)) if model_path.exists() else None,
            'buffer_sha256': _sha256_of_file(str(replay_path)) if Path(replay_path).exists() else None,
        }

        # Агрегаты поведения (по буферу): action_counts и эпизоды с трейдом
        action_counts_total = {0: 0, 1: 0, 2: 0}
        episodes_with_trade_count = 0
        buy_attempts_total = 0
        buy_rejected_vol_total = 0
        buy_rejected_roi_total = 0
        try:
            acts = getattr(buf, 'act', None)
            dones = getattr(buf, 'done', None)
            infos = getattr(buf, 'info', None)
            if acts is not None:
                arr = np.array(acts).reshape(-1)
                for a in (0, 1, 2):
                    action_counts_total[a] = int(np.sum(arr == a))
                if dones is not None:
                    d = np.array(dones).reshape(-1)
                    # Пробежимся по эпизодам: считаем эпизод содержащим трейд, если есть act!=0
                    ep_has_trade = False
                    for i, a in enumerate(arr):
                        if a != 0:
                            ep_has_trade = True
                        if i < len(d) and d[i]:
                            if ep_has_trade:
                                episodes_with_trade_count += 1
                            ep_has_trade = False
            # Попробуем собрать buy-* метрики и трейды из info
            if infos is not None:
                try:
                    flat_infos = np.array(infos, dtype=object).reshape(-1)
                    collected_trades = []
                    for it in flat_infos:
                        if isinstance(it, dict):
                            buy_attempts_total += int(it.get('buy_attempts', 0) or 0)
                            buy_rejected_vol_total += int(it.get('buy_rejected_vol', 0) or 0)
                            buy_rejected_roi_total += int(it.get('buy_rejected_roi', 0) or 0)
                            if 'trades_episode' in it and isinstance(it['trades_episode'], list):
                                collected_trades.extend(it['trades_episode'])
                    if collected_trades:
                        # При желании можно усечь до разумного размера
                        all_trades = collected_trades
                except Exception:
                    pass
        except Exception:
            pass

        # Adaptive normalization snapshot (single or multi)
        adaptive_snapshot = {}
        try:
            if _is_multi_crypto(dfs):
                per_symbol = {}
                for sym, data in (dfs.items() if isinstance(dfs, dict) else []):
                    try:
                        df5s = data.get('df_5min') if isinstance(data, dict) else None
                        if df5s is None:
                            continue
                        base_profile = adaptive_normalizer.get_crypto_profile(sym)
                        market = adaptive_normalizer.analyze_market_conditions(df5s)
                        adapted = adaptive_normalizer.adapt_parameters(sym, df5s)
                        per_symbol[sym] = {
                            'base_profile': base_profile,
                            'market_conditions': market,
                            'adapted_params': adapted,
                        }
                    except Exception:
                        continue
                adaptive_snapshot = {'per_symbol': per_symbol}
            else:
                df5 = dfs.get('df_5min') if isinstance(dfs, dict) else None
                sym = symbol
                if df5 is not None and isinstance(sym, str):
                    base_profile = adaptive_normalizer.get_crypto_profile(sym)
                    market = adaptive_normalizer.analyze_market_conditions(df5)
                    adapted = adaptive_normalizer.adapt_parameters(sym, df5)
                    adaptive_snapshot = {
                        'symbol': sym,
                        'base_profile': base_profile,
                        'market_conditions': market,
                        'adapted_params': adapted,
                    }
        except Exception:
            adaptive_snapshot = {}

        # Статистика по сделкам
        try:
            winrate_from_trades = None
            avg_roi = None
            num_trades = len(all_trades) if isinstance(all_trades, list) else 0
            total_profit = None
            total_loss = None
            avg_duration = None
            if num_trades > 0:
                rois = [float(t.get('roi', 0.0)) for t in all_trades if isinstance(t, dict)]
                if rois:
                    wins = sum(1 for r in rois if r > 0)
                    winrate_from_trades = wins / len(rois)
                    avg_roi = float(np.mean(rois))
                    total_profit = float(sum(r for r in rois if r > 0))
                    total_loss = float(abs(sum(r for r in rois if r < 0)))
                try:
                    durations = [float(t.get('duration', 0.0)) for t in all_trades if isinstance(t, dict) and t.get('duration') is not None]
                    if durations:
                        avg_duration = float(np.mean(durations))
                except Exception:
                    avg_duration = None
        except Exception:
            winrate_from_trades = None
            avg_roi = None
            total_profit = None
            total_loss = None
            avg_duration = None

        # Финальные значения epsilon и lr
        try:
            epsilon_final_value = getattr(policy, 'eps', None)
        except Exception:
            epsilon_final_value = None
        try:
            current_lr = None
            for g in optim.param_groups:
                current_lr = g.get('lr', None)
                break
            learning_rate_final_value = float(current_lr) if current_lr is not None else None
        except Exception:
            learning_rate_final_value = None

        # Сбор агрегатов причин продаж через векторное API (устойчиво для Subproc/Dummy)
        sell_types_agg = {}
        try:
            # 1) Попытка получить экспорт прямо из сред (работает в SubprocVectorEnv через call)
            export_list = None
            try:
                if hasattr(train_envs, 'call'):
                    export_list = train_envs.call('export_cumulative_sell_types')
                elif hasattr(train_envs, 'get_attr'):
                    export_list = train_envs.get_attr('export_cumulative_sell_types')
            except Exception:
                export_list = None
            if isinstance(export_list, list) and export_list:
                for st in export_list:
                    try:
                        if callable(st):
                            st = st()
                        if isinstance(st, dict):
                            for k, v in st.items():
                                sell_types_agg[k] = sell_types_agg.get(k, 0) + int(v or 0)
                    except Exception:
                        pass
            # 2) Фолбэк: из врапперов (snapshot при done)
            if not sell_types_agg:
                env_list = []
                if hasattr(train_envs, 'envs'):
                    env_list = getattr(train_envs, 'envs') or []
                elif isinstance(train_envs, (list, tuple)):
                    env_list = train_envs
                for _env in env_list:
                    try:
                        snap = getattr(_env, 'cumulative_sell_stats', None)
                        if isinstance(snap, dict) and snap:
                            for k, v in snap.items():
                                sell_types_agg[k] = sell_types_agg.get(k, 0) + int(v or 0)
                    except Exception:
                        pass
            # 3) Фолбэк: напрямую из базовых env (cumulative_sell_types / sell_types)
            if not sell_types_agg:
                env_list = []
                if hasattr(train_envs, 'envs'):
                    env_list = getattr(train_envs, 'envs') or []
                elif isinstance(train_envs, (list, tuple)):
                    env_list = train_envs
                for _env in env_list:
                    try:
                        cur = _env
                        for _ in range(10):
                            if hasattr(cur, 'env'):
                                cur = getattr(cur, 'env')
                            else:
                                break
                        st = getattr(cur, 'cumulative_sell_types', None)
                        if not isinstance(st, dict):
                            st = getattr(cur, 'sell_types', None)
                        if isinstance(st, dict):
                            for k, v in st.items():
                                sell_types_agg[k] = sell_types_agg.get(k, 0) + int(v or 0)
                    except Exception:
                        pass
        except Exception:
            sell_types_agg = {}

        training_results = {
            'episodes': episodes,
            'actual_episodes': approx_actual_episodes,
            'total_training_time': total_training_time,
            'episode_winrates': epoch_test_rewards,  # прокси по тест-наградам
            'all_trades': all_trades,
            'bad_trades': [],
            'bad_trades_count': 0,
            'bad_trades_percentage': 0.0,
            'best_winrate': (max(epoch_test_rewards) if epoch_test_rewards else winrate_from_trades),
            'final_stats': {
                'num_trades': num_trades,
                'winrate_from_trades': winrate_from_trades,
                'avg_roi': avg_roi,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'avg_duration': avg_duration,
            },
            'training_date': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'model_path': str(model_path),
            'buffer_path': str(replay_path),
            'symbol': symbol,
            'model_id': this_run_id,
            'early_stopping_triggered': bool(stopped_by_trend),
            'reward_scale': float(single_env_snapshot.get('cfg_reward_scale', 1.0)) if isinstance(single_env_snapshot, dict) else 1.0,
            'total_steps_processed': env_steps_total,
            'episode_length': episode_length,
            'action_counts_total': action_counts_total,
            'episodes_with_trade_count': episodes_with_trade_count,
            'episodes_with_trade_ratio': (episodes_with_trade_count / float(approx_actual_episodes)) if approx_actual_episodes > 0 else 0.0,
            'buy_attempts_total': buy_attempts_total or 0,
            'buy_rejected_vol_total': buy_rejected_vol_total or 0,
            'buy_rejected_roi_total': buy_rejected_roi_total or 0,
            'epsilon_final_value': epsilon_final_value,
            'learning_rate_final_value': learning_rate_final_value,
            'buy_accept_rate': ((action_counts_total.get(1, 0) or 0) / float(buy_attempts_total) if (buy_attempts_total or 0) > 0 else 0.0),
            'avg_minutes_between_buys': (
                ((env_steps_total * float(gym_snapshot.get('step_minutes', 5))) / float(action_counts_total.get(1, 0)))
                if (action_counts_total.get(1, 0) or 0) > 0 else None
            ),
            'best_episode_idx': (best_epoch if isinstance(best_epoch, int) and best_epoch >= 0 else None),
            # Продления эпизодов (агрегаты с train_envs)
            'episode_extensions_total': int(total_ext_count),
            'episode_extension_steps_total': int(total_ext_steps),
            'sell_types_total': sell_types_agg,
        }
        print(f"training_results train_result.pkl {training_results}")
        enriched_results = {
            **training_results,
            'train_metadata': train_metadata,
            'cfg_snapshot': cfg_like,
            'gym_snapshot': gym_snapshot,
            'adaptive_normalization': adaptive_snapshot,
            'hyperparameters': {
                'lr': lr,
                'gamma': gamma,
                'batch_size': batch_size,
                'target_update_freq': target_update_freq,
                'eps_start': (eps_start_override if 'eps_start_override' in locals() and eps_start_override is not None else None),
                'eps_final': (eps_final_override if 'eps_final_override' in locals() and eps_final_override is not None else None),
                'eps_decay_steps': (eps_decay_steps_override if 'eps_decay_steps_override' in locals() and eps_decay_steps_override is not None else None),
                'memory_size': memory_size,
                'train_repeats': train_repeats,
            },
            'architecture': {
                'main': {'model_class': net.__class__.__name__},
                'target': {},
            },
            'weights': weights_info,
        }

        print("—> Вхожу в блок сохранения train_result.pkl")
        results_file = run_dir / 'train_result.pkl'
        import pickle
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(enriched_results, f)
            saved_train_result = True
            print(f"💾 train_result.pkl сохранён: {results_file}")
        except Exception as pe:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Не удалось сохранить train_result.pkl ({pe})")

        # Сохраняем полный список сделок за ран в all_trades.json (если есть)
        try:
            if isinstance(all_trades, list) and len(all_trades) > 0:
                trades_json_path = run_dir / 'all_trades.json'
                try:
                    # Нормализуем сделки к сериализуемому виду
                    def _norm_trade(t):
                        if isinstance(t, dict):
                            return {
                                k: v for k, v in t.items()
                                if isinstance(k, str) and isinstance(v, (int, float, str, bool, type(None)))
                            }
                        return t
                    safe_trades = [_norm_trade(t) for t in all_trades]
                    with open(trades_json_path, 'w', encoding='utf-8') as tf:
                        _json.dump(safe_trades, tf, ensure_ascii=False)
                    print(f"💾 all_trades.json сохранён: {trades_json_path} (count={len(safe_trades)})")
                except Exception as te:
                    print(f"⚠️ Не удалось сохранить all_trades.json: {te}")
        except Exception:
            pass

        # Подробный финальный вывод (как в старом тренере)
        try:
            print("\n" + "="*60)
            training_name = symbol
            print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ для {training_name}")
            print("="*60)
            print(f"⏱️ ВРЕМЯ ОБУЧЕНИЯ:")
            print(f"  • Общее время: {total_training_time:.2f} секунд ({total_training_time/60:.1f} минут)")
            planned = episodes
            actual = approx_actual_episodes
            if actual and actual > 0:
                print(f"  • Время на эпизод: {total_training_time/max(1,actual):.2f} секунд")
                print(f"  • Эпизодов в минуту: {actual/(total_training_time/60):.1f}")
            print(f"\n📈 ЭПИЗОДЫ:")
            print(f"  • Планируемые эпизоды: {planned}")
            print(f"  • Реальные эпизоды (оценка): {actual}")
            print(f"  • Early Stopping: {'Сработал' if stopped_by_trend else 'Не сработал'}")
            try:
                print(f"  • Продления эпизодов: {int(total_ext_count)} раз (+{int(total_ext_steps)} шагов)")
            except Exception:
                pass
            if epoch_test_rewards:
                try:
                    avg_wr = float(np.mean(epoch_test_rewards))
                    print(f"  • Средний winrate (по тест-награде): {avg_wr:.3f}")
                except Exception:
                    pass
            if isinstance(all_trades, list) and len(all_trades) > 0:
                print(f"\n💰 Общая статистика по сделкам:")
                if total_profit is not None:
                    print(f"  • Общая прибыль: {total_profit:.4f}")
                if total_loss is not None:
                    print(f"  • Общий убыток: {total_loss:.4f}")
                if avg_duration is not None:
                    print(f"  • Средняя длительность сделки: {avg_duration:.1f} минут")
            # Печать агрегатов причин продаж
            if isinstance(sell_types_agg, dict) and sell_types_agg:
                print(f"\n🧾 Причины продаж (агрегат):")
                for k, v in sell_types_agg.items():
                    print(f"  • {k}: {int(v)}")
            else:
                print(f"\n⚠️ Нет сделок (по собранной выборке)")
        except Exception as se:
            print(f"⚠️ Не удалось вывести финальную статистику: {se}")

        # manifest.json совместимый
        try:
            manifest = {
                'run_id': this_run_id,
                'parent_run_id': parent_run_id,
                'root_id': root_id or this_run_id,
                'symbol': symbol_dir,
                'seed': (int(seed) if isinstance(seed, int) else None),
                'episodes_start': 0,
                'episodes_end': approx_actual_episodes,
                'episodes_added': approx_actual_episodes,
                'episodes_last': approx_actual_episodes,
                'episodes_best': (best_epoch if best_epoch >= 0 else None),
                'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'debug': bool(debug_run or debug_exploration),
                'debug_flags': [
                    f for f, on in (
                        ('TS_DEBUG_RUN', debug_run),
                        ('TS_DEBUG_EXPLORATION', debug_exploration),
                        ('TS_FORCE_SINGLE', force_single),
                        ('TS_FORCE_DUMMY', force_dummy),
                    ) if on
                ],
                'artifacts': {
                    'model': 'model.pth',
                    'replay': 'replay.pkl' if Path(replay_path).exists() else None,
                    'result': 'train_result.pkl',
                    'best_model': 'best_model.pth' if (run_dir / 'best_model.pth').exists() else None,
                    'last_model': 'last_model.pth' if (run_dir / 'last_model.pth').exists() else None,
                },
                'best_metrics': {
                    'winrate': (winrate_from_trades if (winrate_from_trades is not None) else (max(epoch_test_rewards) if epoch_test_rewards else None)),
                    'reward': (best_reward if best_reward is not None else (max(epoch_test_rewards) if epoch_test_rewards else None))
                }
            }
            # Логирование перед записью (final)
            print(f"💡[Final] Сохраняем manifest.json в {run_dir / 'manifest.json'}")
            print(manifest)
            with open(run_dir / 'manifest.json', 'w', encoding='utf-8') as mf:
                _json.dump(manifest, mf, ensure_ascii=False, indent=2)
            print(f"💾 manifest.json сохранён: {run_dir / 'manifest.json'}")
            try:
                with open(run_dir / 'manifest.json', 'r', encoding='utf-8') as _mf_check:
                    _mf_check.flush(); os.fsync(_mf_check.fileno())
            except Exception:
                pass
        except Exception as me:
            print(f"❌ Не удалось сохранить manifest.json: {me}")
            import traceback; traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"❌ Ошибка в блоке подготовки train_result.pkl: {e}")
        traceback.print_exc()

    print(f"✅ Tianshou DQN: модель сохранена в {model_path}")
    # Восстановим stdout/stderr и закроем лог
    try:
        import sys as _sys
        if _orig_stdout is not None:
            _sys.stdout = _orig_stdout
        if _orig_stderr is not None:
            _sys.stderr = _orig_stderr
        if _log_fp is not None:
            _log_fp.close()
    except Exception:
        pass

    # Проброс seed в окружения (VectorEnv / Dummy / списки)
    try:
        if isinstance(seed, int):
            # train envs
            try:
                if hasattr(train_envs, 'seed'):
                    train_envs.seed(seed)
                else:
                    for i, env in enumerate(train_envs):
                        try:
                            env.reset(seed=seed + i)
                        except Exception:
                            pass
            except Exception:
                pass
            # test envs (сдвиг, чтобы не совпадали)
            try:
                seed_test = seed + 1000
                if hasattr(test_envs, 'seed'):
                    test_envs.seed(seed_test)
                else:
                    for i, env in enumerate(test_envs):
                        try:
                            env.reset(seed=seed_test + i)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception as final_e:
        import traceback
        print(f"❌ Фатальное исключение после тренировки: {final_e}")
        traceback.print_exc()
        raise

    # Финальное сохранение train_result.pkl (повтор) — только если ранее не удалось
    if (not saved_train_result) and ('enriched_results' in locals()):
        print("—> Выполняю финальное сохранение train_result.pkl (повтор)")
        try:
            import pickle, traceback
            results_path = run_dir / 'train_result.pkl'
            with open(results_path, 'wb') as _f:
                pickle.dump(enriched_results, _f)
            saved_train_result = True
            print(f"💾 финальное train_result.pkl сохранён: {results_path}")
        except Exception as _err:
            traceback.print_exc()
    return str(run_dir)


def _sanitize_info_for_tianshou(info: dict) -> dict:
    """Возвращаем ТОЛЬКО фиксированный набор скалярных полей.
    Никаких списков/массивов, стабильный набор ключей на каждом шаге,
    чтобы избежать shape mismatch в буфере Tianshou.
    """
    # Набор ключей, который всегда присутствует
    sanitized = {
        'current_balance': 0.0,
        'current_price': 0.0,
        'total_profit': 0.0,
        'reward': 0.0,
        'action_counts_0': 0,
        'action_counts_1': 0,
        'action_counts_2': 0,
        'trades_count': 0,
        'market_state': 0,
        'mask_0': 1,
        'mask_1': 1,
        'mask_2': 1,
    }
    if not isinstance(info, dict):
        return sanitized

    # Безопасно извлекаем скаляры
    try:
        v = info.get('current_balance')
        if isinstance(v, (int, float)):
            sanitized['current_balance'] = float(v)
    except Exception:
        pass
    try:
        v = info.get('current_price')
        if isinstance(v, (int, float)):
            sanitized['current_price'] = float(v)
    except Exception:
        pass
    try:
        v = info.get('total_profit')
        if isinstance(v, (int, float)):
            sanitized['total_profit'] = float(v)
    except Exception:
        pass
    try:
        v = info.get('reward')
        if isinstance(v, (int, float)):
            sanitized['reward'] = float(v)
    except Exception:
        pass

    # action_counts -> фиксированные ключи
    try:
        ac = info.get('action_counts')
        if isinstance(ac, dict):
            sanitized['action_counts_0'] = int(ac.get(0, 0))
            sanitized['action_counts_1'] = int(ac.get(1, 0))
            sanitized['action_counts_2'] = int(ac.get(2, 0))
    except Exception:
        pass

    # trades_episode -> только количество
    try:
        te = info.get('trades_episode')
        if isinstance(te, (list, tuple)):
            sanitized['trades_count'] = int(len(te))
    except Exception:
        pass

    # market_state and mask fields (must be scalars)
    try:
        v = info.get('market_state')
        if isinstance(v, (int, float)):
            sanitized['market_state'] = int(v)
    except Exception:
        pass
    for k in ('mask_0', 'mask_1', 'mask_2'):
        try:
            v = info.get(k)
            if isinstance(v, (int, float)):
                sanitized[k] = int(v)
        except Exception:
            pass

    return sanitized
