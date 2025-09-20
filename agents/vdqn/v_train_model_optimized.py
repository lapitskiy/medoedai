import os
import re
import sys
import logging
import numpy as np
import torch
import wandb
import time
import psutil
from typing import Dict, List, Optional
import pickle
from pickle import HIGHEST_PROTOCOL
import hashlib
import platform
from datetime import datetime
import subprocess
from utils.adaptive_normalization import adaptive_normalizer

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.vdqn.dqnsolver import DQNSolver
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from envs.dqn_model.gym.gconfig import GymConfig
from agents.vdqn.hyperparameter.symbol_overrides import get_symbol_override


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
            env = MultiCryptoTradingEnv(dfs=dfs, cfg=cfg)
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
                for k, v in override['training_params'].items():
                    if hasattr(cfg, k):
                        try:
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
            env = CryptoTradingEnvOptimized(
                dfs=dfs,
                cfg=cfg,
                lookback_window=override.get('gym_config', {}).get('lookback_window', gym_cfg.lookback_window) if override else gym_cfg.lookback_window,
                indicators_config=indicators_config
            )

            # risk_management в env
            if override and 'risk_management' in override:
                rm = override['risk_management']
                for field_name, env_attr in [
                    ('STOP_LOSS_PCT', 'STOP_LOSS_PCT'),
                    ('TAKE_PROFIT_PCT', 'TAKE_PROFIT_PCT'),
                    ('min_hold_steps', 'min_hold_steps'),
                    ('volume_threshold', 'volume_threshold'),
                ]:
                    if field_name in rm:
                        try:
                            setattr(env, env_attr, rm[field_name])
                        except Exception:
                            pass
                try:
                    print(f"🔧 RISK OVERRIDE[{crypto_symbol}] | SL={getattr(env,'STOP_LOSS_PCT',None)} | TP={getattr(env,'TAKE_PROFIT_PCT',None)} | minHold={getattr(env,'min_hold_steps',None)} | volThr={getattr(env,'volume_threshold',None)}")
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
            gym_snapshot = {
                'symbol': getattr(env, 'symbol', None),
                'lookback_window': getattr(env, 'lookback_window', None),
                'indicators_config': getattr(env, 'indicators_config', None),
                'funding_features': {
                    'present_in_input_df': funding_present,
                    'included': bool(funding_present),
                },
                'risk_management': {
                    'STOP_LOSS_PCT': getattr(env, 'STOP_LOSS_PCT', None),
                    'TAKE_PROFIT_PCT': getattr(env, 'TAKE_PROFIT_PCT', None),
                    'min_hold_steps': getattr(env, 'min_hold_steps', None),
                    'volume_threshold': getattr(env, 'volume_threshold', None),
                    'base_stop_loss': getattr(env, 'base_stop_loss', None),
                    'base_take_profit': getattr(env, 'base_take_profit', None),
                    'base_min_hold': getattr(env, 'base_min_hold', None),
                },
                'position_sizing': {
                    'base_position_fraction': getattr(env, 'base_position_fraction', None),
                    'position_fraction': getattr(env, 'position_fraction', None),
                    'position_confidence_threshold': getattr(env, 'position_confidence_threshold', None),
                },
                'observation_space_shape': getattr(env, 'observation_space_shape', None),
                'step_minutes': getattr(env.cfg, 'step_minutes', 5) if hasattr(env, 'cfg') else 5,
            }
        except Exception:
            gym_snapshot = {}

        # --- Снимок адаптивной нормализации (по символам) ---
        adaptive_snapshot = {}
        try:
            if is_multi_crypto:
                per_symbol = {}
                for sym, data in dfs.items():
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
                df5s = dfs.get('df_5min') if isinstance(dfs, dict) else None
                sym = crypto_symbol
                if df5s is not None and isinstance(sym, str):
                    base_profile = adaptive_normalizer.get_crypto_profile(sym)
                    market = adaptive_normalizer.analyze_market_conditions(df5s)
                    adapted = adaptive_normalizer.adapt_parameters(sym, df5s)
                    adaptive_snapshot = {
                        'symbol': sym,
                        'base_profile': base_profile,
                        'market_conditions': market,
                        'adapted_params': adapted,
                    }
        except Exception:
            adaptive_snapshot = {}
        
        # Проверяем, что окружение правильно инициализировано
        if not hasattr(env, 'observation_space_shape'):
            # Попробуем вычислить размер состояния из observation_space
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
                env.observation_space_shape = env.observation_space.shape[0]
                print(f"⚠️ Вычислен размер состояния из observation_space: {env.observation_space_shape}")
            else:
                raise ValueError("Окружение не имеет observation_space_shape и не может быть вычислен")
        
        # Получаем символ криптовалюты для логирования
        if is_multi_crypto:
            crypto_symbol = "МУЛЬТИВАЛЮТА"  # Для мультивалютного окружения
            print(f"✅ Мультивалютное окружение создано, размер состояния: {env.observation_space_shape}")
        else:
            crypto_symbol = getattr(env, 'symbol', 'UNKNOWN')
            print(f"✅ Окружение создано для {crypto_symbol}, размер состояния: {env.observation_space_shape}")

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

        result_dir = os.path.join("result")
        os.makedirs(result_dir, exist_ok=True)
        symbol_code = _symbol_code(crypto_symbol)
        # Короткий UUID для версионирования (по умолчанию)
        import uuid
        short_id = str(uuid.uuid4())[:4].lower()

        # Структурированный каталог результата: result/<SYMBOL>/runs/<run_id>/
        # Папка символа без суффикса (TON, BTC, BNB...) в верхнем регистре
        symbol_dir_name = _symbol_code(crypto_symbol).upper() if crypto_symbol else "UNKNOWN"
        # Короткий run_id (4 символа) если не передан
        this_run_id = run_id or str(uuid.uuid4())[:4]
        this_root_id = root_id or this_run_id
        run_dir = os.path.join("result", symbol_dir_name, "runs", this_run_id)
        os.makedirs(run_dir, exist_ok=True)

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
            observation_space=env.observation_space_shape,
            action_space=env.action_space.n
        )
        # Если указан путь загрузки существующей модели/буфера — загружаем сначала
        if load_model_path and isinstance(load_model_path, str):
            try:
                dqn_solver.cfg.model_path = load_model_path
            except Exception:
                pass
        if load_buffer_path and isinstance(load_buffer_path, str):
            try:
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
        
        # Загружаем МОДЕЛЬ если есть (либо из указанных путей, либо по дефолту)
        dqn_solver.load_model()
        # Загружаем replay buffer, если был передан путь и файл существует
        try:
            if load_buffer_path and isinstance(load_buffer_path, str) and os.path.exists(load_buffer_path):
                print(f"🧠 Загружаю replay buffer из {load_buffer_path}")
                dqn_solver.load_state()
            else:
                print("ℹ️ Replay buffer не передан или файл отсутствует — начнем с пустой памяти")
        except Exception as _e:
            print(f"⚠️ Не удалось загрузить replay buffer: {_e}")

        # После загрузки переназначаем пути сохранения на НОВЫЕ в result/<symbol>_<id>
        # Обновляем пути сохранения на структурированные независимо от наличия внешней cfg
        try:
            dqn_solver.cfg.model_path = new_model_path
            dqn_solver.cfg.buffer_path = new_buffer_path
        except Exception:
            pass
        try:
            if cfg is not None:
                cfg.model_path = dqn_solver.cfg.model_path
                cfg.buffer_path = dqn_solver.cfg.buffer_path
        except Exception:
            pass
        
        # Переменные для отслеживания прогресса
        all_trades = []
        episode_winrates = []
        best_winrate = 0.0
        best_episode_idx = -1
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
            
            # Эпизод
            step_count = 0
            failed_train_attempts = 0
            while True:
                step_count += 1
                # Показываем прогресс каждые 100 шагов для ускорения
                if step_count % 10000 == 0:
                    print(f"    🔄 Step {step_count} в эпизоде {episode}")
                
                env.epsilon = dqn_solver.epsilon
                
                action = dqn_solver.act(state)
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
                        else:
                            failed_train_attempts += 1
                            break
                    # Обновляем target network по расписанию
                    if global_step % target_update_freq == 0:
                        dqn_solver.update_target_model()

                if terminal:
                    break
            
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
            
            # ИСПРАВЛЕНИЕ: Получаем сделки из env.all_trades вместо env.trades
            if hasattr(env, 'all_trades') and env.all_trades:
                episode_trades = env.all_trades
            else:
                # Fallback: используем env.trades
                episode_trades = env.trades if hasattr(env, 'trades') and env.trades else []
            
            # ИСПРАВЛЕНИЕ: Инициализируем episode_winrate по умолчанию
            episode_winrate = 0.0
            
            if hasattr(env, 'all_trades') and env.all_trades:
                # Используем все сделки из окружения для расчета winrate
                all_profitable = [t for t in env.all_trades if t.get('roi', 0) > 0]
                episode_winrate = len(all_profitable) / len(env.all_trades) if env.all_trades else 0
                episode_winrates.append(episode_winrate)
                
                # Детальная статистика эпизода
                episode_stats = dqn_solver.print_trade_stats(env.all_trades, failed_attempts=failed_train_attempts)
                
                # Добавляем сделки в общий список если их там нет
                if len(all_trades) < len(env.all_trades):
                    all_trades.extend(env.all_trades[len(all_trades):])
                    
            elif episode_trades:
                # Fallback: используем env.trades
                all_trades.extend(episode_trades)
                
                # Вычисляем winrate для эпизода
                profitable_trades = [t for t in episode_trades if t.get('roi', 0) > 0]
                episode_winrate = len(profitable_trades) / len(episode_trades) if episode_trades else 0
                episode_winrates.append(episode_winrate)
                
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
                    episode_stats = dqn_solver.print_trade_stats(recent_trades, failed_attempts=failed_train_attempts)
                else:
                    # Нет сделок вовсе — выводим агрегатную строку статистики с Failed train
                    episode_winrate = 0.0
                    episode_winrates.append(episode_winrate)
                    episode_stats = dqn_solver.print_trade_stats([], failed_attempts=failed_train_attempts)
                
                # Объединяем всю статистику эпизода в одну строку
                action_stats = ""
                if hasattr(env, 'action_counts'):
                    action_stats = f" | HOLD={env.action_counts.get(0, 0)}, BUY={env.action_counts.get(1, 0)}, SELL={env.action_counts.get(2, 0)}"
                
                # Добавляем информацию о времени выполнения
                time_stats = ""
                if hasattr(env, 'episode_start_time') and env.episode_start_time is not None:
                    episode_duration = time.time() - env.episode_start_time
                    steps_per_second = env.episode_step_count / episode_duration if episode_duration > 0 else 0
                    time_stats = f" | {episode_duration:.2f}с, {env.episode_step_count} шагов, {steps_per_second:.1f} шаг/с"
                
                print(f"  🏁 Эпизод {episode} для {current_crypto} завершен | reward={episode_reward:.4f}{action_stats}{time_stats} | {episode_stats}")
                
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
                if hasattr(env, 'action_counts') and isinstance(env.action_counts, dict):
                    action_counts_total[0] = action_counts_total.get(0, 0) + int(env.action_counts.get(0, 0) or 0)
                    action_counts_total[1] = action_counts_total.get(1, 0) + int(env.action_counts.get(1, 0) or 0)
                    action_counts_total[2] = action_counts_total.get(2, 0) + int(env.action_counts.get(2, 0) or 0)
                # Попытки покупок и причины отказов
                buy_attempts_total += int(getattr(env, 'buy_attempts', 0) or 0)
                buy_rejected_vol_total += int(getattr(env, 'buy_rejected_vol', 0) or 0)
                buy_rejected_roi_total += int(getattr(env, 'buy_rejected_roi', 0) or 0)
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
                # Сохраняем последний снапшот last_model.pth
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
        print(f"  • Время на эпизод: {total_training_time/episode:.2f} секунд")
        print(f"  • Эпизодов в минуту: {episode/(total_training_time/60):.1f}")
        
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
        
        if hasattr(cfg, 'use_wandb') and cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        # Финальное сохранение модели и replay buffer
        print("\n💾 Финальное сохранение модели и replay buffer")
        dqn_solver.save()
        # Финальный last_model снапшот
        try:
            import shutil as _sh
            if os.path.exists(cfg.model_path):
                _sh.copy2(cfg.model_path, os.path.join(run_dir, 'last_model.pth'))
        except Exception:
            pass
        
        # Сохраняем детальные результаты обучения
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

        training_results = {
            'episodes': episodes,  # Планируемое количество эпизодов
            'actual_episodes': episode,  # Реальное количество завершенных эпизодов (текущий эпизод)
            'total_training_time': total_training_time,
            'episode_winrates': episode_winrates,
            'all_trades': all_trades,
            'bad_trades': bad_trades_list,
            'bad_trades_count': bad_trades_count,
            'bad_trades_percentage': bad_trades_percentage,
            'best_winrate': best_winrate,
            'final_stats': stats_all,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': cfg.model_path,
            'buffer_path': cfg.buffer_path,
            'symbol': training_name,
            'model_id': short_id,
            'early_stopping_triggered': episode < episodes,  # True если early stopping сработал
            # --- Новые агрегаты для анализа поведения ---
            'action_counts_total': action_counts_total,
            'buy_attempts_total': buy_attempts_total,
            'buy_rejected_vol_total': buy_rejected_vol_total,
            'buy_rejected_roi_total': buy_rejected_roi_total,
            'buy_accept_rate': ( (action_counts_total.get(1, 0) or 0) / float(buy_attempts_total) ) if buy_attempts_total > 0 else 0.0,
            'episodes_with_trade_count': episodes_with_trade_count,
            'episodes_with_trade_ratio': (episodes_with_trade_count / float(episodes)) if episodes > 0 else 0.0,
            'avg_minutes_between_buys': ( (total_steps_processed * 5.0) / float(action_counts_total.get(1, 0) or 1) ) if (action_counts_total.get(1, 0) or 0) > 0 else None,
            'total_steps_processed': total_steps_processed,
        }
        
        # Создаем папку если не существует (используем структурированный run_dir)
        try:
            run_dir  # noqa: F401
        except NameError:
            # Папка символа без суффикса (TON, BTC, BNB...) в верхнем регистре
            symbol_dir_name = _symbol_code(training_name).upper() if training_name else "UNKNOWN"
            # Короткий run_id (4 символа) если не передан
            this_run_id = run_id or str(__import__('uuid').uuid4())[:4]
            this_root_id = root_id or this_run_id
            run_dir = os.path.join("result", symbol_dir_name, "runs", this_run_id)
            os.makedirs(run_dir, exist_ok=True)

        # Сохраняем результаты в файле в run_dir
        results_file = os.path.join(run_dir, 'train_result.pkl')

        # Метаданные окружения и запуска
        try:
            git_commit = None
            try:
                git_commit = subprocess.check_output(['git','rev-parse','--short','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                pass
            # Сохраняем seed, если он был передан аргументом функции
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
        except Exception:
            train_metadata = {}

        # Снимок конфигурации и архитектуры
        try:
            cfg_snapshot = _safe_cfg_snapshot(cfg)
        except Exception:
            cfg_snapshot = {}
        try:
            arch_main = _architecture_summary(dqn_solver.model)
            arch_target = _architecture_summary(dqn_solver.target_model)
        except Exception:
            arch_main, arch_target = {}, {}

        # Информация о весах (пути и хэши)
        weights_info = {
            'model_path': cfg.model_path,
            'buffer_path': cfg.buffer_path,
            'model_sha256': _sha256_of_file(cfg.model_path) if cfg and getattr(cfg, 'model_path', None) and os.path.exists(cfg.model_path) else None,
            'buffer_sha256': _sha256_of_file(cfg.buffer_path) if cfg and getattr(cfg, 'buffer_path', None) and os.path.exists(cfg.buffer_path) else None,
        }

        # Объединяем
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
        }

        with open(results_file, 'wb') as f:
            pickle.dump(enriched_results, f, protocol=HIGHEST_PROTOCOL)
        
        print(f"📊 Детальные результаты сохранены в: {results_file}")

        # === Структурированное сохранение в result/<SYMBOL>/runs/<run_id>/ ===
        try:
            # run_dir уже создан выше

            # Копируем артефакты в папку запуска с фиксированными именами
            try:
                import shutil as _sh
                # Модель
                if cfg and getattr(cfg, 'model_path', None) and os.path.exists(cfg.model_path):
                    _dst_m = os.path.join(run_dir, 'model.pth')
                    if os.path.abspath(cfg.model_path) != os.path.abspath(_dst_m):
                        _sh.copy2(cfg.model_path, _dst_m)
                # Буфер
                if cfg and getattr(cfg, 'buffer_path', None) and os.path.exists(cfg.buffer_path):
                    _dst_b = os.path.join(run_dir, 'replay.pkl')
                    if os.path.abspath(cfg.buffer_path) != os.path.abspath(_dst_b):
                        _sh.copy2(cfg.buffer_path, _dst_b)
                # Результаты
                if os.path.exists(results_file):
                    _dst_r = os.path.join(run_dir, 'train_result.pkl')
                    if os.path.abspath(results_file) != os.path.abspath(_dst_r):
                        _sh.copy2(results_file, _dst_r)
            except Exception as _copy_err:
                print(f"⚠️ Не удалось скопировать артефакты в {run_dir}: {_copy_err}")

            # Пишем манифест (минимум метаданных; подробности уже в train_result.pkl)
            manifest = {
                'run_id': this_run_id,
                'parent_run_id': parent_run_id,
                'root_id': this_root_id,
                'symbol': symbol_dir_name,
                'seed': int(seed) if isinstance(seed, int) else None,
                'episodes_start': 0 if not load_model_path else None,
                'episodes_end': int(training_results.get('actual_episodes') or episodes),
                'episodes_added': int(training_results.get('actual_episodes') or episodes),
                'episodes_last': int(training_results.get('actual_episodes') or episodes),
                'episodes_best': int(best_episode_idx) if best_episode_idx is not None and best_episode_idx >= 0 else None,
                'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'artifacts': {
                    'model': 'model.pth',
                    'replay': 'replay.pkl' if (dqn_solver and getattr(dqn_solver, 'cfg', None) and getattr(dqn_solver.cfg, 'buffer_path', None)) else None,
                    'result': 'train_result.pkl',
                    'best_model': 'best_model.pth' if os.path.exists(os.path.join(run_dir, 'best_model.pth')) else None,
                    'last_model': 'last_model.pth' if os.path.exists(os.path.join(run_dir, 'last_model.pth')) else None
                },
                'best_metrics': {
                    'winrate': float(best_winrate) if isinstance(best_winrate, (int, float)) else None
                }
            }
            try:
                import json as _json
                with open(os.path.join(run_dir, 'manifest.json'), 'w', encoding='utf-8') as mf:
                    _json.dump(manifest, mf, ensure_ascii=False, indent=2)
            except Exception as _mf_err:
                print(f"⚠️ Не удалось записать manifest.json: {_mf_err}")
        except Exception as _struct_err:
            print(f"⚠️ Ошибка структурированного сохранения: {_struct_err}")
        
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
