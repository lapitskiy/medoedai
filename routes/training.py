from flask import Blueprint, jsonify, request, redirect, url_for, current_app as app
from celery.result import AsyncResult
from tasks.celery_tasks import celery, train_dqn_multi_crypto, train_dqn_symbol
from tasks.sac_tasks import train_sac_symbol
from tasks.xgb_tasks import train_xgb_symbol, train_xgb_grid, train_xgb_grid_entry_exit, train_xgb_grid_full
from utils.redis_utils import get_redis_client
import os
import logging
import itertools
from pathlib import Path as _Path

# Инициализируем Redis клиент
redis_client = get_redis_client()

# Создаем Blueprint для маршрутов обучения
training_bp = Blueprint('training', __name__)

# Настройка логирования
logger = logging.getLogger(__name__)

@training_bp.route('/clear_train_lock', methods=['POST'])
def clear_train_lock_route():
    """Сбрасывает per-symbol блокировку обучения в Redis (НЕ останавливает реально running Celery task).

    Важно: кроме `celery:train:task:<SYMBOL>` чистит также бизнес-ключи дедупликации:
    `celery:train:{queued,running,finished}:<SYMBOL>|...`
    """
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or request.form.get('symbol') or '').strip().upper()
        task_id = (data.get('task_id') or request.form.get('task_id') or '').strip()
        if not symbol:
            return jsonify({"success": False, "error": "symbol required"}), 400

        running_key = f"celery:train:task:{symbol}"
        try:
            redis_client.delete(running_key)
        except Exception:
            pass

        # Чистим дедуп-ключи для этого символа (queued/running/finished)
        deleted = []
        try:
            pattern = f"celery:train:*:{symbol}|*"
            for k in redis_client.scan_iter(match=pattern, count=1000):
                try:
                    redis_client.delete(k)
                    deleted.append(k)
                except Exception:
                    continue
        except Exception:
            pass

        # Опционально: убрать task_id из UI списка, если передали
        if task_id:
            try:
                redis_client.lrem("ui:tasks", 0, task_id)
            except Exception:
                pass

        return jsonify({"success": True, "symbol": symbol, "cleared": True, "deleted": len(deleted)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@training_bp.route('/train_dqn_multi_crypto', methods=['POST'])
def train_multi_crypto():
    """Запускает мультивалютное обучение DQN"""
    data = request.get_json(silent=True) or {}
    episodes_str = data.get('episodes') or request.form.get('episodes')
    episodes = None
    try:
        if episodes_str is not None and str(episodes_str).strip() != '':
            episodes = int(episodes_str)
    except Exception:
        episodes = None
    episode_length_str = data.get('episode_length') or request.form.get('episode_length')
    episode_length = None
    try:
        if episode_length_str is not None and str(episode_length_str).strip() != '':
            episode_length = int(episode_length_str)
    except Exception:
        episode_length = None

    seed_raw = (data.get('seed') or request.form.get('seed') or '').strip()
    seed = None
    try:
        if seed_raw != '':
            seed = int(seed_raw)
    except Exception:
        seed = None
    task = train_dqn_multi_crypto.apply_async(kwargs={'episodes': episodes, 'episode_length': episode_length, 'seed': seed}, queue="train")
    try:
        wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    except Exception:
        wants_json = False
    if wants_json:
        return jsonify({"success": True, "task_id": task.id})
    return redirect(url_for("index"))

@training_bp.route('/train_dqn_symbol', methods=['POST'])
def train_dqn_symbol_route():
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol') or request.form.get('symbol') or 'BTCUSDT'
    # Эпизоды: из формы/JSON, fallback None (чтоб Celery взял из ENV)
    episodes_str = data.get('episodes') or request.form.get('episodes')
    episodes = None
    try:
        if episodes_str is not None and str(episodes_str).strip() != '':
            episodes = int(episodes_str)
    except Exception:
        episodes = None
    # Seed: из формы/JSON; если пусто — сгенерируем случайный на бэкенде
    seed_raw = (data.get('seed') or request.form.get('seed') or '').strip()
    seed = None
    try:
        if seed_raw != '':
            seed = int(seed_raw)
    except Exception:
        seed = None
    episode_length_str = data.get('episode_length') or request.form.get('episode_length')
    episode_length = None
    try:
        if episode_length_str is not None and str(episode_length_str).strip() != '':
            episode_length = int(episode_length_str)
    except Exception:
        episode_length = None

    cutoff_days_raw = data.get('cutoff_days') or request.form.get('cutoff_days') or request.args.get('cutoff_days') or 90
    cutoff_days = 90
    try:
        cutoff_days = int(cutoff_days_raw)
    except Exception:
        cutoff_days = 90
    if cutoff_days < 0:
        cutoff_days = 0

    # Новые параметры: выбор энкодера и режим обучения энкодера
    encoder_id = (data.get('encoder_id') or request.form.get('encoder_id') or '').strip()
    train_encoder = data.get('train_encoder')
    if isinstance(train_encoder, str):
        train_encoder = train_encoder.strip().lower() in ('1','true','yes','on')
    train_encoder = bool(train_encoder) if train_encoder is not None else False

    # Проверка активной задачи per-symbol в Redis + Celery
    running_key = f"celery:train:task:{symbol.upper()}"
    try:
        existing_task_id = redis_client.get(running_key)
        if existing_task_id:
            try:
                existing_task_id = existing_task_id.decode('utf-8')
            except Exception:
                existing_task_id = str(existing_task_id)
            ar = AsyncResult(existing_task_id, app=celery)
            if ar.state in ("PENDING", "STARTED", "RETRY", "IN_PROGRESS"):
                logger.info(f"train for {symbol} already running: {existing_task_id} state={ar.state}")
                # Если это fetch — вернём JSON без редиректа
                wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
                if wants_json:
                    return jsonify({
                        "success": False,
                        "error": f"training for {symbol} already running",
                        "task_id": existing_task_id,
                        "state": ar.state
                    })
                return redirect(url_for("index"))
    except Exception as _e:
        logger.error(f"Не удалось проверить активную задачу для {symbol}: {_e}")

    # Временно отправляем в общую очередь 'train' (слушается базовым воркером)
    # Если seed не указан — сгенерируем случайный на стороне web и передадим в Celery
    if seed is None:
        import random as _rnd
        seed = _rnd.randint(1, 2**31 - 1)
        logger.info(f"[train_dqn_symbol] generated random seed={seed}")

    # Направление обучения (long|short)
    direction = (data.get('direction') or request.form.get('direction') or 'long').strip().lower()
    # Переключаемся на Tianshou по умолчанию (engine='ts')
    engine = (data.get('engine') or request.args.get('engine') or 'ts').lower()
    # Optional: env_overrides (grid etc)
    env_overrides = data.get('env_overrides') if isinstance(data, dict) else None
    # Optional: cfg_overrides (DQN hyperparams overrides)
    cfg_overrides = data.get('cfg_overrides') if isinstance(data, dict) else None
    if cfg_overrides is not None and not isinstance(cfg_overrides, dict):
        return jsonify({"success": False, "error": "cfg_overrides must be object"}), 400
    # force=1 отключает дедупликацию queued/running/finished для этого запроса
    force_raw = (data.get('force') if isinstance(data, dict) else None)
    if force_raw is None:
        force_raw = request.args.get('force') or request.form.get('force')
    force = str(force_raw).strip().lower() in ('1', 'true', 'yes', 'on')
    # Дедупликация на этапе постановки: не ставим, если такой же запуск уже в очереди/был недавно
    try:
        def _fmt_f(v):
            try:
                if v is None:
                    return "na"
                if isinstance(v, bool):
                    return "1" if v else "0"
                if isinstance(v, int):
                    return str(v)
                x = float(v)
                s = f"{x:.6f}".rstrip('0').rstrip('.')
                return s if s != '' else '0'
            except Exception:
                return str(v)

        def _rm_tag(overrides: dict | None) -> str:
            try:
                if not isinstance(overrides, dict):
                    return "rm:-"
                rm = overrides.get("risk_management")
                if not isinstance(rm, dict):
                    return "rm:-"
                sl = _fmt_f(rm.get("STOP_LOSS_PCT"))
                tp = _fmt_f(rm.get("TAKE_PROFIT_PCT"))
                mh = _fmt_f(rm.get("min_hold_steps"))
                vt = _fmt_f(rm.get("volume_threshold"))
                return f"rm:{sl}:{tp}:{mh}:{vt}"
            except Exception:
                return "rm:-"

        def _cfg_tag(overrides: dict | None) -> str:
            try:
                if not isinstance(overrides, dict) or not overrides:
                    return "cfg:-"
                # Keep it short; actual parsing/validation in Celery worker
                bs = overrides.get("batch_size")
                ms = overrides.get("memory_size")
                hs = overrides.get("hidden_sizes")
                dec = overrides.get("eps_decay_steps")
                lr = overrides.get("lr")
                tr = overrides.get("train_repeats")
                return f"cfg:bs={_fmt_f(bs)}:ms={_fmt_f(ms)}:hs={str(hs)}:dec={_fmt_f(dec)}:lr={_fmt_f(lr)}:tr={_fmt_f(tr)}"
            except Exception:
                return "cfg:-"

        _engine = (engine or '').lower()
        _enc = (encoder_id or '-')
        _train_enc = 1 if train_encoder else 0
        _eps = episodes if (episodes is not None) else 'env'
        _ep_len = episode_length if (episode_length is not None) else 'cfg'
        _rm = _rm_tag(env_overrides)
        _cfg = _cfg_tag(cfg_overrides)
        _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|cut={cutoff_days}|{(direction or 'long')}|{_rm}|{_cfg}"
        _queued_key = f"celery:train:queued:{_biz_key}"
        _running_key_biz = f"celery:train:running:{_biz_key}"
        _finished_key = f"celery:train:finished:{_biz_key}"
        if not force and (redis_client.get(_running_key_biz) or redis_client.get(_queued_key) or redis_client.get(_finished_key)):
            wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
            if wants_json:
                return jsonify({
                    "success": False,
                    "error": f"training duplicate for {symbol} skipped",
                    "duplicate": True,
                })
            return redirect(url_for("index"))
    except Exception:
        pass

    task = train_dqn_symbol.apply_async(
        kwargs={
            'symbol': symbol,
            'episodes': episodes,
            'seed': seed,
            'episode_length': episode_length,
            'engine': engine,
            'encoder_id': encoder_id,
            'train_encoder': train_encoder,
            'direction': direction,
            'env_overrides': env_overrides,
            'cfg_overrides': cfg_overrides,
            'limit_candles': 100000,
            'cutoff_days': cutoff_days,
        },
        queue="train",
    )
    logger.info(f"/train_dqn_symbol queued symbol={symbol} queue=train task_id={task.id}")
    # Сохраняем task_id для отображения на главной и отметку per-symbol
    try:
        redis_client.lrem("ui:tasks", 0, task.id)  # убираем дубликаты
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)     # ограничиваем список
        redis_client.setex(running_key, 24 * 3600, task.id)
        # Отмечаем бизнес-‘queued’ на короткое время, чтобы двойные клики UI не ставили второй запуск
        try:
            _ttl = int(os.environ.get('TRAIN_QUEUED_TTL', '300'))  # 5 минут по умолчанию
        except Exception:
            _ttl = 300
        try:
            _engine = (engine or '').lower()
            _enc = (encoder_id or '-')
            _train_enc = 1 if train_encoder else 0
            _eps = episodes if (episodes is not None) else 'env'
            _ep_len = episode_length if (episode_length is not None) else 'cfg'
            _rm = _rm_tag(env_overrides)
            _cfg = _cfg_tag(cfg_overrides)
            _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|cut={cutoff_days}|{(direction or 'long')}|{_rm}|{_cfg}"
            redis_client.setex(f"celery:train:queued:{_biz_key}", max(30, _ttl), task.id)
        except Exception:
            pass
    except Exception as _e:
        logger.error(f"/train_dqn_symbol: не удалось записать ui:tasks: {_e}")
    # Ответ для fetch/XHR — JSON; для форм — редирект
    wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    if wants_json:
        return jsonify({
            "success": True,
            "task_id": task.id,
            "symbol": symbol,
            "episodes": episodes,
            "seed": seed,
            "env_overrides": env_overrides,
            "cfg_overrides": cfg_overrides,
            "cutoff_days": cutoff_days,
        })
    return redirect(url_for("index"))


@training_bp.post('/train_dqn_grid')
def train_dqn_grid_route():
    """Ставит в очередь несколько DQN-обучений (grid) с env_overrides поверх GLOBAL_OVERRIDES + optional cfg_overrides."""
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        if not symbol:
            return jsonify({"success": False, "error": "symbol required"}), 400

        # base params
        try:
            episodes = int(data.get('episodes'))
        except Exception:
            return jsonify({"success": False, "error": "episodes required (int)"}), 400
        try:
            episode_length = int(data.get('episode_length'))
        except Exception:
            return jsonify({"success": False, "error": "episode_length required (int)"}), 400
        direction = str(data.get('direction') or 'long').strip().lower()
        engine = str(data.get('engine') or 'ts').strip().lower()
        encoder_id = str(data.get('encoder_id') or '').strip()
        train_encoder = data.get('train_encoder')
        if isinstance(train_encoder, str):
            train_encoder = train_encoder.strip().lower() in ('1', 'true', 'yes', 'on')
        train_encoder = bool(train_encoder) if train_encoder is not None else False

        cutoff_days = data.get('cutoff_days', 90)
        try:
            cutoff_days = int(cutoff_days)
        except Exception:
            cutoff_days = 90
        if cutoff_days < 0:
            cutoff_days = 0

        seed_base_raw = (data.get('seed') or '').strip() if isinstance(data.get('seed'), str) else data.get('seed')
        seed_base = None
        if seed_base_raw not in (None, ''):
            try:
                seed_base = int(seed_base_raw)
            except Exception:
                return jsonify({"success": False, "error": "seed must be int"}), 400
        if seed_base is None:
            import random as _rnd
            seed_base = _rnd.randint(1, 2**31 - 1)

        grid = data.get('grid') or {}
        if not isinstance(grid, dict):
            return jsonify({"success": False, "error": "grid must be object"}), 400

        def _int(v, name):
            try:
                return int(v)
            except Exception:
                raise ValueError(f"{name} must be int")

        def _irange(name: str, a: int, b: int, step: int, max_n=2000) -> list[int]:
            if step <= 0:
                raise ValueError(f"{name}: step must be > 0 (got {step})")
            if b < a:
                raise ValueError(f"{name}: to must be >= from (from={a}, to={b})")
            out = []
            x = int(a)
            for _ in range(max_n):
                if x > int(b):
                    break
                out.append(int(x))
                x += int(step)
            if not out:
                raise ValueError(f"{name}: empty range (from={a}, to={b}, step={step})")
            return out

        def _parse_csv_ints(v, name: str) -> list[int]:
            if v is None:
                raise ValueError(f"{name} is required")
            if isinstance(v, (int, float)):
                return [int(v)]
            if not isinstance(v, str):
                raise ValueError(f"{name} must be CSV string")
            parts = [p.strip() for p in v.replace(';', ',').split(',') if p.strip()]
            if not parts:
                raise ValueError(f"{name} must be non-empty CSV")
            out = []
            for p in parts:
                try:
                    out.append(int(p))
                except Exception:
                    raise ValueError(f"{name} must be CSV of ints")
            return out

        def _float(v, name):
            try:
                return float(v)
            except Exception:
                raise ValueError(f"{name} must be float")

        def _frange(name: str, a, b, step, max_n=200):
            if step <= 0:
                raise ValueError(f"{name}: step must be > 0 (got {step})")
            if b < a:
                raise ValueError(f"{name}: to must be >= from (from={a}, to={b})")
            out = []
            x = a
            # inclusive with tolerance
            for _ in range(max_n):
                if x > b + 1e-12:
                    break
                out.append(float(round(x, 12)))
                x += step
            if not out:
                raise ValueError(f"{name}: empty range (from={a}, to={b}, step={step})")
            return out

        try:
            # ATR-only режим: SL/TP фиксированы как аварийные, варьируем ATR-множители
            atr_sl_from = _float(grid.get('atr_sl_from', '1.0'), 'atr_sl_from')
            atr_sl_to = _float(grid.get('atr_sl_to', '2.0'), 'atr_sl_to')
            atr_sl_step = _float(grid.get('atr_sl_step', '0.5'), 'atr_sl_step')
            # BUY filters knobs (optional, fixed values; empty => None)
            def _opt_float(key: str):
                v = grid.get(key, None)
                if v is None:
                    return None
                if isinstance(v, str) and v.strip() == '':
                    return None
                return _float(v, key)
            buy_roi_thr = _opt_float('buy_roi_thr')          # e.g. -0.01
            buy_trend_thr = _opt_float('buy_trend_thr')      # e.g. 0.0
            buy_volat_thr = _opt_float('buy_volat_thr')      # e.g. 0.003
            strict_floor = _opt_float('buy_strictness_floor')  # e.g. 0.3
            entry_gate = _opt_float('entry_confidence_gate') # e.g. 0.6
            mh_from = _int(grid.get('min_hold_from'), 'min_hold_from')
            mh_to = _int(grid.get('min_hold_to'), 'min_hold_to')
            mh_step = _int(grid.get('min_hold_step'), 'min_hold_step')
            vt_from = _float(grid.get('vol_from'), 'vol_from')
            vt_to = _float(grid.get('vol_to'), 'vol_to')
            vt_step = _float(grid.get('vol_step'), 'vol_step')
            trail_from = _float(grid.get('trail_from', '1.5'), 'trail_from')
            trail_to = _float(grid.get('trail_to', '1.5'), 'trail_to')
            trail_step = _float(grid.get('trail_step', '0.5'), 'trail_step')
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        # DQN hyperparams ranges (cfg_overrides) — required keys
        try:
            bs_from = _int(grid.get('batch_from'), 'batch_from')
            bs_to = _int(grid.get('batch_to'), 'batch_to')
            bs_step = _int(grid.get('batch_step'), 'batch_step')
            mem_from = _int(grid.get('mem_from'), 'mem_from')
            mem_to = _int(grid.get('mem_to'), 'mem_to')
            mem_step = _int(grid.get('mem_step'), 'mem_step')
            dec_from = _int(grid.get('eps_decay_from'), 'eps_decay_from')
            dec_to = _int(grid.get('eps_decay_to'), 'eps_decay_to')
            dec_step = _int(grid.get('eps_decay_step'), 'eps_decay_step')
            h_from = _parse_csv_ints(grid.get('hidden_from'), 'hidden_from')
            h_to = _parse_csv_ints(grid.get('hidden_to'), 'hidden_to')
            h_step = _parse_csv_ints(grid.get('hidden_step'), 'hidden_step')
            if not (len(h_from) == len(h_to) == len(h_step)):
                return jsonify({"success": False, "error": "hidden_from/hidden_to/hidden_step must have same length"}), 400
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        try:
            batch_sizes = _irange("batch_size", bs_from, bs_to, bs_step, max_n=200)
            memory_sizes = _irange("memory_size", mem_from, mem_to, mem_step, max_n=400)
            eps_decay_steps_list = _irange("eps_decay_steps", dec_from, dec_to, dec_step, max_n=400)
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        # hidden sizes per-layer ranges -> cartesian product -> tuples
        try:
            layer_lists = []
            for i in range(len(h_from)):
                layer_lists.append(_irange(f"hidden[{i}]", int(h_from[i]), int(h_to[i]), int(h_step[i]), max_n=50))
            hidden_sizes_list = [tuple(int(x) for x in combo) for combo in itertools.product(*layer_lists)]
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        try:
            max_models = int(data.get('max_models'))
        except Exception:
            max_models = 10
        if max_models <= 0 or max_models > 500:
            return jsonify({"success": False, "error": "max_models must be 1..500"}), 400

        atr_sl_list = _frange("atr_sl_mult", atr_sl_from, atr_sl_to, atr_sl_step)
        mh_list = list(range(mh_from, mh_to + 1, mh_step)) if mh_step > 0 else []
        if not mh_list:
            return jsonify({"success": False, "error": "min_hold range invalid"}), 400
        vt_list = _frange("volume_threshold", vt_from, vt_to, vt_step)
        trail_list = _frange("atr_trail_mult", trail_from, trail_to, trail_step)

        # cartesian size
        total = len(atr_sl_list) * len(mh_list) * len(vt_list) * len(trail_list) * len(batch_sizes) * len(memory_sizes) * len(hidden_sizes_list) * len(eps_decay_steps_list)
        if total <= 0:
            return jsonify({"success": False, "error": "empty grid"}), 400
        if total > 500:
            return jsonify({"success": False, "error": f"grid too large: {total} (>500)"}), 400

        # ATR-only: SL/TP фиксированы как аварийные (широкие), выход через ATR trailing/SL
        EMERGENCY_SL = -0.50
        EMERGENCY_TP = 0.15

        # select subset deterministically if needed
        combos = []
        for asl in atr_sl_list:
            for mh in mh_list:
                for vt in vt_list:
                  for tr in trail_list:
                    for bs in batch_sizes:
                        for ms in memory_sizes:
                            for hs in hidden_sizes_list:
                                for dec in eps_decay_steps_list:
                                    rm = {"STOP_LOSS_PCT": EMERGENCY_SL, "TAKE_PROFIT_PCT": EMERGENCY_TP, "min_hold_steps": mh, "volume_threshold": vt, "atr_trail_mult": tr, "atr_sl_mult": asl}
                                    # Optional buy-filter overrides
                                    if buy_roi_thr is not None:
                                        rm["buy_roi_thr"] = float(buy_roi_thr)
                                    if buy_trend_thr is not None:
                                        rm["buy_trend_thr"] = float(buy_trend_thr)
                                    if buy_volat_thr is not None:
                                        rm["buy_volat_thr"] = float(buy_volat_thr)
                                    if strict_floor is not None:
                                        rm["buy_strictness_floor"] = float(strict_floor)
                                    if entry_gate is not None:
                                        rm["entry_confidence_gate"] = float(entry_gate)
                                    cfg_ov = {}
                                    if bs is not None:
                                        cfg_ov["batch_size"] = int(bs)
                                    if ms is not None:
                                        cfg_ov["memory_size"] = int(ms)
                                    if hs is not None:
                                        cfg_ov["hidden_sizes"] = list(hs)
                                    if dec is not None:
                                        cfg_ov["eps_decay_steps"] = int(dec)
                                    combos.append({"risk_management": rm, "cfg_overrides": (cfg_ov or None)})
        selected = combos
        if len(combos) > max_models:
            idxs = []
            for i in range(max_models):
                idxs.append(int((i * len(combos)) / max_models))
            selected = [combos[i] for i in idxs]

        # queue tasks
        queued = []
        for i, rm in enumerate(selected):
            env_overrides = {"risk_management": rm.get("risk_management")}
            cfg_overrides = rm.get("cfg_overrides")
            seed_i = int(seed_base) + int(i)
            task = train_dqn_symbol.apply_async(
                kwargs={
                    "symbol": symbol,
                    "episodes": episodes,
                    "seed": seed_i,
                    "episode_length": episode_length,
                    "engine": engine,
                    "encoder_id": (encoder_id or None),
                    "train_encoder": bool(train_encoder),
                    "direction": direction,
                    "env_overrides": env_overrides,
                    "cfg_overrides": cfg_overrides,
                    "limit_candles": 100000,
                    "cutoff_days": cutoff_days,
                },
                queue="train",
            )
            queued.append({"i": i, "task_id": task.id, "seed": seed_i, "risk_management": env_overrides.get("risk_management"), "cfg_overrides": cfg_overrides})
            try:
                redis_client.lrem("ui:tasks", 0, task.id)
                redis_client.lpush("ui:tasks", task.id)
                redis_client.ltrim("ui:tasks", 0, 99)
            except Exception:
                pass
            # mark biz queued
            try:
                _ttl = int(os.environ.get('TRAIN_QUEUED_TTL', '300'))
            except Exception:
                _ttl = 300
            try:
                def _fmt_f(v):
                    if v is None:
                        return "na"
                    if isinstance(v, int):
                        return str(v)
                    s = f"{float(v):.6f}".rstrip('0').rstrip('.')
                    return s if s else "0"
                _rm_obj = env_overrides.get("risk_management") if isinstance(env_overrides, dict) else None
                _rm = "rm:-"
                if isinstance(_rm_obj, dict):
                    _rm = f"rm:{_fmt_f(_rm_obj.get('STOP_LOSS_PCT'))}:{_fmt_f(_rm_obj.get('TAKE_PROFIT_PCT'))}:{_fmt_f(_rm_obj.get('min_hold_steps'))}:{_fmt_f(_rm_obj.get('volume_threshold'))}:{_fmt_f(_rm_obj.get('atr_trail_mult'))}"
                _cfg = "cfg:-"
                try:
                    if isinstance(cfg_overrides, dict) and cfg_overrides:
                        _cfg = f"cfg:bs={_fmt_f(cfg_overrides.get('batch_size'))}:ms={_fmt_f(cfg_overrides.get('memory_size'))}:hs={str(cfg_overrides.get('hidden_sizes'))}:dec={_fmt_f(cfg_overrides.get('eps_decay_steps'))}"
                except Exception:
                    _cfg = "cfg:-"
                _enc = (encoder_id or '-')
                _train_enc = 1 if train_encoder else 0
                _biz_key = f"{symbol}|{engine}|{_enc}|{_train_enc}|{episodes}|{episode_length}|cut={cutoff_days}|{direction}|{_rm}|{_cfg}"
                redis_client.setex(f"celery:train:queued:{_biz_key}", max(30, _ttl), task.id)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "symbol": symbol,
            "total_combinations": total,
            "selected": len(selected),
            "seed_base": seed_base,
            "cutoff_days": cutoff_days,
            "queued": queued,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@training_bp.route('/train_xgb_symbol', methods=['POST'])
def train_xgb_symbol_route():
    """
    Запускает обучение XGB (buy/sell/hold) для одного символа.
    """
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or request.form.get('symbol') or 'BTCUSDT').strip().upper()
    direction = (data.get('direction') or request.form.get('direction') or 'long').strip().lower()
    horizon_steps = data.get('horizon_steps') or request.form.get('horizon_steps')
    threshold = data.get('threshold') or request.form.get('threshold')
    limit_candles = data.get('limit_candles') or request.form.get('limit_candles')
    task_name = data.get('task') or request.form.get('task')
    fee_bps = data.get('fee_bps') or request.form.get('fee_bps')
    max_hold_steps = data.get('max_hold_steps') or request.form.get('max_hold_steps')
    min_profit = data.get('min_profit') or request.form.get('min_profit')
    label_delta = data.get('label_delta') or request.form.get('label_delta')
    entry_stride = data.get('entry_stride') or request.form.get('entry_stride')
    max_trades = data.get('max_trades') or request.form.get('max_trades')
    cutoff_days = data.get('cutoff_days') or request.form.get('cutoff_days') or request.args.get('cutoff_days') or 90
    try:
        horizon_steps = int(horizon_steps) if horizon_steps not in (None, '') else None
    except Exception:
        horizon_steps = None
    try:
        threshold = float(threshold) if threshold not in (None, '') else None
    except Exception:
        threshold = None
    try:
        limit_candles = int(limit_candles) if limit_candles not in (None, '') else None
    except Exception:
        limit_candles = None
    try:
        fee_bps = float(fee_bps) if fee_bps not in (None, '') else None
    except Exception:
        fee_bps = None
    try:
        max_hold_steps = int(max_hold_steps) if max_hold_steps not in (None, '') else None
    except Exception:
        max_hold_steps = None
    try:
        min_profit = float(min_profit) if min_profit not in (None, '') else None
    except Exception:
        min_profit = None
    try:
        label_delta = float(label_delta) if label_delta not in (None, '') else None
    except Exception:
        label_delta = None
    try:
        entry_stride = int(entry_stride) if entry_stride not in (None, '') else None
    except Exception:
        entry_stride = None
    try:
        max_trades = int(max_trades) if max_trades not in (None, '') else None
    except Exception:
        max_trades = None
    try:
        cutoff_days = int(cutoff_days) if cutoff_days not in (None, '') else 90
    except Exception:
        cutoff_days = 90
    if cutoff_days < 0:
        cutoff_days = 0

    task = train_xgb_symbol.apply_async(kwargs={
        'symbol': symbol,
        'direction': direction,
        'horizon_steps': horizon_steps,
        'threshold': threshold,
        'limit_candles': limit_candles,
        'task': task_name,
        'fee_bps': fee_bps,
        'max_hold_steps': max_hold_steps,
        'min_profit': min_profit,
        'label_delta': label_delta,
        'entry_stride': entry_stride,
        'max_trades': max_trades,
        'cutoff_days': cutoff_days,
    }, queue='celery')

    # UI list
    try:
        redis_client.lrem("ui:tasks", 0, task.id)
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)
    except Exception:
        pass

    wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    if wants_json:
        return jsonify({"success": True, "task_id": task.id, "symbol": symbol, "direction": direction, "task": task_name})
    return redirect(url_for("index"))


@training_bp.route('/train_xgb_grid', methods=['POST'])
def train_xgb_grid_route():
    """
    Запускает быстрый grid (threshold/horizon/direction) и затем финальный train лучшего конфига.
    """
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or request.form.get('symbol') or 'BTCUSDT').strip().upper()
    limit_final = data.get('limit_candles_final') or request.form.get('limit_candles_final')
    limit_quick = data.get('limit_candles_quick') or request.form.get('limit_candles_quick')
    max_hold_steps = data.get('max_hold_steps') or request.form.get('max_hold_steps')
    min_profit = data.get('min_profit') or request.form.get('min_profit')
    cutoff_days = data.get('cutoff_days') or request.form.get('cutoff_days') or request.args.get('cutoff_days') or 90
    try:
        limit_final = int(limit_final) if limit_final not in (None, '') else None
    except Exception:
        limit_final = None
    try:
        limit_quick = int(limit_quick) if limit_quick not in (None, '') else None
    except Exception:
        limit_quick = None
    try:
        max_hold_steps = int(max_hold_steps) if max_hold_steps not in (None, '') else None
    except Exception:
        max_hold_steps = None
    try:
        min_profit = float(min_profit) if min_profit not in (None, '') else None
    except Exception:
        min_profit = None
    try:
        cutoff_days = int(cutoff_days) if cutoff_days not in (None, '') else 90
    except Exception:
        cutoff_days = 90
    if cutoff_days < 0:
        cutoff_days = 0

    task = train_xgb_grid.apply_async(kwargs={
        'symbol': symbol,
        'limit_candles_final': limit_final,
        'limit_candles_quick': limit_quick,
        'base_max_hold_steps': max_hold_steps,
        'base_min_profit': min_profit,
        'cutoff_days': cutoff_days,
    }, queue='celery')

    try:
        redis_client.lrem("ui:tasks", 0, task.id)
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)
    except Exception:
        pass

    wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    if wants_json:
        return jsonify({"success": True, "task_id": task.id, "symbol": symbol})
    return redirect(url_for("index"))


@training_bp.route('/train_xgb_grid_task', methods=['POST'])
def train_xgb_grid_task_route():
    """
    Универсальный grid: directional или entry/exit в зависимости от task.
    """
    data = request.get_json(silent=True) or {}
    symbol = (data.get('symbol') or request.form.get('symbol') or 'BTCUSDT').strip().upper()
    direction = (data.get('direction') or request.form.get('direction') or 'long').strip().lower()
    task_name = (data.get('task') or request.form.get('task') or 'directional').strip().lower()
    limit_final = data.get('limit_candles_final') or request.form.get('limit_candles_final')
    limit_quick = data.get('limit_candles_quick') or request.form.get('limit_candles_quick')
    fee_bps = data.get('fee_bps') or request.form.get('fee_bps')
    max_hold_steps = data.get('max_hold_steps') or request.form.get('max_hold_steps')
    min_profit = data.get('min_profit') or request.form.get('min_profit')
    cutoff_days = data.get('cutoff_days') or request.form.get('cutoff_days') or request.args.get('cutoff_days') or 90
    try:
        limit_final = int(limit_final) if limit_final not in (None, '') else None
    except Exception:
        limit_final = None
    try:
        limit_quick = int(limit_quick) if limit_quick not in (None, '') else None
    except Exception:
        limit_quick = None
    try:
        fee_bps = float(fee_bps) if fee_bps not in (None, '') else None
    except Exception:
        fee_bps = None
    try:
        max_hold_steps = int(max_hold_steps) if max_hold_steps not in (None, '') else None
    except Exception:
        max_hold_steps = None
    try:
        min_profit = float(min_profit) if min_profit not in (None, '') else None
    except Exception:
        min_profit = None
    try:
        cutoff_days = int(cutoff_days) if cutoff_days not in (None, '') else 90
    except Exception:
        cutoff_days = 90
    if cutoff_days < 0:
        cutoff_days = 0

    if task_name == 'directional':
        task = train_xgb_grid.apply_async(kwargs={
            'symbol': symbol,
            'limit_candles_final': limit_final,
            'limit_candles_quick': limit_quick,
            'base_max_hold_steps': max_hold_steps,
            'base_min_profit': min_profit,
            'cutoff_days': cutoff_days,
        }, queue='celery')
    else:
        task = train_xgb_grid_entry_exit.apply_async(kwargs={
            'symbol': symbol,
            'task': task_name,
            'direction': direction,
            'limit_candles_final': limit_final,
            'limit_candles_quick': limit_quick,
            'base_max_hold_steps': max_hold_steps,
            'base_fee_bps': fee_bps,
            'base_min_profit': min_profit,
            'cutoff_days': cutoff_days,
        }, queue='celery')

    try:
        redis_client.lrem("ui:tasks", 0, task.id)
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)
    except Exception:
        pass

    wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    if wants_json:
        return jsonify({"success": True, "task_id": task.id, "symbol": symbol, "task": task_name, "direction": direction})
    return redirect(url_for("index"))


@training_bp.route('/train_xgb_grid_full', methods=['POST'])
def train_xgb_grid_full_route():
    """Full hyper-parameter grid: labeling + model params."""
    data = request.get_json(silent=True) or {}

    def _get(key, cast=None):
        v = data.get(key)
        if v is None or v == '':
            return None
        if cast:
            return cast(v)
        return v

    def _bool(key):
        v = data.get(key)
        if isinstance(v, bool):
            return v
        return str(v or "").strip().lower() in ("1", "true", "yes", "on")

    task = train_xgb_grid_full.apply_async(kwargs={
        'symbol': _get('symbol') or 'BTCUSDT',
        'direction': _get('direction') or 'long',
        'task': _get('task') or 'entry_long',
        'parallel_mode': _bool('parallel_mode'),
        'parallel_workers': _get('parallel_workers', int),
        'limit_candles': _get('limit_candles', int),
        'cutoff_days': _get('cutoff_days', int) or 90,
        'p_enter_threshold_list': data.get('p_enter_threshold_list'),
        'entry_tp_pct': _get('entry_tp_pct', float),
        'entry_sl_pct': _get('entry_sl_pct', float),
        'entry_trail_pct': _get('entry_trail_pct', float),
        'entry_tp_pct_list': data.get('entry_tp_pct_list'),
        'entry_sl_pct_list': data.get('entry_sl_pct_list'),
        'entry_trail_pct_list': data.get('entry_trail_pct_list'),
        'horizon_steps_list': data.get('horizon_steps_list'),
        'threshold_list': data.get('threshold_list'),
        'max_hold_steps_list': data.get('max_hold_steps_list'),
        'min_profit_list': data.get('min_profit_list'),
        'fee_bps_list': data.get('fee_bps_list'),
        'max_depth_list': data.get('max_depth_list'),
        'learning_rate_list': data.get('learning_rate_list'),
        'n_estimators_list': data.get('n_estimators_list'),
        'subsample_list': data.get('subsample_list'),
        'colsample_bytree_list': data.get('colsample_bytree_list'),
        'reg_lambda_list': data.get('reg_lambda_list'),
        'min_child_weight_list': data.get('min_child_weight_list'),
        'gamma_list': data.get('gamma_list'),
        'scale_pos_weight_list': data.get('scale_pos_weight_list'),
        'early_stopping_rounds': _get('early_stopping_rounds', int) or 50,
        'keep_top_n': _get('keep_top_n', int) or 20,
        'use_1m_microvol': _bool('use_1m_microvol'),
        'use_1m_momentum': _bool('use_1m_momentum'),
        'use_1m_candle_structure': _bool('use_1m_candle_structure'),
        'use_1m_volume': _bool('use_1m_volume'),
        'use_1d_regime': _bool('use_1d_regime'),
        'use_sr_features': True if 'use_sr_features' not in data else _bool('use_sr_features'),
    }, queue='train')

    try:
        redis_client.lrem("ui:tasks", 0, task.id)
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)
    except Exception:
        pass

    return jsonify({"success": True, "task_id": task.id, "symbol": data.get('symbol'), "task": data.get('task')})


@training_bp.route('/xgb_grid_status', methods=['GET'])
def xgb_grid_status():
    task_id = (request.args.get("task_id") or "").strip()
    if not task_id:
        return jsonify({"success": False, "error": "task_id required"}), 400
    ar = AsyncResult(task_id, app=celery)
    resp = {"success": True, "task_id": task_id, "state": ar.state}
    if ar.state in ("PROGRESS",):
        meta = ar.info or {}
        resp["done"] = meta.get("done", 0)
        resp["total"] = meta.get("total", 0)
        resp["keep_top_n"] = meta.get("keep_top_n", 0)
        logs = meta.get("logs", [])
        resp["last_log"] = logs[-1] if logs else ""
    elif ar.state == "SUCCESS":
        meta = ar.result if isinstance(ar.result, dict) else {}
        if meta.get("success") is False:
            resp["state"] = "FAILURE"
            resp["error"] = meta.get("error") or "unknown task error"
        else:
            resp["done"] = meta.get("total_runs", meta.get("done", 0))
            resp["total"] = meta.get("total_runs", meta.get("total", 0))
            resp["best"] = meta.get("best")
    elif ar.state == "FAILURE":
        resp["error"] = str(ar.result) if ar.result else "unknown"
    return jsonify(resp)


@training_bp.route('/xgb_active_training', methods=['GET'])
def xgb_active_training():
    """Scan Redis for active XGB training tasks and return their Celery status."""
    import time as _time
    import re as _re
    try:
        keys = redis_client.keys("celery:train:xgb:*")
        active = []
        seen_task_ids = set()
        inspected_active = {}
        try:
            inspected = celery.control.inspect(timeout=1).active() or {}
        except Exception:
            inspected = {}
        for worker_tasks in inspected.values():
            if not isinstance(worker_tasks, list):
                continue
            for task in worker_tasks:
                if not isinstance(task, dict):
                    continue
                task_name_full = str(task.get("name") or task.get("type") or "")
                task_id = str(task.get("id") or "").strip()
                if task_id and task_name_full.startswith("tasks.xgb_tasks."):
                    inspected_active[task_id] = task

        for key in keys:
            try:
                key_type = redis_client.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode()
                if str(key_type) != "string":
                    continue
            except Exception:
                continue

            task_id = redis_client.get(key)
            if not task_id:
                continue
            if isinstance(task_id, bytes):
                task_id = task_id.decode()
            key_str = key.decode() if isinstance(key, bytes) else key
            if str(key_str).startswith("celery:train:xgb:progress:"):
                continue

            ar = AsyncResult(task_id, app=celery)
            state = ar.state
            active_by_worker = task_id in inspected_active
            if active_by_worker and state in ("SUCCESS", "FAILURE", "REVOKED"):
                state = "PROGRESS"

            if state in ("SUCCESS", "FAILURE", "REVOKED") and not active_by_worker:
                continue

            parts = key_str.replace("celery:train:xgb:", "").split(":")
            grid_type = parts[0] if parts else "unknown"
            symbol = parts[1] if len(parts) > 1 else ""
            task_name = parts[2] if len(parts) > 2 else ""

            info = {
                "task_id": task_id,
                "state": state,
                "grid_type": grid_type,
                "symbol": symbol,
                "task_name": task_name,
            }

            if state == "PROGRESS":
                meta = ar.info or {}
                info["done"] = meta.get("done", 0)
                info["total"] = meta.get("total", 0)
                info["symbol"] = meta.get("symbol") or symbol
                info["task_name"] = meta.get("task") or task_name
                info["started_at"] = meta.get("started_at")
                logs = meta.get("logs", [])
                info["last_log"] = logs[-1] if logs else ""
                info["logs_tail"] = logs[-5:] if logs else []
                all_logs = logs
                sa = meta.get("started_at")
                d = meta.get("done", 0)
                t = meta.get("total", 0)
                if sa and d > 0 and t > 0:
                    elapsed = _time.time() - float(sa)
                    avg_per_run = elapsed / d
                    remaining = (t - d) * avg_per_run
                    info["elapsed_sec"] = round(elapsed)
                    info["eta_sec"] = round(remaining)
                    info["avg_per_run_sec"] = round(avg_per_run, 1)
                elif d > 0 and t > 0 and len(all_logs) >= 2:
                    _ts_re = _re.compile(r'^\[(\d{2}):(\d{2}):(\d{2})\].*?(\d+)/(\d+)')
                    pairs = []
                    for lg in all_logs:
                        m = _ts_re.match(lg)
                        if m:
                            sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                            pairs.append((sec, int(m.group(4))))
                    if len(pairs) >= 2:
                        dt = pairs[-1][0] - pairs[0][0]
                        if dt < 0:
                            dt += 86400
                        dd = pairs[-1][1] - pairs[0][1]
                        if dt > 0 and dd > 0:
                            avg_per_run = dt / dd
                            remaining = (t - d) * avg_per_run
                            first_ts = pairs[0][0]
                            now_s = _time.gmtime()
                            now_sec = now_s.tm_hour * 3600 + now_s.tm_min * 60 + now_s.tm_sec
                            elapsed_approx = now_sec - first_ts
                            if elapsed_approx < 0:
                                elapsed_approx += 86400
                            info["elapsed_sec"] = round(elapsed_approx)
                            info["eta_sec"] = round(remaining)
                            info["avg_per_run_sec"] = round(avg_per_run, 1)
            elif state == "PENDING":
                info["done"] = 0
                info["total"] = 0

            active.append(info)
            seen_task_ids.add(task_id)

        for key in redis_client.keys("celery:train:xgb:progress:*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            task_id = str(key_str).replace("celery:train:xgb:progress:", "", 1)
            if not task_id or task_id in seen_task_ids:
                continue

            ar = AsyncResult(task_id, app=celery)
            state = ar.state
            active_by_worker = task_id in inspected_active
            if active_by_worker and state in ("SUCCESS", "FAILURE", "REVOKED"):
                state = "PROGRESS"
            if state in ("SUCCESS", "FAILURE", "REVOKED") and not active_by_worker:
                try:
                    redis_client.delete(key)
                except Exception:
                    pass
                continue

            progress = redis_client.hgetall(key) or {}
            meta = ar.info if isinstance(ar.info, dict) else {}
            logs = meta.get("logs", []) if isinstance(meta.get("logs", []), list) else []
            symbol = meta.get("symbol") or progress.get("symbol") or ""
            task_name = meta.get("task") or progress.get("task") or ""

            try:
                done = int(meta.get("done", progress.get("done", 0)) or 0)
            except Exception:
                done = 0
            try:
                total = int(meta.get("total", progress.get("total", 0)) or 0)
            except Exception:
                total = 0

            info = {
                "task_id": task_id,
                "state": state,
                "grid_type": "gridfull",
                "symbol": symbol,
                "task_name": task_name,
                "done": done,
                "total": total,
                "started_at": meta.get("started_at") or (inspected_active.get(task_id) or {}).get("time_start"),
                "last_log": logs[-1] if logs else "",
                "logs_tail": logs[-5:] if logs else [],
            }
            sa = info.get("started_at")
            if sa and done > 0 and total > 0:
                elapsed = _time.time() - float(sa)
                avg_per_run = elapsed / done
                info["elapsed_sec"] = round(elapsed)
                info["eta_sec"] = round((total - done) * avg_per_run)
                info["avg_per_run_sec"] = round(avg_per_run, 1)

            active.append(info)
            seen_task_ids.add(task_id)

        for task_id, task in inspected_active.items():
            task_name_full = str(task.get("name") or task.get("type") or "")
            if not task_name_full.startswith("tasks.xgb_tasks."):
                continue
            if not task_id or task_id in seen_task_ids:
                continue
            kwargs = task.get("kwargs") if isinstance(task.get("kwargs"), dict) else {}
            progress = redis_client.hgetall(f"celery:train:xgb:progress:{task_id}") or {}
            try:
                done = int(progress.get("done", 0) or 0)
            except Exception:
                done = 0
            try:
                total = int(progress.get("total", 0) or 0)
            except Exception:
                total = 0
            active.append({
                "task_id": task_id,
                "state": "PROGRESS",
                "grid_type": task_name_full.rsplit(".", 1)[-1].replace("train_xgb_", ""),
                "symbol": kwargs.get("symbol") or progress.get("symbol") or "",
                "task_name": kwargs.get("task") or progress.get("task") or kwargs.get("direction") or "",
                "done": done,
                "total": total,
                "started_at": task.get("time_start"),
                "last_log": "Celery worker reports this XGB task as active",
                "logs_tail": ["Celery worker reports this XGB task as active"],
            })
            if task.get("time_start") and done > 0 and total > 0:
                active[-1]["elapsed_sec"] = round(_time.time() - float(task["time_start"]))
                active[-1]["avg_per_run_sec"] = round(active[-1]["elapsed_sec"] / done, 1)
                active[-1]["eta_sec"] = round((total - done) * active[-1]["avg_per_run_sec"])
            seen_task_ids.add(task_id)

        return jsonify({"success": True, "active": active})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@training_bp.route('/continue_training', methods=['POST'])
def continue_training_route():
    """Продолжает обучение из существующих файлов в result (dqn_model_*.pth, replay_buffer_*.pkl).

    Body: { file: 'result/train_result_<code>.pkl', episodes?: int }
    По имени train_result_* извлекается <code>, затем берутся dqn_model_<code>.pth и replay_buffer_<code>.pkl.
    """
    try:
        data = request.get_json(silent=True) or {}
        requested_file = (data.get('file') or data.get('model') or '').strip()
        episodes_str = data.get('episodes') or request.form.get('episodes')
        episodes = None
        try:
            if episodes_str is not None and str(episodes_str).strip() != '':
                episodes = int(episodes_str)
        except Exception:
            episodes = None
        # Разобрать seed; если не указан — сгенерировать позже
        seed_raw = (data.get('seed') or request.form.get('seed') or '').strip()
        seed = None
        try:
            if seed_raw != '':
                seed = int(seed_raw)
        except Exception:
            seed = None
        episode_length_str = data.get('episode_length') or request.form.get('episode_length')
        episode_length = None
        try:
            if episode_length_str is not None and str(episode_length_str).strip() != '':
                episode_length = int(episode_length_str)
        except Exception:
            episode_length = None
        # Direction (long/short) для DQN do-обучения
        direction_raw = (data.get('direction') or request.form.get('direction') or '').strip().lower()
        direction = 'short' if direction_raw == 'short' else 'long'

        result_dir = _Path('result')
        if not result_dir.exists():
            return jsonify({"success": False, "error": "Папка result не найдена"}), 400

        if not requested_file:
            return jsonify({"success": False, "error": "Не указан путь к файлу весов dqn_model_*.pth"}), 400

        # Нормализуем и проверяем путь
        req_norm = requested_file.replace('\\', '/')
        safe_path = _Path(req_norm)
        if not safe_path.is_absolute():
            if not (safe_path.parts and safe_path.parts[0].lower() == result_dir.name.lower()):
                safe_path = result_dir / safe_path.name
        try:
            cand_resolved = safe_path.resolve()
        except Exception:
            cand_resolved = safe_path
        inside_result = str(cand_resolved).lower().startswith(str(result_dir.resolve()).lower())

        if not (inside_result and cand_resolved.exists() and cand_resolved.is_file()):
            return jsonify({"success": False, "error": "Неверный путь к файлу или файл не существует"}), 400

        model_file = None
        replay_file = None
        code = None
        symbol_from_path = None

        # Поддержка трёх форматов: train_result_*, dqn_model_* и runs/<run_id>/model.pth
        if cand_resolved.name.startswith('train_result_') and cand_resolved.suffix == '.pkl':
            fname = cand_resolved.stem  # train_result_<code>
            parts = fname.split('_', 2)
            if len(parts) < 3:
                return jsonify({"success": False, "error": "Не удалось извлечь код модели из имени файла"}), 400
            code = parts[2]
            model_file = result_dir / f'dqn_model_{code}.pth'
            replay_file = result_dir / f'replay_buffer_{code}.pkl'
            if not model_file.exists():
                return jsonify({"success": False, "error": f"Файл {model_file.name} не найден"}), 400
            # replay_file может отсутствовать – это допустимо
        elif cand_resolved.name.startswith('dqn_model_') and cand_resolved.suffix == '.pth':
            fname = cand_resolved.stem  # dqn_model_<code>
            parts = fname.split('_', 2)
            if len(parts) < 3:
                return jsonify({"success": False, "error": "Не удалось извлечь код модели из имени весов"}), 400
            code = parts[2]
            model_file = cand_resolved
            replay_file = result_dir / f'replay_buffer_{code}.pkl'  # может не существовать
        elif cand_resolved.name == 'model.pth' and 'runs' in cand_resolved.parts:
            # Новый формат: result/<SYMBOL>/runs/<run_id>/model.pth
            try:
                parts = list(cand_resolved.parts)
                runs_idx = parts.index('runs')
                run_id = parts[runs_idx + 1] if runs_idx + 1 < len(parts) else None
                symbol_from_path = parts[runs_idx - 1] if runs_idx - 1 >= 0 else None
                if not run_id:
                    return jsonify({"success": False, "error": "Некорректный путь к run"}), 400
                code = run_id
                model_file = cand_resolved
                run_dir = cand_resolved.parent
                replay_file = run_dir / 'replay.pkl'
                # train_result не обязателен для запуска дообучения
            except Exception:
                return jsonify({"success": False, "error": "Не удалось распознать путь run"}), 400
        else:
            return jsonify({"success": False, "error": "Ожидался файл dqn_model_*.pth, train_result_*.pkl или runs/.../model.pth"}), 400

        # Определяем тип модели
        model_type = 'dqn'
        if symbol_from_path and str(symbol_from_path).lower() == 'sac':
            model_type = 'sac'
        elif 'result/sac/' in str(model_file).replace('\\','/').lower():
            model_type = 'sac'

        symbol_guess = None
        try:
            if symbol_from_path and model_type == 'dqn':
                symbol_guess = (str(symbol_from_path).upper())
            elif symbol_from_path and model_type == 'sac':
                # NOTE: в этом модуле pathlib.Path импортирован как _Path
                symbol_guess = (str(_Path(symbol_from_path).stem).upper())
            else:
                symbol_guess = (code.split('_', 1)[0] + 'USDT').upper()
        except Exception:
            symbol_guess = None
        if not symbol_guess:
            symbol_guess = 'BTCUSDT'

        # DQN: когда путь вида result/dqn/<SYMBOL>/runs/<run_id>/model.pth,
        # <SYMBOL> обычно "TON"/"BTC" без суффикса USDT. Per-symbol overrides
        # (lookback_window/indicators_config/и т.д.) завязаны на ключи типа TONUSDT,
        # поэтому нормализуем для совместимости continue-training.
        try:
            if model_type == 'dqn' and symbol_guess and not str(symbol_guess).upper().endswith('USDT'):
                symbol_guess = str(symbol_guess).upper() + 'USDT'
        except Exception:
            pass

        if seed is None:
            import random as _rnd
            seed = _rnd.randint(1, 2**31 - 1)
            logger.info(f"[continue_training] generated random seed={seed}")

        if model_type == 'sac':
            task = train_sac_symbol.apply_async(kwargs={'symbol': symbol_guess, 'episodes': episodes, 'seed': seed, 'episode_length': episode_length, 'model_path': str(model_file), 'buffer_path': str(replay_file) if replay_file and os.path.exists(replay_file) else None}, queue='train')
        else:
            # Прокидываем пути на веса/буфер через Redis, чтобы Celery-задача могла продолжить обучение
            try:
                redis_client.setex('continue:model_path', 600, str(model_file))
                if replay_file and os.path.exists(replay_file):
                    redis_client.setex('continue:buffer_path', 600, str(replay_file))
                else:
                    try:
                        redis_client.delete('continue:buffer_path')
                    except Exception:
                        pass
            except Exception:
                pass
            task = train_dqn_symbol.apply_async(kwargs={'symbol': symbol_guess, 'episodes': episodes, 'seed': seed, 'episode_length': episode_length, 'direction': direction}, queue='train')

        # Добавим задачу в UI список
        try:
            redis_client.lrem("ui:tasks", 0, task.id)
            redis_client.lpush("ui:tasks", task.id)
            redis_client.ltrim("ui:tasks", 0, 49)
        except Exception:
            pass

        return jsonify({"success": True, "task_id": task.id, "seed": seed, "model_type": model_type, "symbol": symbol_guess, "direction": direction})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


