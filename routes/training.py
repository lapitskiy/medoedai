from flask import Blueprint, jsonify, request, redirect, url_for, current_app as app
from celery.result import AsyncResult
from tasks.celery_tasks import celery, train_dqn_multi_crypto, train_dqn_symbol
from tasks.sac_tasks import train_sac_symbol
from tasks.xgb_tasks import train_xgb_symbol, train_xgb_grid, train_xgb_grid_entry_exit, train_xgb_grid_full
from utils.redis_utils import get_redis_client
import os
import logging
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

        _engine = (engine or '').lower()
        _enc = (encoder_id or '-')
        _train_enc = 1 if train_encoder else 0
        _eps = episodes if (episodes is not None) else 'env'
        _ep_len = episode_length if (episode_length is not None) else 'cfg'
        _rm = _rm_tag(env_overrides)
        _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|{(direction or 'long')}|{_rm}"
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

    task = train_dqn_symbol.apply_async(kwargs={'symbol': symbol, 'episodes': episodes, 'seed': seed, 'episode_length': episode_length, 'engine': engine, 'encoder_id': encoder_id, 'train_encoder': train_encoder, 'direction': direction, 'env_overrides': env_overrides}, queue="train")
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
            _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|{(direction or 'long')}|{_rm}"
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
        })
    return redirect(url_for("index"))


@training_bp.post('/train_dqn_grid')
def train_dqn_grid_route():
    """Ставит в очередь несколько DQN-обучений (grid) с env_overrides поверх GLOBAL_OVERRIDES."""
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

        def _float(v, name):
            try:
                return float(v)
            except Exception:
                raise ValueError(f"{name} must be float")

        def _int(v, name):
            try:
                return int(v)
            except Exception:
                raise ValueError(f"{name} must be int")

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
            sl_from = _float(grid.get('sl_from'), 'sl_from')
            sl_to = _float(grid.get('sl_to'), 'sl_to')
            sl_step = _float(grid.get('sl_step'), 'sl_step')
            tp_from = _float(grid.get('tp_from'), 'tp_from')
            tp_to = _float(grid.get('tp_to'), 'tp_to')
            tp_step = _float(grid.get('tp_step'), 'tp_step')
            mh_from = _int(grid.get('min_hold_from'), 'min_hold_from')
            mh_to = _int(grid.get('min_hold_to'), 'min_hold_to')
            mh_step = _int(grid.get('min_hold_step'), 'min_hold_step')
            vt_from = _float(grid.get('vol_from'), 'vol_from')
            vt_to = _float(grid.get('vol_to'), 'vol_to')
            vt_step = _float(grid.get('vol_step'), 'vol_step')
        except ValueError as ve:
            return jsonify({"success": False, "error": str(ve)}), 400

        try:
            max_models = int(data.get('max_models'))
        except Exception:
            max_models = 10
        if max_models <= 0 or max_models > 500:
            return jsonify({"success": False, "error": "max_models must be 1..500"}), 400

        sl_list = _frange("SL", sl_from, sl_to, sl_step)
        tp_list = _frange("TP", tp_from, tp_to, tp_step)
        mh_list = list(range(mh_from, mh_to + 1, mh_step)) if mh_step > 0 else []
        if not mh_list:
            return jsonify({"success": False, "error": "min_hold range invalid"}), 400
        vt_list = _frange("volume_threshold", vt_from, vt_to, vt_step)

        # cartesian size
        total = len(sl_list) * len(tp_list) * len(mh_list) * len(vt_list)
        if total <= 0:
            return jsonify({"success": False, "error": "empty grid"}), 400
        if total > 500:
            return jsonify({"success": False, "error": f"grid too large: {total} (>500)"}), 400

        # select subset deterministically if needed
        combos = []
        for sl in sl_list:
            for tp in tp_list:
                for mh in mh_list:
                    for vt in vt_list:
                        combos.append({"STOP_LOSS_PCT": sl, "TAKE_PROFIT_PCT": tp, "min_hold_steps": mh, "volume_threshold": vt})
        selected = combos
        if len(combos) > max_models:
            idxs = []
            for i in range(max_models):
                idxs.append(int((i * len(combos)) / max_models))
            selected = [combos[i] for i in idxs]

        # queue tasks
        queued = []
        for i, rm in enumerate(selected):
            env_overrides = {"risk_management": rm}
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
                },
                queue="train",
            )
            queued.append({"i": i, "task_id": task.id, "seed": seed_i, "risk_management": rm})
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
                _rm = f"rm:{_fmt_f(rm.get('STOP_LOSS_PCT'))}:{_fmt_f(rm.get('TAKE_PROFIT_PCT'))}:{_fmt_f(rm.get('min_hold_steps'))}:{_fmt_f(rm.get('volume_threshold'))}"
                _enc = (encoder_id or '-')
                _train_enc = 1 if train_encoder else 0
                _biz_key = f"{symbol}|{engine}|{_enc}|{_train_enc}|{episodes}|{episode_length}|{direction}|{_rm}"
                redis_client.setex(f"celery:train:queued:{_biz_key}", max(30, _ttl), task.id)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "symbol": symbol,
            "total_combinations": total,
            "selected": len(selected),
            "seed_base": seed_base,
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

    task = train_xgb_grid.apply_async(kwargs={
        'symbol': symbol,
        'limit_candles_final': limit_final,
        'limit_candles_quick': limit_quick,
        'base_max_hold_steps': max_hold_steps,
        'base_min_profit': min_profit,
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
    label_delta = data.get('label_delta') or request.form.get('label_delta')
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
        label_delta = float(label_delta) if label_delta not in (None, '') else None
    except Exception:
        label_delta = None

    if task_name == 'directional':
        task = train_xgb_grid.apply_async(kwargs={
            'symbol': symbol,
            'limit_candles_final': limit_final,
            'limit_candles_quick': limit_quick,
            'base_max_hold_steps': max_hold_steps,
            'base_min_profit': min_profit,
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
            'base_label_delta': label_delta,
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

    task = train_xgb_grid_full.apply_async(kwargs={
        'symbol': _get('symbol') or 'BTCUSDT',
        'direction': _get('direction') or 'long',
        'task': _get('task') or 'entry_long',
        'limit_candles': _get('limit_candles', int),
        'horizon_steps_list': data.get('horizon_steps_list'),
        'threshold_list': data.get('threshold_list'),
        'max_hold_steps_list': data.get('max_hold_steps_list'),
        'min_profit_list': data.get('min_profit_list'),
        'fee_bps_list': data.get('fee_bps_list'),
        'label_delta_list': data.get('label_delta_list'),
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
    }, queue='celery')

    try:
        redis_client.lrem("ui:tasks", 0, task.id)
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)
    except Exception:
        pass

    return jsonify({"success": True, "task_id": task.id, "symbol": data.get('symbol'), "task": data.get('task')})


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


