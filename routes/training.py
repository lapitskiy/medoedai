from flask import Blueprint, jsonify, request, redirect, url_for, current_app as app
from celery.result import AsyncResult
from tasks.celery_tasks import celery, train_dqn_multi_crypto, train_dqn_symbol
from tasks.sac_tasks import train_sac_symbol
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
    # Дедупликация на этапе постановки: не ставим, если такой же запуск уже в очереди/был недавно
    try:
        _engine = (engine or '').lower()
        _enc = (encoder_id or '-')
        _train_enc = 1 if train_encoder else 0
        _eps = episodes if (episodes is not None) else 'env'
        _ep_len = episode_length if (episode_length is not None) else 'cfg'
        _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|{(direction or 'long')}"
        _queued_key = f"celery:train:queued:{_biz_key}"
        _running_key_biz = f"celery:train:running:{_biz_key}"
        _finished_key = f"celery:train:finished:{_biz_key}"
        if redis_client.get(_running_key_biz) or redis_client.get(_queued_key) or redis_client.get(_finished_key):
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

    task = train_dqn_symbol.apply_async(kwargs={'symbol': symbol, 'episodes': episodes, 'seed': seed, 'episode_length': episode_length, 'engine': engine, 'encoder_id': encoder_id, 'train_encoder': train_encoder, 'direction': direction}, queue="train")
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
            _biz_key = f"{symbol.upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|{(direction or 'long')}"
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
            "seed": seed
        })
    return redirect(url_for("index"))


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
                symbol_guess = (str(Path(symbol_from_path).stem).upper())
            else:
                symbol_guess = (code.split('_', 1)[0] + 'USDT').upper()
        except Exception:
            symbol_guess = None
        if not symbol_guess:
            symbol_guess = 'BTCUSDT'

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


