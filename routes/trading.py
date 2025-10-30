from flask import Blueprint, jsonify, request, render_template
from utils.redis_utils import get_redis_client
from tasks.celery_task_trade import start_trading_task
from utils.trade_utils import get_recent_trades, get_trade_statistics, get_trades_by_symbol, get_model_predictions
import logging
import docker
import json
from datetime import datetime
from typing import Optional
import time

trading_bp = Blueprint('trading', __name__)


@trading_bp.route('/trading_agent')
def trading_agent_page():
    """Страница торгового агента"""
    return render_template('trading_agent.html')

@trading_bp.route('/agent/<symbol>')
def agent_symbol_page(symbol: str):
    """Страница агента, отфильтрованная по конкретному символу (BTCUSDT, TONUSDT и т.д.)"""
    try:
        sym = (symbol or '').upper().strip()
        # Простейшая валидация: только латиница и 'USDT' в конце
        import re
        if not re.match(r'^[A-Z]{2,10}USDT$', sym):
            # дефолт на BTCUSDT
            sym = 'BTCUSDT'
        return render_template('agent_symbol.html', symbol=sym)
    except Exception:
        return render_template('agent_symbol.html', symbol='BTCUSDT')

@trading_bp.post('/api/trading/save_config')
def save_trading_config():
    """Автосохранение выбора моделей и консенсуса без запуска торговли."""
    try:
        data = request.get_json() or {}
        try:
            logging.info(f"[save_config] payload symbols={data.get('symbols')} sel_paths={len(data.get('model_paths') or [])} counts={(data.get('consensus') or {}).get('counts')}")
            logging.info(f"[save_config] FULL payload: {data}")
            # Детальный лог consensus
            consensus = data.get('consensus')
            if consensus:
                logging.info(f"[save_config] consensus counts: {consensus.get('counts')}")
                logging.info(f"[save_config] consensus percents: {consensus.get('percents')}")
        except Exception:
            pass
        symbols = data.get('symbols') or []
        model_paths = data.get('model_paths') or []
        consensus = data.get('consensus') or None
        take_profit_pct = data.get('take_profit_pct')  # Процент тейк-профита
        stop_loss_pct = data.get('stop_loss_pct')      # Процент стоп-лосса
        risk_management_type = data.get('risk_management_type')  # Способ управления рисками
        account_pct = data.get('account_pct')  # Доля счёта для сделки, %
        exit_mode = str(data.get('exit_mode') or '').strip() or None  # 'prediction' | 'risk_orders'
        leverage = data.get('leverage')  # 1..5
        import json as _json
        rc = get_redis_client()

        # Изменяем логику: вместо перезаписи, объединяем символы
        if symbols:
            existing_raw = rc.get('trading:symbols')
            existing_symbols = _json.loads(existing_raw) if existing_raw else []
            if not isinstance(existing_symbols, list):
                existing_symbols = []
            
            merged_symbols = list(set(existing_symbols + symbols))
            rc.set('trading:symbols', _json.dumps(merged_symbols, ensure_ascii=False))

        # Сохраняем только непустые списки моделей (и глобально, и per‑symbol)
        if isinstance(model_paths, list) and len(model_paths) > 0:
            rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
            # Пер‑символьное хранилище для корректного отображения и оркестрации
            symbol = symbols[0] if symbols else 'ALL'
            rc.set(f'trading:model_paths:{symbol}', _json.dumps(model_paths, ensure_ascii=False))
            try:
                logging.info(f"[save_config] symbol={symbol} model_paths_selected={len(model_paths)} -> saved per-symbol model_paths")
            except Exception:
                pass
        # Не перетираем консенсус пустыми/дефолтными значениями
        if consensus is not None and isinstance(model_paths, list) and len(model_paths) > 0:
            symbol = symbols[0] if symbols else 'ALL'
            # Опционально: синхронизируем total_selected с фактическим списком
            try:
                c = consensus.get('counts') if isinstance(consensus, dict) else None
                if isinstance(c, dict):
                    before = dict(c)
                    c['total_selected'] = len(model_paths)
                    logging.info(f"[save_config] symbol={symbol} counts_in={before} -> counts_saved={c}")
            except Exception:
                pass
            rc.set(f'trading:consensus:{symbol}', _json.dumps(consensus, ensure_ascii=False))
            try:
                logging.info(f"[save_config] symbol={symbol} consensus saved")
            except Exception:
                pass
        if take_profit_pct is not None:
            rc.set('trading:take_profit_pct', str(take_profit_pct))
        if stop_loss_pct is not None:
            rc.set('trading:stop_loss_pct', str(stop_loss_pct))
        if risk_management_type is not None:
            rc.set('trading:risk_management_type', str(risk_management_type))
        if account_pct is not None:
            try:
                ap = int(account_pct)
                if 1 <= ap <= 100:
                    rc.set('trading:account_pct', str(ap))
            except Exception:
                pass
        try:
            if exit_mode and symbols:
                sym = (symbols[0] if isinstance(symbols, list) and symbols else 'ALL')
                rc.set(f'trading:exit_mode:{sym}', exit_mode)
            if leverage is not None and symbols:
                try:
                    lev_int = int(leverage)
                    if 1 <= lev_int <= 5:
                        sym = (symbols[0] if isinstance(symbols, list) and symbols else 'ALL')
                        rc.set(f'trading:leverage:{sym}', str(lev_int))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            logging.info("[save_config] ✓ done")
        except Exception:
            pass
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/start')
def start_trading():
    """
    Запуск торговли в контейнере trading_agent через Celery задачу
    """
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTCUSDT'])
        account_id = str(data.get('account_id') or '').strip()
        # Новые параметры стратегии исполнения
        execution_mode = str(data.get('execution_mode') or '').strip() or None  # 'market' | 'limit_post_only'
        limit_config = data.get('limit_config') if isinstance(data.get('limit_config'), dict) else None
        immediate = bool(data.get('immediate') or False)
        immediate_side = str(data.get('side') or '').lower() if data.get('side') else None  # 'buy' | 'sell'
        exit_mode = str(data.get('exit_mode') or '').strip() or None  # 'prediction' | 'risk_orders'
        leverage = data.get('leverage')  # 1..5
        account_pct = data.get('account_pct')  # Доля счёта, %
        # Поддержка многомодельного запуска: model_paths (список) + совместимость с model_path
        model_paths = data.get('model_paths') or []
        model_path = data.get('model_path')
        if (not model_path) and isinstance(model_paths, list) and len(model_paths) > 0:
            model_path = model_paths[0]
        if not model_path:
            model_path = '/workspace/models/btc/ensemble-a/current/dqn_model.pth'
        
        # Сохраняем выбранные параметры в Redis для последующих вызовов (status/stop/balance/history)
        try:
            import json as _json
            _rc = get_redis_client()
            # Снимаем флаг ручного отключения для основного символа (если был выставлен стопом)
            try:
                sym0 = symbols[0] if symbols else None
                if sym0:
                    _rc.delete(f'trading:disabled:{sym0}')
            except Exception:
                pass
            _rc.set('trading:model_path', model_path)
            # НЕ перезаписываем общие модели, чтобы не убить другие агенты
            # try:
            #     if isinstance(model_paths, list):
            #         _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            # except Exception:
            #     pass
            # НЕ перезаписываем общие символы, чтобы не убить другие агенты
            # _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
            if account_id:
                _rc.set('trading:account_id', account_id)
            # Консенсус (counts/percents) — запоминаем выбор пользователя для конкретного символа
            try:
                consensus = data.get('consensus')
                if consensus is not None:
                    # Сохраняем консенсус для конкретного символа
                    symbol = symbols[0] if symbols else 'ALL'
                    _rc.set(f'trading:consensus:{symbol}', _json.dumps(consensus, ensure_ascii=False))
                    # Не сохраняем больше глобальные ключи консенсуса, чтобы агенты не перетирали друг друга
            except Exception:
                pass
            # Обновим last_model_paths для фолбэка тиков
            try:
                if isinstance(model_paths, list) and model_paths:
                    _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
            except Exception:
                pass
            # Сохраняем настройки для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            _rc.set(f'trading:symbols:{symbol}', _json.dumps(symbols, ensure_ascii=False))
            # Сохраняем выбранный режим исполнения и конфиг (per-symbol)
            try:
                if execution_mode:
                    _rc.set(f'trading:execution_mode:{symbol}', execution_mode)
                if isinstance(limit_config, dict):
                    _rc.set(f'trading:limit_config:{symbol}', _json.dumps(limit_config, ensure_ascii=False))
                if exit_mode:
                    _rc.set(f'trading:exit_mode:{symbol}', exit_mode)
                if leverage is not None:
                    try:
                        lev_int = int(leverage)
                        if 1 <= lev_int <= 5:
                            _rc.set(f'trading:leverage:{symbol}', str(lev_int))
                    except Exception:
                        pass
                if account_pct is not None:
                    try:
                        ap = int(account_pct)
                        if 1 <= ap <= 100:
                            _rc.set('trading:account_pct', str(ap))
                    except Exception:
                        pass
            except Exception:
                pass
            _rc.set(f'trading:model_path:{symbol}', model_path)
            if isinstance(model_paths, list):
                _rc.set(f'trading:model_paths:{symbol}', _json.dumps(model_paths, ensure_ascii=False))
            
            # Пишем мгновенный «активный» статус для конкретного символа
            initial_status = {
                'success': True,
                'is_trading': True,
                'trading_status': 'Активна',
                'trading_status_emoji': '🟢',
                'trading_status_full': '🟢 Активна',
                'symbol': symbol,
                'symbol_display': symbol,
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {},
                'current_price': 0.0,
                'last_model_prediction': None,
            }
            # Сохраняем статус для конкретного символа
            _rc.set(f'trading:status:{symbol}', _json.dumps(initial_status, ensure_ascii=False))
            # НЕ перезаписываем общий статус, чтобы не убить другие агенты
            # _rc.set('trading:current_status', _json.dumps(initial_status, ensure_ascii=False))
            from datetime import datetime as _dt
            _rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
        except Exception as _e:
            logging.error(f"Не удалось сохранить параметры торговли в Redis: {_e}")

        # Redis-лок: если уже идёт торговый шаг для этого символа, не стартуем второй параллельно
        try:
            _rc_lock = get_redis_client()
            # Проверяем блокировку для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            lock_key = f'trading:agent_lock:{symbol}'
            if _rc_lock.get(lock_key):
                return jsonify({
                    'success': False,
                    'error': f'Торговый шаг для {symbol} уже выполняется (agent_lock_active)'
                }), 429
        except Exception:
            pass

        # Запускаем Celery задачу для старта торговли (старый поток предсказаний/торгового шага)
        task = start_trading_task.apply_async(args=[symbols, model_path], countdown=0, expires=300, queue='trade')

        # Опционально: немедленно запустить стратегию исполнения (DDD) для текущего символа
        try:
            from tasks.celery_task_trade import start_execution_strategy as _start_exec
            if execution_mode and immediate and immediate_side in ('buy','sell'):
                sym0 = symbols[0] if symbols else 'BTCUSDT'
                # qty: возьмётся по умолчанию внутри задачи, если не задан
                _start_exec.apply_async(kwargs={
                    'symbol': sym0,
                    'execution_mode': execution_mode,
                    'side': immediate_side,
                    'qty': data.get('qty'),
                    'limit_config': (limit_config or {})
                }, queue='trade')
        except Exception:
            pass
        
        return jsonify({
            'success': True,
            'message': 'Торговля запущена через Celery задачу',
            'task_id': task.id
        }), 200
    except Exception as e:
        logging.error(f"Ошибка запуска торговли: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@trading_bp.post('/api/trading/execute_now')
def execute_now():
    """
    Разовый немедленный торговый шаг для конкретного символа (без ожидания периодики).
    Использует тот же пайплайн, что и периодический цикл, но единично.
    По умолчанию работает в боевом режиме (не dry_run), чтобы реально исполнять сделки.
    """
    try:
        data = request.get_json(silent=True) or {}
        symbols = data.get('symbols') or [data.get('symbol') or 'BTCUSDT']
        # dry_run не передаём в таск для совместимости со старыми воркерами
        # Передаём напрямую в очередь trade ту же задачу, что и периодика
        from tasks.celery_task_trade import execute_trade as _exec
        res = _exec.apply_async(kwargs={
            'symbols': symbols,
            'model_path': None,
            'model_paths': None,
        }, queue='trade')
        return jsonify({'success': True, 'enqueued': True, 'task_id': res.id}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/stop')
def stop_trading():
    """Остановка торговли в контейнере trading_agent"""
    try:
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                rc = get_redis_client()
                mp = rc.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Останавливаем торговлю через exec
            if model_path:
                cmd = f'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\"{model_path}\\"); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            else:
                cmd = 'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            
            exec_result = container.exec_run(cmd, tty=True)
            
            # Логируем результат выполнения команды
            logging.info(f"Stop trading - Exit code: {exec_result.exit_code}")
            if exec_result.output:
                output_str = exec_result.output.decode('utf-8')
                logging.info(f"Stop trading - Output: {output_str}")
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except Exception as parse_error:
                        logging.error(f"Ошибка парсинга результата: {parse_error}")
                        return jsonify({
                            'success': True,
                            'message': 'Торговля остановлена',
                            'output': output
                        }), 200
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Торговля остановлена',
                        'output': output
                    }), 200
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                logging.error(f"Ошибка выполнения команды остановки торговли: {error_output}")
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {error_output}'
                }), 500
                
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден. Запустите docker-compose up medoedai'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка stop_trading: {str(e)}'
        }), 500

@trading_bp.post('/api/trading/stop_symbol')
def stop_trading_symbol():
    try:
        data = request.get_json(silent=True) or {}
        symbol = str(data.get('symbol') or '').strip().upper()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        rc = get_redis_client()
        # Снимаем лок
        try:
            rc.delete(f'trading:agent_lock:{symbol}')
        except Exception:
            pass
        # Ставим флаг ручного отключения символа
        try:
            rc.set(f'trading:disabled:{symbol}', '1')
        except Exception:
            pass
        # Обновляем статус
        try:
            raw = rc.get(f'trading:status:{symbol}')
            status = json.loads(raw) if raw else {}
            if not isinstance(status, dict):
                status = {}
            status.update({
                'is_trading': False,
                'trading_status': 'Остановлена',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Остановлена'
            })
            rc.set(f'trading:status:{symbol}', json.dumps(status, ensure_ascii=False))
        except Exception:
            pass
        return jsonify({'success': True, 'symbol': symbol})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/agents')
def list_trading_agents():
    try:
        rc = get_redis_client()
        # Базовый список символов (можно расширить)
        known = ['BTCUSDT','ETHUSDT','SOLUSDT','TONUSDT','ADAUSDT','BNBUSDT','XRPUSDT']
        agents = []
        for sym in known:
            # Активность по локам (могут отсутствовать после рестарта)
            try:
                ttl = rc.ttl(f'trading:agent_lock:{sym}')
            except Exception:
                ttl = None
            # Статус per-symbol
            try:
                raw = rc.get(f'trading:status:{sym}')
                status = json.loads(raw) if raw else None
            except Exception:
                status = None
            # Фолбэк активности: если статус говорит, что торгуем — считаем активным
            try:
                is_trading_flag = bool(status and (status.get('is_trading') is True or str(status.get('trading_status') or '').strip() in ('Активна','🟢 Активна')))
            except Exception:
                is_trading_flag = False
            is_active = ((ttl is not None and isinstance(ttl, int) and ttl > 0) or is_trading_flag)
            # Консенсус per-symbol
            try:
                cons_raw = rc.get(f'trading:consensus:{sym}')
                consensus = json.loads(cons_raw) if cons_raw else None
            except Exception:
                consensus = None
            try:
                logging.info(f"[agents] {sym}: raw_consensus={consensus}")
            except Exception:
                pass
            # Список моделей
            total_models = 0
            try:
                mps_raw = rc.get(f'trading:model_paths:{sym}')
                if mps_raw:
                    parsed = json.loads(mps_raw)
                    if isinstance(parsed, list):
                        total_models = len(parsed)
            except Exception:
                total_models = 0
            # Вычисляем требуемые пороги так же, как оркестратор
            def _required(total_sel: int, counts: dict | None, regime: str) -> tuple[int,int,int,str,int]:
                req_flat = None
                req_trend = None
                try:
                    c = counts or {}
                    if isinstance(c.get('flat'), (int, float)):
                        req_flat = int(max(1, c.get('flat')))
                    if isinstance(c.get('trend'), (int, float)):
                        req_trend = int(max(1, c.get('trend')))
                except Exception:
                    pass
                default_req = 2 if total_sel >= 3 else max(1, total_sel)
                if req_flat is None:
                    req_flat = default_req
                if req_trend is None:
                    req_trend = default_req
                req_flat = int(min(max(1, req_flat), total_sel if total_sel>0 else 1))
                req_trend = int(min(max(1, req_trend), total_sel if total_sel>0 else 1))
                req_type = 'trend' if regime in ('uptrend','downtrend') else 'flat'
                required = (req_trend if req_type=='trend' else req_flat)
                return total_sel, req_flat, req_trend, req_type, required
            counts = (consensus or {}).get('counts') if isinstance(consensus, dict) else None
            regime = (status or {}).get('market_regime') or 'flat'
            # Насильно приводим total_selected к фактическому выбору моделей
            try:
                if isinstance(counts, dict):
                    before_ts = counts.get('total_selected')
                    counts['total_selected'] = total_models
                    try:
                        logging.info(f"[agents] {sym}: total_models={total_models}, counts_in_ts={before_ts}, counts_out_ts={counts.get('total_selected')}, flat={counts.get('flat')}, trend={counts.get('trend')}, regime={regime}")
                    except Exception:
                        pass
                    # Если исправили в памяти, сохраняем обратно в Redis
                    if before_ts != total_models and total_models > 0:
                        try:
                            consensus['counts'] = counts
                            rc.set(f'trading:consensus:{sym}', json.dumps(consensus, ensure_ascii=False))
                            logging.info(f"[agents] {sym}: FIXED Redis total_selected {before_ts} -> {total_models}")
                        except Exception as e:
                            logging.error(f"[agents] {sym}: Failed to fix Redis: {e}")
            except Exception:
                pass
            # Используем исправленное значение total_selected из counts
            corrected_total = counts.get('total_selected', total_models) if isinstance(counts, dict) else total_models
            total_sel, req_flat, req_trend, req_type, required = _required(corrected_total, counts, regime)
            try:
                logging.info(f"[agents] {sym}: required={required} ({req_type}), req_flat={req_flat}, req_trend={req_trend}")
            except Exception:
                pass
            agent_obj = {
                'symbol': sym,
                'active': bool(is_active),
                'status': status or {},
                'consensus': consensus or {},
                'total_models': total_models,
                'required_flat': req_flat,
                'required_trend': req_trend,
                'required_type': req_type,
                'required': required,
                'lock_ttl': (int(ttl) if ttl is not None else None)
            }
            try:
                logging.info(f"[agents] {sym}: agent_obj={agent_obj}")
            except Exception:
                pass
            agents.append(agent_obj)
        return jsonify({'success': True, 'agents': agents, 'ts': datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/status')
def trading_status():
    """Статус торговли из контейнера trading_agent"""
    try:
        # Входная точка: фиксируем вызов эндпоинта
        try:
            logging.info("[trading_status] ▶ request received")
        except Exception:
            pass
        # Сначала пробуем быстрый статус из Redis (обновляется периодическим таском)
        try:
            _rc = get_redis_client()
            cached = _rc.get('trading:current_status')
            cached_ts = _rc.get('trading:current_status_ts')
            if cached:
                status_obj = json.loads(cached)
                # Проверим свежесть (не старее 6 минут, > интервала beat)
                from datetime import timedelta
                is_fresh = True
                try:
                    if cached_ts:
                        ts = datetime.fromisoformat(cached_ts)
                        is_fresh = datetime.utcnow() <= (ts + timedelta(minutes=6))
                except Exception:
                    is_fresh = True
                if is_fresh:
                    # Возвращаем плоскую структуру для совместимости с фронтендом
                    flat = {'success': True, 'agent_status': status_obj}
                    if isinstance(status_obj, dict):
                        flat.update(status_obj)
                    # Приложим exit_mode per-symbol, если есть
                    try:
                        sym = flat.get('symbol') or flat.get('symbol_display')
                        if sym:
                            em = _rc.get(f'trading:exit_mode:{sym}')
                            if em:
                                flat['exit_mode'] = em.decode('utf-8') if isinstance(em, (bytes, bytearray)) else str(em)
                    except Exception:
                        pass
                    # Добавим pending_order (DDD), если активен intent
                    try:
                        import time as _t
                        import redis as _r
                        _rc2 = _r.Redis(host='redis', port=6379, db=0, decode_responses=True)
                        sym = flat.get('symbol') or flat.get('symbol_display')
                        if sym:
                            aid = _rc2.get(f'exec:active_intent:{sym}')
                            if aid:
                                raw = _rc2.get(f'exec:intent:{aid}')
                                if raw:
                                    import json as _json
                                    data = _json.loads(raw)
                                    flat['pending_order'] = {
                                        'symbol': data.get('symbol'),
                                        'intent_id': data.get('intent_id'),
                                        'state': data.get('state'),
                                        'price': data.get('price'),
                                        'attempts': data.get('attempts'),
                                        'age_sec': int(max(0, (_t.time() - float(data.get('created_at') or 0)))) if data.get('created_at') else None,
                                        'last_error': data.get('last_error')
                                    }
                    except Exception:
                        pass
                    try:
                        logging.info(f"[trading_status] ✓ using cached status | keys={list(flat.keys())}")
                        # Краткий обзор важных полей
                        logging.info("[trading_status] summary: is_trading=%s, position=%s, trades_count=%s",
                                    flat.get('is_trading'), bool(flat.get('position') or flat.get('current_position')), flat.get('trades_count'))
                    except Exception:
                        pass
                    return jsonify(flat), 200
        except Exception:
            pass

        # Нет свежего статуса в Redis — возвращаем понятный OFF статус для UI
        try:
            # Попробуем дополнить OFF статус сведениями о pending intent (DDD)
            pending_block = None
            try:
                import redis as _r
                _rc2 = _r.Redis(host='redis', port=6379, db=0, decode_responses=True)
                # Пройдём по известным символам и найдём active_intent
                known = ['BTCUSDT','ETHUSDT','SOLUSDT','TONUSDT','ADAUSDT','BNBUSDT','XRPUSDT']
                for sym in known:
                    aid = _rc2.get(f'exec:active_intent:{sym}')
                    if aid:
                        raw = _rc2.get(f'exec:intent:{aid}')
                        if raw:
                            import json as _json
                            data = _json.loads(raw)
                            pending_block = {
                                'symbol': data.get('symbol'),
                                'intent_id': data.get('intent_id'),
                                'state': data.get('state'),
                                'price': data.get('price'),
                                'attempts': data.get('attempts'),
                                'age_sec': int(max(0, (time.time() - float(data.get('created_at') or 0)))) if data.get('created_at') else None,
                                'last_error': data.get('last_error')
                            }
                            break
                    # если нашли — прекращаем цикл
                    if pending_block:
                        break
            except Exception:
                pending_block = None

            default_status = {
                'success': True,
                'is_trading': False,
                'trading_status': 'Остановлена',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Остановлена (агент не запущен)',
                'symbol': None,
                'symbol_display': 'Не указана',
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {},
                'current_price': 0.0,
                'last_model_prediction': None,
                'is_fresh': False,
                'reason': 'status not available in redis',
                'pending_order': pending_block
            }
            flat = {'success': True, 'agent_status': default_status}
            flat.update(default_status)
            try:
                logging.info("[trading_status] ⚠ no redis status, returning OFF state")
            except Exception:
                pass
            return jsonify(flat), 200
        except Exception:
            return jsonify({'success': False, 'error': 'status not available in redis', 'is_fresh': False}), 200
            
    except Exception as e:
        logging.error(f"Ошибка получения статуса: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

# Добавляем остальные торговые эндпоинты
@trading_bp.get('/api/trading/status_all')
def trading_status_all():
    """Итоговый статус: активные агенты"""
    try:
        _rc = get_redis_client()
        active_agents = []
        stored_symbols_data = _rc.get('trading:symbols')
        logging.info(f'[status_all] Raw trading:symbols data from Redis: {stored_symbols_data}')
        symbols = json.loads(stored_symbols_data) if stored_symbols_data else []
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TONUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
        for symbol in symbols:
            try:
                status_key = f'trading:status:{symbol}'
                status_data = _rc.get(status_key)
                if not status_data:
                    continue
                status_obj = json.loads(status_data)
                if not isinstance(status_obj, dict):
                    continue
                lock_key = f'trading:agent_lock:{symbol}'
                ttl = _rc.ttl(lock_key)
                status_text = str(status_obj.get('trading_status') or '').strip()
                is_trading_flag = bool(status_obj.get('is_trading') is True or status_text in ('торгуется', 'не торгуется'))
                is_active = ((ttl is not None and isinstance(ttl, int) and ttl > 0) or is_trading_flag)
                if not is_active:
                    continue
                agent_status = {
                    'symbol': symbol,
                    'is_active': True,
                    'ttl_seconds': int(ttl) if ttl is not None and isinstance(ttl, int) and ttl > 0 else 0,
                    'status': 'торгуется',
                    'current_price': status_obj.get('current_price'),
                    'position': status_obj.get('position'),
                    'trades_count': status_obj.get('trades_count'),
                    'last_prediction': status_obj.get('last_model_prediction'),
                    'amount': status_obj.get('amount'),
                    'amount_display': status_obj.get('amount_display')
                }
                active_agents.append(agent_status)
            except Exception:
                continue
        return jsonify({
            'success': True,
            'active_agents': active_agents,
            'total_active': len(active_agents)
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/latest_results')
def trading_latest_results():
    """Получение последних результатов торговли из Celery"""
    try:
        requested_symbol = (request.args.get('symbol') or '').upper().strip()
        latest_results = []
        try:
            rc = get_redis_client()
            keys = rc.keys('trading:latest_result_*') or []
            for k in keys:
                try:
                    raw = rc.get(k)
                    if not raw:
                        continue
                    result = json.loads(raw.decode('utf-8'))
                    latest_results.append(result)
                except Exception:
                    continue
        except Exception:
            pass
        if requested_symbol:
            latest_results = [r for r in latest_results if isinstance(r.get('symbols'), list) and requested_symbol in r.get('symbols')]
        latest_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify({
            'success': True,
            'latest_results': latest_results,
            'total_results': len(latest_results)
        }), 200
    except Exception as e:
        logging.error(f"Ошибка получения последних результатов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/balance')
def trading_balance():
    """Баланс берём из Redis-кэша trading:current_status"""
    try:
        _rc = get_redis_client()
        cached = _rc.get('trading:current_status') if _rc else None
        resp_obj = None
        if cached:
            try:
                st = json.loads(cached)
                if isinstance(st, dict) and st.get('balance'):
                    resp_obj = {
                        'success': True,
                        'balance': st.get('balance'),
                        'is_trading': st.get('is_trading', False),
                        'is_fresh': st.get('is_fresh', False)
                    }
            except Exception:
                resp_obj = None
        if resp_obj is None:
            resp_obj = {
                'success': True,
                'balance': {},
                'message': 'balance not available (agent not running)'
            }
        return jsonify(resp_obj), 200
    except Exception as e:
        logging.error(f"Ошибка получения баланса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/manual_buy')
def trading_manual_buy_bypass_prediction():
    """Реальная покупка в обход предсказания с установкой TP/SL по текущим настройкам.

    Body JSON: { symbol: 'BTCUSDT', qty?: float, override?: {execution_mode, leverage, limit_config, exit_mode, take_profit_pct, stop_loss_pct, risk_management_type} }
    """
    try:
        data = request.get_json() or {}
        sym = str(data.get('symbol') or 'BTCUSDT').upper().strip()
        if not sym.endswith('USDT'):
            sym = f"{sym}USDT"

        # 1) Создаём TradingAgent
        try:
            from trading_agent.trading_agent import TradingAgent
        except Exception as e:
            return jsonify({'success': False, 'error': f'TradingAgent import error: {e}'}), 500

        # Путь модели из Redis (если есть)
        model_path: Optional[str] = None
        try:
            rc = get_redis_client()
            mp = rc.get('trading:model_path')
            if mp:
                try:
                    model_path = mp.decode('utf-8') if isinstance(mp, (bytes, bytearray)) else str(mp)
                except Exception:
                    model_path = str(mp)
        except Exception:
            rc = None

        agent = TradingAgent(model_path=model_path)  # type: ignore
        agent.symbol = sym
        agent.base_symbol = sym

        # Количество (если не задано — рассчитать)
        qty = data.get('qty')
        try:
            qty = float(qty) if qty is not None else None
        except Exception:
            qty = None
        if qty is None or qty <= 0:
            try:
                qty = float(agent._calculate_trade_amount())  # type: ignore
            except Exception:
                qty = 0.001

        # 2) Эффективные настройки (per-symbol с фолбэком на глобальные)
        take_profit_pct = 1.0
        stop_loss_pct = 1.0
        risk_type = 'exchange_orders'
        exit_mode = 'prediction'
        execution_mode = 'market'
        leverage_val = 1
        limit_config = None
        try:
            if rc is None:
                rc = get_redis_client()

            # Глобальные/пер-символьные ключи
            tp_sym = rc.get(f'trading:take_profit_pct:{sym}')
            sl_sym = rc.get(f'trading:stop_loss_pct:{sym}')
            tp_glob = rc.get('trading:take_profit_pct')
            sl_glob = rc.get('trading:stop_loss_pct')
            rt = rc.get('trading:risk_management_type')
            em = rc.get(f'trading:exit_mode:{sym}')
            ex_mode = rc.get(f'trading:execution_mode:{sym}')
            lev = rc.get(f'trading:leverage:{sym}')
            lc = rc.get(f'trading:limit_config:{sym}')

            # Переопределения из запроса (если пришли)
            override = data.get('override') or {}
            def _pick_num(val, redis_val, default):
                if val is not None:
                    return float(val)
                if redis_val is not None:
                    rv = redis_val.decode('utf-8') if isinstance(redis_val, (bytes, bytearray)) else redis_val
                    try:
                        return float(rv)
                    except Exception:
                        return default
                return default

            take_profit_pct = _pick_num(override.get('take_profit_pct') or data.get('take_profit_pct'), tp_sym or tp_glob, take_profit_pct)
            stop_loss_pct = _pick_num(override.get('stop_loss_pct') or data.get('stop_loss_pct'), sl_sym or sl_glob, stop_loss_pct)

            if override.get('risk_management_type') or data.get('risk_management_type') or rt:
                raw = override.get('risk_management_type') or data.get('risk_management_type') or rt
                risk_type = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)

            # exit_mode / execution_mode / leverage / limit_config
            raw_exit = override.get('exit_mode') or data.get('exit_mode') or em
            if raw_exit:
                exit_mode = raw_exit.decode('utf-8') if isinstance(raw_exit, (bytes, bytearray)) else str(raw_exit)
            raw_exec = override.get('execution_mode') or data.get('execution_mode') or ex_mode
            if raw_exec:
                execution_mode = raw_exec.decode('utf-8') if isinstance(raw_exec, (bytes, bytearray)) else str(raw_exec)
            raw_lev = override.get('leverage') or data.get('leverage') or lev
            if raw_lev is not None:
                try:
                    leverage_val = max(1, min(5, int(raw_lev.decode('utf-8') if isinstance(raw_lev, (bytes, bytearray)) else str(raw_lev))))
                except Exception:
                    leverage_val = 1
            if override.get('limit_config') or data.get('limit_config') or lc:
                try:
                    limit_config = override.get('limit_config') or data.get('limit_config') or json.loads(lc)  # type: ignore
                except Exception:
                    limit_config = None
        except Exception as e:
            logging.warning(f"[manual_buy] Не удалось прочитать настройки: {e}")

        # 3) Исполнение по execution_mode
        executed_price = 0.0
        risk_orders = {}
        buy_result = None
        try:
            if execution_mode == 'limit_post_only':
                # Запускаем DDD-стратегию с заданным плечом
                from tasks.celery_task_trade import start_execution_strategy
                start_execution_strategy.apply_async(kwargs={
                    'symbol': sym,
                    'execution_mode': 'limit_post_only',
                    'side': 'buy',
                    'qty': float(qty),
                    'limit_config': (limit_config or {}),
                    'leverage': leverage_val
                }, queue='trade')
                buy_result = {"success": True, "action": "limit_post_only_enqueued", "side": "buy"}
                return jsonify({'success': True, 'symbol': sym, 'buy': buy_result, 'risk_orders': {'mode': 'pending_ddd'}}), 200
            else:
                # Market: явно ставим плечо, затем покупка
                try:
                    if hasattr(agent, 'exchange') and hasattr(agent.exchange, 'set_leverage'):
                        try:
                            agent.exchange.set_leverage(leverage_val, sym)  # type: ignore
                        except Exception:
                            try:
                                agent.exchange.set_leverage(str(leverage_val), sym, {'buyLeverage': str(leverage_val), 'sellLeverage': str(leverage_val)})  # type: ignore
                            except Exception:
                                pass
                except Exception:
                    pass

                # Предрасчет TP/SL
                m = agent.exchange.market(sym)  # type: ignore
                tick = None
                try:
                    info = m.get('info', {})
                    pf = info.get('priceFilter', {})
                    tick = float(pf.get('tickSize')) if pf.get('tickSize') else None
                except Exception:
                    tick = None
                if tick is None:
                    precision = m.get('precision', {}).get('price', 2)
                    tick = 10 ** (-precision)
                cur_price = float(agent._get_current_price() or 0.0)  # type: ignore
                def _norm_price(p: float) -> float:
                    try:
                        return round((round(p / tick)) * tick, 8)  # type: ignore
                    except Exception:
                        return float(p)
                tp_price_pre = _norm_price(cur_price * (1.0 + take_profit_pct / 100.0))
                sl_price_pre = _norm_price(cur_price * (1.0 - stop_loss_pct / 100.0))

                attach_params = {
                    'leverage': str(leverage_val),
                    'marginMode': 'isolated',
                }
                if risk_type in ('exchange_orders', 'both'):
                    attach_params.update({
                        'takeProfit': tp_price_pre,
                        'stopLoss': sl_price_pre,
                        'tpTriggerBy': 'LastPrice',
                        'slTriggerBy': 'LastPrice',
                    })

                order = agent.exchange.create_market_buy_order(  # type: ignore
                    sym,
                    float(qty),
                    attach_params,
                )
                buy_result = {
                    'success': True,
                    'symbol': sym,
                    'action': 'buy',
                    'quantity': float(qty),
                    'order': order,
                }
                executed_price = float(order.get('average') or order.get('price') or cur_price or 0.0)
                risk_orders = {'mode': 'attached'}
        except Exception as _ex:
            # Fallback: обычная покупка + отдельные reduceOnly TP/SL
            buy_result = agent.execute_direct_order('buy', symbol=sym, quantity=qty)  # type: ignore
            if not buy_result or not buy_result.get('success'):
                return jsonify({'success': False, 'error': buy_result.get('error') if isinstance(buy_result, dict) else 'buy failed'}), 500
            try:
                order = buy_result.get('order') or {}
                executed_price = order.get('average') or order.get('price')
                if executed_price is None or float(executed_price) <= 0:
                    executed_price = agent._get_current_price()  # type: ignore
                executed_price = float(executed_price)
            except Exception:
                executed_price = 0.0

            if executed_price and executed_price > 0 and risk_type in ('exchange_orders', 'both'):
                try:
                    # Нормализация цен по tickSize
                    m = agent.exchange.market(sym)  # type: ignore
                    tick = None
                    try:
                        info = m.get('info', {})
                        pf = info.get('priceFilter', {})
                        tick = float(pf.get('tickSize')) if pf.get('tickSize') else None
                    except Exception:
                        tick = None
                    if tick is None:
                        precision = m.get('precision', {}).get('price', 2)
                        tick = 10 ** (-precision)
                    def _norm_price2(p: float) -> float:
                        try:
                            return round((round(p / tick)) * tick, 8)  # type: ignore
                        except Exception:
                            return float(p)
                    tp_price = _norm_price2(executed_price * (1.0 + take_profit_pct / 100.0))
                    sl_price = _norm_price2(executed_price * (1.0 - stop_loss_pct / 100.0))
                    amount = float(qty)
                    # TP
                    try:
                        tp_order = agent.exchange.create_limit_sell_order(  # type: ignore
                            sym,
                            amount,
                            tp_price,
                            {
                                'reduceOnly': True,
                                'timeInForce': 'GTC',
                                'postOnly': False,
                            }
                        )
                        risk_orders['take_profit'] = {
                            'order_id': tp_order.get('id'),
                            'price': tp_price,
                            'amount': amount,
                        }
                    except Exception as e:
                        risk_orders['take_profit_error'] = str(e)
                    # SL
                    try:
                        sl_order = agent.exchange.create_stop_market_sell_order(  # type: ignore
                            sym,
                            amount,
                            sl_price,
                            {
                                'reduceOnly': True,
                                'stopPrice': sl_price,
                            }
                        )
                        risk_orders['stop_loss'] = {
                            'order_id': sl_order.get('id'),
                            'stop_price': sl_price,
                            'amount': amount,
                        }
                    except Exception as e:
                        risk_orders['stop_loss_error'] = str(e)
                except Exception as e:
                    risk_orders['error'] = f'risk setup failed: {e}'

        return jsonify({'success': True, 'symbol': sym, 'buy': buy_result, 'executed_price': executed_price, 'risk_orders': risk_orders}), 200
    except Exception as e:
        logging.error(f"[manual_buy] Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/test_order')
def trading_test_order():
    """Мгновенное размещение РЕАЛЬНОГО рыночного ордера BUY/SELL"""
    try:
        data = request.get_json() or {}
        action = (data.get('action') or '').lower()
        symbol = data.get('symbol')
        quantity = data.get('quantity')
        if action not in ('buy', 'sell'):
            return jsonify({'success': False, 'error': "action должен быть 'buy' или 'sell'"}), 400
        client = docker.from_env()
        try:
            container = client.containers.get('medoedai')
            if container.status != 'running':
                return jsonify({'success': False, 'error': f'Контейнер medoedai не запущен. Статус: {container.status}'}), 500
            model_path = None
            try:
                rc = get_redis_client()
                mp = rc.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass
            def _py_str_literal(s):
                if s is None:
                    return 'None'
                return repr(str(s))
            cmd = f'python -c "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path={_py_str_literal(model_path)}); result = agent.execute_direct_order(action={_py_str_literal(action)}, symbol={_py_str_literal(symbol)}, quantity={quantity}); import json; print(\'RESULT: \' + json.dumps(result))"'
            exec_result = container.exec_run(cmd, tty=True)
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except Exception:
                        return jsonify({'success': True, 'message': 'Ордер выполнен', 'output': output}), 200
                else:
                    return jsonify({'success': True, 'message': 'Ордер выполнен', 'output': output}), 200
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                return jsonify({'success': False, 'error': f'Ошибка выполнения ордера: {error_output}'}), 500
        except docker.errors.NotFound:
            return jsonify({'success': False, 'error': 'Контейнер medoedai не найден'}), 500
        except Exception as e:
            return jsonify({'success': False, 'error': f'Ошибка Docker: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/history')
def trading_history():
    """История торговли"""
    try:
        from utils.trade_utils import get_recent_trades, get_trade_statistics
        recent_trades = get_recent_trades(limit=100)
        statistics = get_trade_statistics()
        return jsonify({
            'success': True,
            'trades': recent_trades,
            'statistics': statistics
        }), 200
    except Exception as e:
        logging.error(f"Ошибка получения истории торговли: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/api/trading/regime_config', methods=['GET', 'POST'])
def regime_config():
    """Конфигурация торговых режимов"""
    try:
        rc = get_redis_client()
        if request.method == 'GET':
            config = rc.get('trading:regime_config')
            if config:
                return jsonify({'success': True, 'config': json.loads(config)}), 200
            else:
                return jsonify({'success': True, 'config': {}}), 200
        else:  # POST
            data = request.get_json() or {}
            rc.set('trading:regime_config', json.dumps(data, ensure_ascii=False))
            return jsonify({'success': True, 'message': 'Конфигурация сохранена'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/manual_periodic')
def manual_periodic_trading():
    try:
        # Просто запускаем таск с пустыми аргументами
        task = start_trading_task.apply_async(
            args=[[], None], 
            countdown=0, 
            expires=300, 
            queue='trade'
        )
        
        return jsonify({
            'success': True, 
            'message': 'Периодическая торговая задача запущена вне расписания',
            'task_id': task.id
        }), 200
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

# ==================== ТОРГОВЫЕ СДЕЛКИ API ====================

@trading_bp.get('/api/trades/recent')
def get_recent_trades_api():
    """Получение последних сделок из базы данных"""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = get_recent_trades(limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'trades': trades_data,
            'total': len(trades_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения последних сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/statistics')
def get_trade_statistics_api():
    """Получение статистики торговых сделок"""
    try:
        statistics = get_trade_statistics()
        return jsonify({
            'success': True,
            'statistics': statistics
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения статистики сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/by_symbol/<symbol_name>')
def get_trades_by_symbol_api(symbol_name):
    """Получение сделок по конкретному символу"""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = get_trades_by_symbol(symbol_name, limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol_name,
            'trades': trades_data,
            'total': len(trades_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения сделок по символу {symbol_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/matched_full')
def get_matched_full_trades():
    """Получение полных совпавших сделок с предсказаниями"""
    try:
        def unify_symbol(s: str) -> str:
            return s.replace('USDT', '').replace('USDC', '').upper()
        
        def to_ms(v):
            try:
                if isinstance(v, str):
                    from datetime import datetime
                    return int(datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp() * 1000)
                return int(v)
            except Exception:
                return 0
        
        def bucket_5m(ms: int):
            return (ms // (5 * 60 * 1000)) * (5 * 60 * 1000)
        
        def pick_pred(bkt: int, sym_u: str, typ: str):
            try:
                rc = get_redis_client()
                key = f'pred:{sym_u}:{bkt}:{typ}'
                raw = rc.get(key)
                if raw:
                    return json.loads(raw.decode('utf-8'))
            except Exception:
                pass
            return None
        
        # Получаем параметры запроса
        symbol = request.args.get('symbol', 'BTCUSDT')
        action = request.args.get('action', 'buy')
        limit = int(request.args.get('limit', 100))
        
        # Получаем сделки
        trades = get_recent_trades(limit=limit * 2)  # Берем больше, чтобы отфильтровать
        
        # Фильтруем по символу и действию
        filtered_trades = []
        for trade in trades:
            if (trade.symbol and trade.symbol.name == symbol and 
                trade.action == action and trade.is_successful):
                filtered_trades.append(trade)
        
        # Ограничиваем результат
        filtered_trades = filtered_trades[:limit]
        
        # Обогащаем данными предсказаний
        enriched_trades = []
        for trade in filtered_trades:
            trade_data = {
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            }
            
            # Добавляем данные предсказания
            if trade.created_at:
                ms = to_ms(trade.created_at.isoformat())
                bkt = bucket_5m(ms)
                sym_u = unify_symbol(symbol)
                pred = pick_pred(bkt, sym_u, action)
                
                if pred:
                    trade_data['prediction'] = pred
                    trade_data['prediction_bucket'] = bkt
                    trade_data['prediction_symbol'] = sym_u
            
            enriched_trades.append(trade_data)
        
        return jsonify({
            'success': True,
            'trades': enriched_trades,
            'total': len(enriched_trades),
            'symbol': symbol,
            'action': action
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения полных совпавших сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500