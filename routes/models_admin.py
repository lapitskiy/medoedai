from flask import Blueprint, jsonify, request, current_app
from celery.result import AsyncResult
from tasks import celery
from pathlib import Path
import os
from utils.config_loader import get_config_value
import json
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
from agents.vdqn.cfg.vconfig import vDqnConfig
import pickle
import numpy as np
import pandas as pd
import requests
from redis import Redis
from datetime import datetime

models_admin_bp = Blueprint('models_admin', __name__)

@models_admin_bp.get('/api/runs/symbols')
def api_runs_symbols():
    try:
        base = Path('result')
        if not base.exists():
            return jsonify({'success': True, 'symbols': []})
        symbols = []
        for d in base.iterdir():
            if d.is_dir() and (d / 'runs').exists():
                # проверим наличие хотя бы одного run
                has_run = any((d / 'runs').iterdir())
                if has_run:
                    symbols.append(d.name)
        symbols.sort()
        return jsonify({'success': True, 'symbols': symbols})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/api/strategies/save')
def api_strategy_save():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        code = (data.get('code') or '').strip()
        label = (data.get('label') or '').strip()
        params = data.get('params') or {}
        metrics = data.get('metrics') or {}
        set_current = bool(data.get('set_current') or False)

        # Определяем run_dir из filename или code
        run_dir = None
        symbol = None
        run_id = None
        if filename:
            p = Path(filename.replace('\\','/')).resolve()
            if p.name == 'model.pth' and 'runs' in p.parts:
                runs_idx = list(p.parts).index('runs')
                run_id = p.parts[runs_idx+1] if runs_idx+1 < len(p.parts) else None
                symbol = p.parts[runs_idx-1] if runs_idx-1 >= 0 else None
                run_dir = Path('result')/str(symbol)/'runs'/str(run_id)
        if run_dir is None or not run_dir.exists():
            return jsonify({'success': False, 'error': 'cannot resolve run directory'}), 400

        # Совместим сохранение стратегий с oos_results.json: складываем в раздел 'strategies'
        stg_file = run_dir / 'oos_results.json'
        payload = {
            'label': label,
            'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'params': params,
            'metrics': metrics,
            'symbol': str(symbol),
            'run_id': str(run_id),
        }
        stg_id = f"stg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        # Читаем/пишем файл oos_results.json
        doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
        if stg_file.exists():
            try:
                doc = json.loads(stg_file.read_text(encoding='utf-8'))
                if not isinstance(doc, dict):
                    doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
            except Exception:
                doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
        if 'strategies' not in doc or not isinstance(doc['strategies'], dict):
            doc['strategies'] = {'current': None, 'items': {}}
        items = doc['strategies'].get('items') or {}
        items[stg_id] = payload
        doc['strategies']['items'] = items
        if set_current:
            doc['strategies']['current'] = stg_id
        stg_file.parent.mkdir(parents=True, exist_ok=True)
        stg_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        return jsonify({'success': True, 'stg_id': stg_id, 'file': str(stg_file)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/api/strategies/apply')
def api_strategy_apply():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        stg_id = (data.get('stg_id') or '').strip()
        if not filename or not stg_id:
            return jsonify({'success': False, 'error': 'filename and stg_id required'}), 400
        p = Path(filename.replace('\\','/')).resolve()
        if not (p.name == 'model.pth' and 'runs' in p.parts):
            return jsonify({'success': False, 'error': 'bad run path'}), 400
        runs_idx = list(p.parts).index('runs')
        run_id = p.parts[runs_idx+1] if runs_idx+1 < len(p.parts) else None
        symbol = p.parts[runs_idx-1] if runs_idx-1 >= 0 else None
        run_dir = Path('result')/str(symbol)/'runs'/str(run_id)
        stg_file = run_dir / 'oos_results.json'
        if not stg_file.exists():
            return jsonify({'success': False, 'error': 'oos_results.json not found'}), 404
        doc = json.loads(stg_file.read_text(encoding='utf-8'))
        strategies = (doc.get('strategies') or {})
        items = (strategies.get('items') or {})
        item = items.get(stg_id)
        if not item:
            return jsonify({'success': False, 'error': 'strategy not found'}), 404
        # Обновим current
        if 'strategies' not in doc or not isinstance(doc['strategies'], dict):
            doc['strategies'] = {'current': None, 'items': {}}
        doc['strategies']['current'] = stg_id
        stg_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        # Применим qgate в Redis (если доступен)
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
            params = item.get('params') or {}
            t1 = float(params.get('t1_base') or 0)
            t2 = float(params.get('t2_base') or 0)
            # gate_scale/gate_disable можно учесть на стороне трейдинга, пока сохранём базовые
            rc.set('trading:qgate', json.dumps({'T1': t1, 'T2': t2}, ensure_ascii=False))
        except Exception:
            pass
        return jsonify({'success': True, 'current': stg_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.get('/api/runs/list')
def api_runs_list():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400
        runs_dir = Path('result') / symbol / 'runs'
        if not runs_dir.exists():
            return jsonify({'success': True, 'runs': []})
        runs = []
        for rd in runs_dir.iterdir():
            if not rd.is_dir():
                continue
            run_id = rd.name
            manifest = {}
            mf = rd / 'manifest.json'
            if mf.exists():
                try:
                    manifest = json.loads(mf.read_text(encoding='utf-8'))
                except Exception:
                    manifest = {}
            model_path = str(rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = str(rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = str(rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None
            # Попытаемся извлечь краткую статистику
            winrate = None; pl_ratio = None; episodes = None
            if result_path:
                try:
                    import pickle as _pkl
                    with open(result_path,'rb') as _f:
                        _res = _pkl.load(_f)
                    _fs = _res.get('final_stats') or {}
                    winrate = _fs.get('winrate')
                    pl_ratio = _fs.get('pl_ratio')
                    episodes = _res.get('actual_episodes', _res.get('episodes'))
                except Exception:
                    pass
            runs.append({
                'run_id': run_id,
                'parent_run_id': manifest.get('parent_run_id'),
                'root_id': manifest.get('root_id'),
                'seed': manifest.get('seed'),
                'episodes_end': manifest.get('episodes_end'),
                'created_at': manifest.get('created_at'),
                'model_path': model_path,
                'replay_path': replay_path,
                'result_path': result_path,
                'winrate': winrate,
                'pl_ratio': pl_ratio,
                'episodes': episodes,
            })
        # Простая сортировка по created_at, затем по run_id
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        return jsonify({'success': True, 'runs': runs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.get('/api/runs/oos_results')
def api_runs_oos_results():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()
        if not symbol or not run_id:
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400
        oos_file = Path('result') / symbol / 'runs' / run_id / 'oos_results.json'
        if not oos_file.exists():
            return jsonify({'success': False, 'error': 'oos_results.json not found'}), 404
        try:
            data = json.loads(oos_file.read_text(encoding='utf-8'))
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to parse oos_results.json: {e}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.post('/api/runs/delete')
def api_runs_delete():
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        run_id = (data.get('run_id') or '').strip()
        if not symbol or not run_id:
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400
        run_dir = Path('result') / symbol / 'runs' / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            return jsonify({'success': False, 'error': 'run directory not found'}), 404
        deleted = []
        errors = []
        # Удаляем все файлы в директории run
        try:
            for p in run_dir.iterdir():
                if p.is_file():
                    try:
                        os.remove(p)
                        deleted.append(str(p))
                    except Exception as e:
                        errors.append(f'{p.name}: {e}')
        except Exception as e:
            errors.append(f'iterdir: {e}')
        # Пытаемся удалить директорию, если пуста
        removed_dir = False
        try:
            if run_dir.exists() and len(list(run_dir.iterdir())) == 0:
                run_dir.rmdir()
                removed_dir = True
        except Exception:
            removed_dir = False
        return jsonify({'success': True, 'deleted': deleted, 'removed_dir': removed_dir, 'errors': errors})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.post('/oos_test_model')
def oos_test_model():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        code = (data.get('code') or '').strip()
        days = int(data.get('days') or 30)
        # Стартовый капитал для расчёта эквити/просадки
        try:
            start_capital = float(data.get('start_capital') or get_config_value('OOS_START_CAPITAL', 10000))
        except Exception:
            start_capital = 10000.0

        # Комиссии: по умолчанию учитываем taker 0.055% на сторону
        fee_enabled = bool(data.get('fee_enabled') if data.get('fee_enabled') is not None else True)
        try:
            fee_rate = data.get('fee_rate')
            # fee_rate может прийти в процентах (например 0.055) или доле (0.00055)
            if fee_rate is None or fee_rate == '':
                fee_rate = 0.00055  # 0.055% за сторону
            else:
                fee_rate = float(fee_rate)
                # Если пользователь передал как проценты (> 0.01), конвертируем в долю
                if fee_rate > 0.01:
                    fee_rate = fee_rate / 100.0
            if fee_rate < 0:
                fee_rate = 0.0
        except Exception:
            fee_rate = 0.00055
        # Управление Q‑gate через UI: disable или понижающий коэффициент (0..1)
        gate_disable = bool(data.get('gate_disable') or False)
        gate_scale = None
        try:
            if data.get('gate_scale') is not None:
                gate_scale = float(data.get('gate_scale'))
        except Exception:
            gate_scale = None
        try:
            current_app.logger.info(f"[OOS] request filename='{filename}' code='{code}' days={days}")
        except Exception:
            pass

        # Нормализуем код/путь
        if not filename and not code:
            return jsonify({'success': False, 'error': 'filename or code required'}), 400
        if not code and filename:
            base = os.path.basename(filename).replace('\\', '/').split('/')[-1]
            if base.startswith('dqn_model_') and base.endswith('.pth'):
                code = base[len('dqn_model_'):-len('.pth')]

        # Прямой синхронный расчёт простого OOS (без Celery, чтобы быстрее показать результат)
        # Используем логику, схожую с execute_trade: готовим state по БД и дергаем serving последовательно.
        from utils.db_utils import db_get_or_fetch_ohlcv

        # Определяем символ максимально надёжно
        symbol = 'BTCUSDT'
        symbol_resolved = False
        try:
            # 1) Если передан путь runs/.../model.pth — читаем symbol из manifest.json рядом
            if filename:
                p = Path(filename.replace('\\', '/'))
                try:
                    p = p.resolve()
                except Exception:
                    pass
                if p.exists() and p.is_file() and p.name == 'model.pth' and 'runs' in p.parts:
                    manifest_path = p.parent / 'manifest.json'
                    # 1a) Пытаемся взять из manifest.json
                    try:
                        if manifest_path.exists():
                            mf = json.loads(manifest_path.read_text(encoding='utf-8'))
                            base = str((mf.get('symbol') or '')).upper().replace('/', '')
                            if base:
                                symbol = base if base.endswith('USDT') else (base + 'USDT')
                                symbol_resolved = True
                    except Exception:
                        pass
                    # 1b) Если не удалось — берём символ из сегмента пути перед 'runs'
                    if not symbol_resolved:
                        parts = list(p.parts)
                        try:
                            runs_idx = parts.index('runs')
                            if runs_idx - 1 >= 0:
                                base = str(parts[runs_idx - 1]).upper().replace('/', '')
                                if base:
                                    symbol = base if base.endswith('USDT') else (base + 'USDT')
                                    symbol_resolved = True
                        except Exception:
                            pass
            # 2) Фоллбек по коду dqn_model_<code>.pth → prefix до '_'
            if (not symbol_resolved) and code:
                base = code.split('_')[0].upper()
                if base and len(base) <= 6:
                    symbol = base + 'USDT'
            # 3) Если всё ещё по умолчанию или base выглядит как run_id (например 4AB3),
            #    попробуем найти run по коду и взять символ из каталога result/<SYMBOL>/runs/<code>
            if (not symbol_resolved) and code:
                try:
                    runs_root = Path('result')
                    if runs_root.exists():
                        for sym_dir in runs_root.iterdir():
                            try:
                                rdir = sym_dir / 'runs' / code
                                if (rdir / 'model.pth').exists():
                                    sym = str(sym_dir.name).upper()
                                    symbol = sym if sym.endswith('USDT') else (sym + 'USDT')
                                    symbol_resolved = True
                                    break
                            except Exception:
                                continue
                except Exception:
                    pass
        except Exception:
            pass
        try:
            current_app.logger.info(f"[OOS] resolved symbol={symbol}")
        except Exception:
            pass

        # Пользователь может указать биржу; по умолчанию bybit. При отсутствии данных делаем фолбэк на альтернативу
        exchange_req = str(data.get('exchange') or 'bybit').lower().strip()
        if exchange_req not in ('bybit', 'binance'):
            exchange_req = 'bybit'
        primary_exchange = exchange_req
        fallback_exchange = 'binance' if primary_exchange == 'bybit' else 'bybit'
        try:
            current_app.logger.info(f"[OOS] using exchange={primary_exchange} (fallback={fallback_exchange}) for {symbol}")
        except Exception:
            pass
        df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id=primary_exchange)
        if df is None or df.empty:
            try:
                current_app.logger.warning(f"[OOS] no candles from {primary_exchange} for {symbol}, trying {fallback_exchange}")
            except Exception:
                pass
            df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id=fallback_exchange)
        if df is None or df.empty:
            try:
                current_app.logger.error(f"[OOS] no candles for {symbol}")
            except Exception:
                pass
            return jsonify({'success': False, 'error': f'no candles for {symbol}'}), 400
        # === Читаем train_result.pkl для получения config, lookback, indicators ===
        train_result_path = None
        if filename:
            p = Path(filename.replace('\\','/')).resolve()
            if p.name == 'model.pth' and 'runs' in p.parts:
                train_result_path = p.parent / 'train_result.pkl'
        
        gym_config_from_train = None
        vdqn_config_from_train = None
        adaptive_normalization_params = None
        if train_result_path and train_result_path.exists():
            try:
                with open(train_result_path, 'rb') as f:
                    train_data = pickle.load(f)
                    gym_config_from_train = train_data.get('gym_snapshot')
                    vdqn_config_from_train = train_data.get('cfg_snapshot')
                    adaptive_normalization_params = train_data.get('adaptive_normalization')
            except Exception as e:
                current_app.logger.warning(f"[OOS] failed to load train_result.pkl for {filename}: {e}")

        # Берём последние N дней (до предыдущей закрытой свечи)
        try:
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            cutoff = df['dt'].max().floor('5min') - pd.Timedelta(minutes=5)
            start_ts = cutoff - pd.Timedelta(days=int(max(1, days)))
            df = df[(df['dt'] > start_ts) & (df['dt'] <= cutoff)]
        except Exception:
            pass
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'no oos window'}), 400

        # Один раз готовим полные данные для env
        df_5min = df.copy().reset_index(drop=True)

        df_15min = df_5min.resample('15min', on='dt').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

        df_1h = df_5min.resample('1h', on='dt').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()

        # Конвертируем в NumPy
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df_5min_np = df_5min[ohlcv_cols].values.astype(np.float32)
        df_15min_np = df_15min[ohlcv_cols].values.astype(np.float32)
        df_1h_np = df_1h[ohlcv_cols].values.astype(np.float32)

        # Индикаторы из train_result.pkl, если есть
        if gym_config_from_train and gym_config_from_train.get('indicators_config'):
            indicators_config = gym_config_from_train['indicators_config']
        else:
            indicators_config = {
                'rsi': {'length': 14},
                'ema': {'lengths': [100, 200]},
                'ema_cross': {'pairs': [(100, 200)], 'include_cross_signal': True},
                'sma': {'length': 14}
            }
        indicators = preprocess_dataframes(df_5min_np, df_15min_np, df_1h_np, indicators_config)

        # --- Диагностика индикаторов: проверяем NaN / Inf ------------------
        n_nan = 0
        n_inf = 0
        try:
            import numpy as _np
            n_nan = int(_np.isnan(indicators).sum())
            n_inf = int(_np.isinf(indicators).sum())
            if n_nan or n_inf:
                current_app.logger.warning(f"[OOS] ⚠️ indicators NaN={n_nan} Inf={n_inf}")
                # Не бросаем исключение, просто логируем
            else:
                current_app.logger.info("[OOS] ✅ indicators: no NaN/Inf")
        except Exception as diag_err:
            try:
                current_app.logger.warning(f"[OOS] indicator diagnostics failed: {diag_err}")
            except Exception:
                pass
        # --------------------------------------------------------------------

        # Создаём env ОДИН РАЗ
        # Используем cfg_snapshot из train_result, если есть
        if vdqn_config_from_train:
            # Создаем экземпляр vDqnConfig и обновляем его атрибуты
            cfg = vDqnConfig()
            for k, v in vdqn_config_from_train.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            cfg = vDqnConfig() # Fallback к дефолтному конфигу
        
        # lookback_window из train_result.pkl, если есть
        lookback_window = 20 # Дефолт, если не найдем
        if gym_config_from_train and gym_config_from_train.get('lookback_window'):
            lookback_window = gym_config_from_train['lookback_window']
        
        # Загружаем normalization_stats из train_result.pkl, если есть
        normalization_stats = None
        try:
            if train_result_path and isinstance(train_data, dict):
                # Попытка извлечь из best_model или model чекпойнта, если он был включён в результаты
                # В базовой версии train_result не содержит нормализацию, поэтому используем из самого чекпойнта модели
                ckpt_path = None
                # Если filename задан, используем его
                if filename:
                    ckpt_path = filename
                else:
                    # попытка найти model.pth рядом (уже вычислено ниже, но нам нужно раньше)
                    cand = Path('result')/symbol.replace('USDT','')/'runs'/code/'model.pth'
                    if cand.exists():
                        ckpt_path = str(cand)
                if ckpt_path and os.path.exists(ckpt_path):
                    try:
                        import torch
                        ckpt = torch.load(ckpt_path, map_location='cpu')
                        if isinstance(ckpt, dict) and 'normalization_stats' in ckpt:
                            normalization_stats = ckpt.get('normalization_stats')
                            current_app.logger.info(f"[OOS] Loaded normalization_stats from checkpoint.")
                        else:
                            current_app.logger.info(f"[OOS] normalization_stats not found in checkpoint.")
                    except Exception as e:
                        current_app.logger.warning(f"[OOS] Failed to load normalization_stats from checkpoint {ckpt_path}: {e}")
                        normalization_stats = None
        except Exception as e:
            current_app.logger.warning(f"[OOS] Failed to attempt loading normalization_stats: {e}")
            normalization_stats = None

        env = CryptoTradingEnvOptimized(
            dfs={'df_5min': df_5min, 'df_15min': df_15min, 'df_1h': df_1h, 'symbol': symbol},
            cfg=cfg,
            lookback_window=lookback_window,
            normalization_stats=normalization_stats
        )

        # Применяем adaptive_normalization, если она была в обучении
        if adaptive_normalization_params:
            if 'base_profile' in adaptive_normalization_params and hasattr(env, 'risk_management'):
                # Обновляем параметры risk_management в cfg.gym_env
                for k, v in adaptive_normalization_params['base_profile'].items():
                    if hasattr(env.risk_management, k):
                        setattr(env.risk_management, k, v)
            if 'adapted_params' in adaptive_normalization_params and hasattr(env, 'risk_management'):
                for k, v in adaptive_normalization_params['adapted_params'].items():
                    if hasattr(env.risk_management, k):
                        setattr(env.risk_management, k, v)
        
        # === ВОССТАНАВЛИВАЕМ ИНИЦИАЛИЗАЦИЮ ПЕРЕМЕННЫХ, КОТОРЫЕ НУЖНЫ ДАЛЬШЕ ===
        serving_url = os.environ.get('SERVING_URL', 'http://serving:8000/predict_ensemble')

        # Путь к модели (или пытаемся найти в result/ по run-коду)
        model_paths = []
        if filename:
            model_paths = [filename]
        else:
            cand = Path('result')/symbol.replace('USDT','')/'runs'/code/'model.pth'
            if cand.exists():
                model_paths = [str(cand)]
        if not model_paths:
            return jsonify({'success': False, 'error': 'model file not found'}), 400
        # Абсолютные пути
        abs_list = []
        for mp in model_paths:
            pp = Path(str(mp).replace('\\','/'))
            if not pp.is_absolute():
                pp = (Path.cwd()/pp).resolve()
            abs_list.append(str(pp))
        model_paths = abs_list

        # Читаем consensus из Redis (как в бою)
        consensus_cfg = None
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
            raw = rc.get('trading:consensus')
            if raw:
                consensus_cfg = json.loads(raw)
        except Exception:
            consensus_cfg = None

        # Q-gate базовые пороги
        try:
            T1 = float(os.environ.get('QGATE_T1', '0.35'))
        except Exception:
            T1 = 0.35
        try:
            T2 = float(os.environ.get('QGATE_T2', '0.25'))
        except Exception:
            T2 = 0.25
        try:
            flat_factor = float(os.environ.get('QGATE_FLAT', '1.0'))
        except Exception:
            flat_factor = 1.0
        if gate_disable:
            T1_user = 0.0; T2_user = 0.0
        else:
            if isinstance(gate_scale, (int,float)) and gate_scale>=0:
                T1_user = max(0.0, T1*(1.0-float(gate_scale)))
                T2_user = max(0.0, T2*(1.0-float(gate_scale)))
            else:
                T1_user = T1; T2_user = T2

        # Счётчики результата
        pnl_total = 0.0; wins=0; losses=0; trades=0
        peak = float(start_capital); equity = float(start_capital); max_dd=0.0
        position=None; entry_price=None; entry_ts=None
        trades_details = []
        decisions_counts = {'buy':0,'sell':0,'hold':0}

        # Локальная функция: определяем режим рынка из последних свечей
        def _compute_regime_local(df_prices: pd.DataFrame):
            try:
                closes_full = df_prices['close'].astype(float).values
                if closes_full.size < 310:
                    return 'flat', {'windows':[60,180,300],'labels':['flat','flat','flat'],'weights':[1,1,1],'voting':'majority','tie_break':'flat'}
                windows = [60,180,300]
                labels = []
                weights = [1,1,1]
                drift_thr = 0.002
                vol_flat_thr = 0.0025
                for w in windows:
                    c = closes_full[-w:]
                    start = float(c[0]); end = float(c[-1])
                    drift = (end - start) / max(start, 1e-9)
                    rr = np.diff(c) / np.maximum(c[:-1], 1e-9)
                    vol = float(np.std(rr))
                    if abs(drift) < (0.75 * drift_thr) and vol < vol_flat_thr:
                        labels.append('flat')
                    elif drift >= drift_thr:
                        labels.append('uptrend')
                    elif drift <= -drift_thr:
                        labels.append('downtrend')
                    else:
                        labels.append('flat')
                counts = {'flat': labels.count('flat'), 'uptrend': labels.count('uptrend'), 'downtrend': labels.count('downtrend')}
                winner = max(counts, key=counts.get)
                return winner, {'windows':windows,'labels':labels,'weights':[1,1,1],'voting':'majority','tie_break':'flat'}
            except Exception:
                return 'flat', {'windows':[60,180,300],'labels':['flat','flat','flat'],'weights':[1,1,1],'voting':'majority','tie_break':'flat'}

        closes = df['close'].astype(float).values
        for i in range(120, len(df)):
            try:
                env.current_step = i
                state = env._get_state()
                if state is None:
                    continue
                current_app.logger.info(f"[OOS] Raw state at step {i}: {state.tolist()[:10]}...") # Логируем первые 10 элементов для краткости
                state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).tolist()  # Конвертируем в list без NaN/Inf
                current_app.logger.info(f"[OOS] Processed state at step {i}: {state[:10]}...") # Логируем первые 10 элементов обработанного state
            except Exception as e:
                current_app.logger.error(f"[OOS] state get error at step {i}: {e}")
                continue

            # Вычислим режим рынка и применим q‑gate факторы
            regime, regime_details = _compute_regime_local(df.iloc[:i])
            base_T1 = T1_user
            base_T2 = T2_user
            eff_T1 = base_T1 * (flat_factor if (regime == 'flat') else 1.0)
            eff_T2 = base_T2 * (flat_factor if (regime == 'flat') else 1.0)
            payload = {"state": state, "model_paths": model_paths, "symbol": symbol, "consensus": (consensus_cfg or {}), "market_regime": regime, "market_regime_details": regime_details}
            
            try:
                resp = requests.post(serving_url, json=payload, timeout=15)
                if not resp.ok:
                    current_app.logger.error(f"[OOS] serving error status={resp.status_code} body={resp.text[:400]}")
                    continue
                pj = resp.json()
            except Exception as srv_err:
                current_app.logger.error(f"[OOS] serving request error: {srv_err}")
                continue

            if not pj.get('success'):
                continue
            # Применим тот же консенсус и Q‑gate, что и в бою (упрощённо)
            decision = pj.get('decision') or 'hold'
            preds_list = pj.get('predictions') or []
            
            # Диагностика: логируем сырое предсказание модели
            if preds_list and len(preds_list) > 0:
                for idx, p in enumerate(preds_list):
                    raw_action = p.get('action', 'hold')
                    raw_qv = p.get('q_values', [])
                    try:
                        current_app.logger.info(f"[OOS] RAW model[{idx}] prediction: action={raw_action} q_values={raw_qv}")
                    except Exception:
                        pass
            if preds_list:
                # Пер‑модельный Q‑gate и голоса
                votes = {'buy':0,'sell':0,'hold':0}
                total_sel = len(preds_list)
                for p in preds_list:
                    act = str(p.get('action') or 'hold').lower()
                    qv = p.get('q_values') or []
                    gate_ok = True
                    q_max = None; q_gap = None
                    try:
                        if isinstance(qv, list) and len(qv) >= 3:
                            if act == 'buy':
                                qb = float(qv[1]); other = max(float(qv[0]), float(qv[2]))
                                gate_ok = (qb >= eff_T1) and ((qb - other) >= eff_T2)
                                q_max = qb; q_gap = qb - other
                            elif act == 'sell':
                                qs = float(qv[2]); other = max(float(qv[0]), float(qv[1]))
                                gate_ok = (qs >= eff_T1) and ((qs - other) >= eff_T2)
                                q_max = qs; q_gap = qs - other
                    except Exception:
                        gate_ok = (act == 'hold')
                    if act in ('buy','sell'):
                        if gate_ok:
                            votes[act] += 1
                    else:
                        votes['hold'] += 1
                    # Лог по каждой модели
                    try:
                        current_app.logger.info(
                            f"[OOS] model action={act} gate_ok={gate_ok} q_max={None if q_max is None else round(q_max,3)} gap={None if q_gap is None else round(q_gap,3)} T1={round(eff_T1,3)} T2={round(eff_T2,3)} qv={[(round(float(x),3) if isinstance(x,(int,float)) else x) for x in (qv if isinstance(qv,list) else [])]}"
                        )
                    except Exception:
                        pass
                # Требования из consensus
                req_flat = None; req_trend = None
                if isinstance(consensus_cfg, dict):
                    counts = (consensus_cfg.get('counts') or {})
                    perc = (consensus_cfg.get('percents') or {})
                    if isinstance(counts.get('flat'), (int,float)):
                        req_flat = int(max(1, counts.get('flat')))
                    if isinstance(counts.get('trend'), (int,float)):
                        req_trend = int(max(1, counts.get('trend')))
                    if req_flat is None and isinstance(perc.get('flat'), (int,float)):
                        req_flat = int(max(1, int(np.ceil(total_sel * (float(perc.get('flat'))/100.0)))))
                    if req_trend is None and isinstance(perc.get('trend'), (int,float)):
                        req_trend = int(max(1, int(np.ceil(total_sel * (float(perc.get('trend'))/100.0)))))
                # Для OOS тестирования с одной моделью не требуем консенсус
                if total_sel == 1:
                    req_flat = 1
                    req_trend = 1
                else:
                    if req_flat is None: req_flat = total_sel
                    if req_trend is None: req_trend = total_sel
                required = req_trend if regime in ('uptrend','downtrend') else req_flat
                if votes['buy'] >= required and votes['buy'] > votes['sell']:
                    decision = 'buy'
                elif votes['sell'] >= required and votes['sell'] > votes['buy']:
                    decision = 'sell'
                else:
                    decision = 'hold'
                # Итоговое голосование/решение
                try:
                    current_app.logger.info(f"[OOS] regime={regime} votes={votes} required={required} decision={decision}")
                except Exception:
                    pass
            # Копим сводку по решениям за окно
            try:
                if isinstance(decision, str) and decision in decisions_counts:
                    decisions_counts[decision] += 1
            except Exception:
                pass
            # Итоговое решение есть — симулируем сделку
            price = float(closes[i])
            ts_ms = int(df.iloc[i]['timestamp']) if 'timestamp' in df.columns else None
            try:
                ts_iso = datetime.fromtimestamp(ts_ms / 1000).isoformat() if ts_ms else None
            except Exception:
                ts_iso = None
            # Простая симуляция: buy -> открыть лонг; sell -> закрыть, если был лонг (profit/loss)
            if decision == 'buy' and position is None:
                position = 'long'; entry_price = price; entry_ts = ts_iso
                try:
                    current_app.logger.info(f"[OOS] BUY open ts={ts_iso} price={price}")
                except Exception:
                    pass
            elif decision == 'sell' and position == 'long':
                pl_gross = price - float(entry_price)
                # Комиссии за вход и выход (за сторону): notional = price * 1.0 условных единиц
                fee_entry = (float(entry_price) * fee_rate) if fee_enabled else 0.0
                fee_exit = (float(price) * fee_rate) if fee_enabled else 0.0
                pl_net = pl_gross - fee_entry - fee_exit
                pnl_total += pl_net
                trades += 1
                wins += 1 if pl_net > 0 else 0
                losses += 1 if pl_net <= 0 else 0
                try:
                    trade_rec = {
                        'entry_ts': entry_ts,
                        'entry_price': float(entry_price),
                        'exit_ts': ts_iso,
                        'exit_price': float(price),
                        'pnl_gross': float(pl_gross),
                        'fee_entry': float(fee_entry),
                        'fee_exit': float(fee_exit),
                        'pnl': float(pl_net)
                    }
                    trades_details.append(trade_rec)
                    current_app.logger.info(f"[OOS] SELL close ts={ts_iso} price={price} pnl_net={pl_net} (gross={pl_gross} fee_in={fee_entry} fee_out={fee_exit})")
                except Exception:
                    pass
                position = None; entry_price = None; entry_ts = None
            # Эквити и maxDD
            equity = float(start_capital) + float(pnl_total)
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

        # !!! Добавляем логику принудительного закрытия позиции в конце OOS-окна !!!
        if position == 'long':
            # Закрываем по последней цене доступной свечи
            final_price = float(closes[-1])
            final_ts_ms = int(df.iloc[len(df)-1]['timestamp']) if 'timestamp' in df.columns else None
            try:
                final_ts_iso = datetime.fromtimestamp(final_ts_ms / 1000).isoformat() if final_ts_ms else None
            except Exception:
                final_ts_iso = None

            pl_gross = final_price - float(entry_price)
            fee_entry = (float(entry_price) * fee_rate) if fee_enabled else 0.0
            fee_exit = (float(final_price) * fee_rate) if fee_enabled else 0.0
            pl_net = pl_gross - fee_entry - fee_exit

            pnl_total += pl_net
            trades += 1
            wins += 1 if pl_net > 0 else 0
            losses += 1 if pl_net <= 0 else 0

            try:
                trade_rec = {
                    'entry_ts': entry_ts,
                    'entry_price': float(entry_price),
                    'exit_ts': final_ts_iso,
                    'exit_price': float(final_price),
                    'pnl_gross': float(pl_gross),
                    'fee_entry': float(fee_entry),
                    'fee_exit': float(fee_exit),
                    'pnl': float(pl_net)
                }
                trades_details.append(trade_rec)
                current_app.logger.info(f"[OOS] FORCED SELL close ts={final_ts_iso} price={final_price} pnl_net={pl_net} (gross={pl_gross} fee_in={fee_entry} fee_out={fee_exit})")
            except Exception:
                pass
        # Конец логики принудительного закрытия

        winrate = (wins / trades) if trades > 0 else None
        pl_ratio = None
        try:
            if wins > 0 and losses > 0:
                # Рассчитываем средние прибыль и убыток из деталей сделок
                winning_trades = [t for t in trades_details if t.get('pnl', 0) > 0]
                losing_trades = [t for t in trades_details if t.get('pnl', 0) <= 0]
                if winning_trades and losing_trades:
                    avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
                    avg_loss = abs(sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades))
                    pl_ratio = (avg_win / avg_loss) if avg_loss > 1e-12 else None
            elif wins > 0 and losses == 0:
                # Все сделки прибыльные - PLR = бесконечность, но показываем как очень высокий
                pl_ratio = 999.99
        except Exception:
            pl_ratio = None
        profit_factor = None
        try:
            if trades > 0:
                # Рассчитываем из деталей сделок
                gross_profit = sum(max(0, t.get('pnl', 0)) for t in trades_details)
                gross_loss = abs(sum(min(0, t.get('pnl', 0)) for t in trades_details))
                if gross_loss > 1e-12:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    # Только прибыльные сделки - PF = бесконечность, но показываем как очень высокий
                    profit_factor = 999.99
        except Exception:
            profit_factor = None

        # Классификация результата для агрегирования в oos_results.json
        try:
            roi_pct = ((pnl_total / float(start_capital)) * 100.0) if (start_capital and isinstance(pnl_total, (int, float))) else None
        except Exception:
            roi_pct = None
        is_good = (roi_pct is not None and roi_pct > 10.0) \
            and (profit_factor is None or (isinstance(profit_factor, (int, float)) and profit_factor >= 1.2)) \
            and (pl_ratio is None or (isinstance(pl_ratio, (int, float)) and pl_ratio > 1.0)) \
            and (max_dd is None or (isinstance(max_dd, (int, float)) and max_dd < 0.40))
        is_bad = (roi_pct is not None and roi_pct <= 0.0) \
            or (isinstance(profit_factor, (int, float)) and profit_factor < 1.0) \
            or (isinstance(pl_ratio, (int, float)) and pl_ratio < 1.0) \
            or (isinstance(max_dd, (int, float)) and max_dd > 0.70)
        classification = 'good' if is_good else ('bad' if is_bad else 'neutral')

        nan_count = n_nan
        inf_count = n_inf
        result_payload = {
            'success': True,
            'result': {
                'winrate': winrate,
                'pl_ratio': pl_ratio,
                'profit_factor': profit_factor,
                'max_dd': max_dd,
                'pnl_total': pnl_total,
                'trades': trades,
                'symbol': symbol,
                'days': days,
                'gate_disable': bool(gate_disable),
                'gate_scale': (float(gate_scale) if isinstance(gate_scale, (int, float)) else None),
                't1_base': float(T1_user),
                't2_base': float(T2_user),
                'start_capital': float(start_capital),
                'equity_end': float(equity),
                'classification': classification,
                'nan_indicators': nan_count,
                'inf_indicators': inf_count,
            }
        }
        # Определяем run_id для сохранения trades_details и oos_results
        _run_id = None
        _symbol_for_run_dir = symbol.replace('USDT','') # Используем символ без USDT для пути
        # Попробуем извлечь из filename
        if filename:
            pth = Path(str(filename).replace('\\','/')).resolve()
            if pth.name == 'model.pth' and 'runs' in pth.parts:
                runs_idx = list(pth.parts).index('runs')
                _run_id = pth.parts[runs_idx+1] if runs_idx+1 < len(pth.parts) else None
                # Предпочтем символ из пути, если он есть
                try:
                    _symbol_from_path = pth.parts[runs_idx-1] if runs_idx-1 >= 0 else None
                    if _symbol_from_path:
                        _symbol_for_run_dir = str(_symbol_from_path)
                except Exception:
                    pass
        # Если filename не дал run_id, возьмем из code
        if (not _run_id) and code:
            _run_id = str(code)
        # Если символ/папка все еще не ясны, попробуем найти по run_id среди result/*/runs/<run_id>
        if _run_id:
            try:
                runs_root = Path('result')
                if runs_root.exists():
                    for sym_dir in runs_root.iterdir():
                        rdir = sym_dir / 'runs' / _run_id
                        if rdir.exists():
                            _symbol_for_run_dir = sym_dir.name
                            break
            except Exception:
                pass
        
        trades_file_path = None
        if _run_id:
            run_dir = Path('result') / _symbol_for_run_dir / 'runs' / _run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            trade_ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            trades_file_name = f'oos_trades_{trade_ts}.json'
            trades_file_path = run_dir / trades_file_name
            try:
                # Сохраняем trades_details в отдельный файл
                with open(trades_file_path, 'w', encoding='utf-8') as f:
                    json.dump({'trades': trades_details}, f, ensure_ascii=False, indent=2)
                result_payload['result']['trades_file'] = str(trades_file_path.name)
                current_app.logger.info(f"[OOS] trades details saved to {trades_file_path}")
            except Exception as e:
                current_app.logger.error(f"[OOS] failed to save trades details to {trades_file_path}: {e}")
        
        # Удаляем trades_details из основного payload
        result_payload['result'].pop('trades_details', None)

        # Автосохранение в oos_results.json (суммируем счётчики и пишем историю)
        try:
            run_dir = None
            if _run_id:
                run_dir = Path('result') / _symbol_for_run_dir / 'runs' / _run_id

            if run_dir:
                run_dir.mkdir(parents=True, exist_ok=True) # Убедимся, что директория существует
                oos_file = run_dir / 'oos_results.json'
                doc = {'version': 1, 'counters': {'good': 0, 'bad': 0, 'neutral': 0}, 'history': []}
                if oos_file.exists():
                    try:
                        doc = json.loads(oos_file.read_text(encoding='utf-8'))
                        if not isinstance(doc, dict):
                            doc = {'version': 1, 'counters': {'good': 0, 'bad': 0, 'neutral': 0}, 'history': []}
                    except Exception as e: # Добавляем логирование ошибки
                        current_app.logger.error(f"[OOS] Failed to parse existing oos_results.json: {e}")
                        doc = {'version': 1, 'counters': {'good': 0, 'bad': 0, 'neutral': 0}, 'history': []}
                # Обновляем счётчики
                counters = doc.get('counters') or {}
                for k in ('good','bad','neutral'):
                    if k not in counters or not isinstance(counters.get(k), int):
                        counters[k] = 0
                counters[classification] = int(counters.get(classification, 0)) + 1
                doc['counters'] = counters
                # Добавляем запись истории
                try:
                    ts_utc = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                except Exception:
                    ts_utc = None
                history = doc.get('history') or []
                history.append({
                    'ts': ts_utc,
                    'days': int(days),
                    'gate_disable': bool(gate_disable),
                    'gate_scale': (float(gate_scale) if isinstance(gate_scale, (int, float)) else None),
                    't1_base': float(T1_user),
                    't2_base': float(T2_user),
                    'metrics': {
                        'winrate': winrate,
                        'pl_ratio': pl_ratio,
                        'profit_factor': profit_factor,
                        'max_dd': max_dd,
                        'pnl_total': pnl_total,
                        'trades': trades,
                        'start_capital': float(start_capital),
                        'roi_pct': roi_pct,
                        'nan_indicators': nan_count,
                        'inf_indicators': inf_count,
                        'trades_file': (str(trades_file_path.name) if trades_file_path else None),
                    },
                    'classification': classification,
                    'symbol': symbol,
                })
                # Ограничим историю, чтобы файл не рос бесконечно
                if len(history) > 500:
                    history = history[-500:]
                doc['history'] = history
                oos_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    last = history[-1] if history else None
                    current_app.logger.info(
                        f"[OOS] saving oos_results.json run_id={_run_id} symbol_dir={_symbol_for_run_dir} path={oos_file}"
                    )
                    try:
                        current_app.logger.info(
                            f"[OOS] counters={doc.get('counters')} history_len={len(history)} "
                            f"last_class={last.get('classification') if isinstance(last, dict) else None} "
                            f"last_trades={(last.get('metrics') or {}).get('trades') if isinstance(last, dict) else None} "
                            f"last_roi={(last.get('metrics') or {}).get('roi_pct') if isinstance(last, dict) else None}"
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
                oos_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
                try:
                    fsize = os.path.getsize(oos_file)
                    current_app.logger.info(
                        f"[OOS] oos_results.json saved ok path={oos_file} size={fsize}B history_len={len(history)}"
                    )
                except Exception:
                    current_app.logger.info(f"[OOS] oos_results.json saved to {oos_file} (size unknown)")
            else:
                current_app.logger.error(
                    f"[OOS] cannot resolve run_dir to save oos_results.json: run_id={_run_id} symbol_dir={_symbol_for_run_dir} filename={filename} code={code}"
                )
        except Exception as e: # Добавляем логирование ошибки для всего блока сохранения
            current_app.logger.error(f"[OOS] Failed to save oos_results.json: {e}")
        try:
            current_app.logger.info(f"[OOS] decisions summary: {decisions_counts}")
            current_app.logger.info(f"[OOS] done symbol={symbol} days={days} trades={trades} winrate={winrate} pf={profit_factor} maxDD={max_dd} pnl={pnl_total}")
        except Exception:
            pass
        # Добавляем сводку решений в ответ
        try:
            result_payload['result']['decisions_counts'] = {
                'buy': int(decisions_counts.get('buy', 0)),
                'hold': int(decisions_counts.get('hold', 0)),
                'sell': int(decisions_counts.get('sell', 0)),
            }
            result_payload['result']['decisions_total'] = int(sum(int(v) for v in decisions_counts.values()))
        except Exception:
            pass

        # Добавляем данные OHLCV и run_id в result_payload
        try:
            # Ограничиваем количество свечей для графика (например, последние 500)
            ohlcv_data_for_chart = df[['dt', 'open', 'high', 'low', 'close']].tail(500).copy()
            ohlcv_data_for_chart['dt'] = ohlcv_data_for_chart['dt'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            result_payload['ohlcv_data'] = ohlcv_data_for_chart.to_dict(orient='records')
            result_payload['run_id'] = _run_id # Добавляем run_id
        except Exception as e:
            current_app.logger.error(f"[OOS] Failed to add OHLCV data to payload: {e}")
            result_payload['ohlcv_data'] = []

        return jsonify(result_payload)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/oos_test_model_async')
def oos_test_model_async():
    try:
        data = request.get_json(silent=True) or {}
        payload = {
            'filename': (data.get('filename') or data.get('model') or '').strip(),
            'code': (data.get('code') or '').strip(),
            'days': int(data.get('days') or 30),
            'start_capital': data.get('start_capital'),
            'fee_enabled': data.get('fee_enabled'),
            'fee_rate': data.get('fee_rate'),
            'gate_disable': data.get('gate_disable'),
            'gate_scale': data.get('gate_scale'),
            'exchange': (data.get('exchange') or 'bybit'),
        }
        # Отправляем задачу по имени без прямого импорта, чтобы избежать циклов
        task = celery.send_task('tasks.oos_tasks.run_oos_test', kwargs={'payload': payload}, queue='oos')
        try:
            rc_id = f"ui:oos:tasks"
            from utils.redis_utils import get_redis_client
            r = get_redis_client()
            r.lrem(rc_id, 0, task.id)
            r.lpush(rc_id, task.id)
            r.ltrim(rc_id, 0, 99)
        except Exception:
            pass
        return jsonify({'success': True, 'task_id': task.id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.get('/oos_test_status')
def oos_test_status():
    try:
        task_id = (request.args.get('task_id') or '').strip()
        if not task_id:
            return jsonify({'success': False, 'error': 'task_id required'}), 400
        ar = AsyncResult(task_id, app=celery)
        resp = {
            'success': True,
            'task_id': task_id,
            'state': ar.state,
        }
        if ar.state in ('SUCCESS', 'FAILURE'):
            try:
                res = ar.result
                # результат нашей задачи — json словарь
                if isinstance(res, dict):
                    resp['result'] = res
            except Exception:
                pass
        return jsonify(resp)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.get('/api/runs/trades')
def api_runs_trades():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()
        trades_filename = (request.args.get('trades_file') or '').strip()

        if not symbol or not run_id or not trades_filename:
            return jsonify({'success': False, 'error': 'symbol, run_id and trades_file required'}), 400
        
        trades_file_path = Path('result') / symbol / 'runs' / run_id / trades_filename.split('/')[-1] # Принимаем относительный путь, но берем только имя файла
        
        if not trades_file_path.exists():
            return jsonify({'success': False, 'error': 'trades file not found'}), 404
        
        try:
            with open(trades_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'trades': data.get('trades', [])})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to parse trades file: {e}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


