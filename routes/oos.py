from flask import Blueprint, jsonify, request, render_template, current_app # type: ignore
from celery.result import AsyncResult # type: ignore
from utils.db_utils import db_get_or_fetch_ohlcv
import ccxt # type: ignore
from utils.cctx_utils import normalize_symbol
from pathlib import Path
import json
import os

import pandas as pd # type: ignore
import pickle # type: ignore
import numpy as np # type: ignore
import requests # type: ignore
from redis import Redis # type: ignore
from datetime import datetime
import torch # type: ignore

from utils.config_loader import get_config_value
from agents.vdqn.cfg.vconfig import vDqnConfig # type: ignore
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized # type: ignore
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes # type: ignore
from utils.path import list_symbol_dirs, resolve_run_dir, resolve_symbol_dir
from tasks import celery # type: ignore
from agents.vdqn.tianshou_train import make_env_fn, TradingEnvWrapper
from agents.vdqn.dqnn import DQNN

oos_bp = Blueprint('oos', __name__)


@oos_bp.get('/oos_graph')
def oos_graph_page():
    try:
        agent_type = request.args.get('agent_type', 'dqn').strip().lower()
        model_id = request.args.get('id', '').strip()
        return render_template('oos/oos_graph.html', agent_type=agent_type, model_id=model_id)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/oos')
def oos_page():
    try:
        sac_runs = sorted({str(p.name) for p in list_symbol_dirs('sac')})
        dqn_symbols = sorted({str(p.name) for p in list_symbol_dirs('dqn')})

        return render_template('oos/oos.html', sac_runs=sac_runs, dqn_symbols=dqn_symbols)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/api/dqn_runs/symbols')
def api_dqn_runs_symbols():
    try:
        symbols = []
        dqn_base = Path('result')
        if dqn_base.exists():
            for d in dqn_base.iterdir():
                if d.is_dir() and d.name.lower() != 'sac':
                    if (d / 'runs').exists():
                        symbols.append(d.name)
        symbols.sort()
        return jsonify({'success': True, 'symbols': symbols})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/api/dqn_runs/list')
def api_dqn_runs_list():
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

            manifest = {}
            metrics_file = rd / 'train_result.pkl' # DQN metrics are often in pkl
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'rb') as f:
                        metrics_data = pickle.load(f)
                        # Extract relevant stats
                        final_stats = metrics_data.get('final_stats', {}) or {}
                        train_metadata = metrics_data.get('train_metadata', {}) or {}
                        manifest.update(final_stats)
                        manifest.update(train_metadata)
                        manifest['episodes'] = metrics_data.get('actual_episodes', metrics_data.get('episodes'))
                        manifest['created_at'] = train_metadata.get('training_start_time')
                except Exception:
                    manifest = {}

            model_path = str(rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = str(rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = str(rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None

            winrate = manifest.get('winrate')
            pl_ratio = manifest.get('pl_ratio')
            max_dd = manifest.get('max_drawdown') # May not be present in DQN pkl
            episodes = manifest.get('episodes')

            runs.append({
                'run_id': rd.name,
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
                'max_dd': max_dd,
                'episodes': episodes,
                'agent_type': manifest.get('agent_type', 'dqn'), # Default to dqn
                'version': manifest.get('version', 'N/A'),
            })
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        return jsonify({'success': True, 'runs': runs})
    except Exception as e:
        import traceback
        print(f"[API_DQN_RUNS_LIST] Exception: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/api/runs/trades_files')
def api_runs_trades_files():
    try:
        agent_type = request.args.get('agent_type', 'dqn').strip().lower()
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()

        print(f"[TRADES_FILES] Request: agent_type='{agent_type}', symbol='{symbol}', run_id='{run_id}'")

        if not run_id:
            print(f"[TRADES_FILES] Missing parameter: run_id={run_id}")
            return jsonify({'success': False, 'error': 'run_id required'}), 400

        run_dir = None
        if agent_type == 'dqn':
            if not symbol:
                print(f"[TRADES_FILES] Missing parameter for DQN: symbol={symbol}")
                return jsonify({'success': False, 'error': 'symbol required for dqn agent_type'}), 400
            run_dir = Path('result') / symbol / 'runs' / run_id
        elif agent_type == 'sac':
            run_dir = Path('result') / 'sac' / run_id
        else:
            return jsonify({'success': False, 'error': 'Unsupported agent_type'}), 400

        print(f"[TRADES_FILES] Looking for run_dir: {run_dir}")
        print(f"[TRADES_FILES] run_dir.exists(): {run_dir.exists()}")

        if not run_dir.exists():
            # Для DQN пробуем альтернативный путь, если символ оканчивается на USDT
            if agent_type == 'dqn' and symbol.endswith('USDT'):
                alt_symbol = symbol.replace('USDT', '')
                alt_run_dir = Path('result') / alt_symbol / 'runs' / run_id
                print(f"[TRADES_FILES] Trying alternative path for DQN: {alt_run_dir}")
                print(f"[TRADES_FILES] alt_run_dir.exists(): {alt_run_dir.exists()}")
                if alt_run_dir.exists():
                    run_dir = alt_run_dir
                    print(f"[TRADES_FILES] Using alternative path for DQN: {run_dir}")
                else:
                    print(f"[TRADES_FILES] Alternative path for DQN also not found")
                    return jsonify({'success': False, 'error': f'run directory not found for DQN: {run_dir}'}), 404
            else:
                print(f"[TRADES_FILES] Path not found and no alternative to try for {agent_type}")
                return jsonify({'success': False, 'error': f'run directory not found for {agent_type}: {run_dir}'}), 404

        # Читаем oos_results.json для получения информации о периодах
        results_file = run_dir / 'oos_results.json'
        trades_info = []

        print(f"[DEBUG] results_file: {results_file}")
        print(f"[DEBUG] results_file.exists(): {results_file.exists()}")

        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                    history = results_data.get('history', [])
                    print(f"[DEBUG] Found {len(history)} entries in history")

                    for entry in history:
                        trades_file = entry.get('metrics', {}).get('trades_file')
                        days = entry.get('days', 'N/A')
                        ts = entry.get('ts', '')
                        print(f"[DEBUG] Processing entry: trades_file={trades_file}, days={days}")

                        if trades_file and (run_dir / trades_file).exists():
                            trades_info.append({
                                'file': trades_file,
                                'days': days,
                                'timestamp': ts,
                                'classification': entry.get('classification', 'unknown')
                            })
                            print(f"[DEBUG] Added trades file: {trades_file}")
                        else:
                            print(f"[DEBUG] Skipped trades file: {trades_file} (exists: {(run_dir / trades_file).exists() if trades_file else False})")
            except Exception as e:
                print(f"Error reading oos_results.json: {e}")

        # Если oos_results.json не найден, ищем все oos_trades_*.json файлы
        if not trades_info:
            print(f"[DEBUG] No trades_info from oos_results.json, searching for oos_trades_*.json files")
            for file_path in run_dir.glob('oos_trades_*.json'):
                trades_info.append({
                    'file': file_path.name,
                    'days': 'N/A',
                    'timestamp': '',
                    'classification': 'unknown'
                })
                print(f"[DEBUG] Found trades file: {file_path.name}")

        print(f"[TRADES_FILES] Final trades_info: {trades_info}")
        print(f"[TRADES_FILES] Returning {len(trades_info)} files")

        return jsonify({'success': True, 'files': trades_info})
    except Exception as e:
        print(f"[TRADES_FILES] Exception: {e}")
        import traceback
        print(f"[TRADES_FILES] Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/api/runs/ohlcv')
def api_runs_ohlcv():
    try:
        agent_type = request.args.get('agent_type', 'dqn').strip().lower()
        model_id = (request.args.get('id') or '').strip().upper() # id может быть символом для DQN или run_id для SAC
        days = int(request.args.get('days') or 90)
        exchange_req = str(request.args.get('exchange') or 'bybit').lower().strip()
        live = str(request.args.get('live') or '0').strip() in ('1','true','yes')

        if exchange_req not in ('bybit', 'binance'):
            exchange_req = 'bybit'

        if not model_id:
            return jsonify({'success': False, 'error': 'model_id required'}), 400

        symbol = model_id # По умолчанию считаем model_id символом, для SAC будем использовать run_id
        if agent_type == 'sac':
            # Для SAC нам нужен символ, который должен быть сохранен где-то в метаданных run_id
            # Пока что, если модель SAC, мы не можем извлечь символ напрямую из model_id (run_id)
            # Это может потребовать более сложной логики чтения метаданных SAC-модели
            # Временное решение: пока что для SAC не будем использовать symbol для ohlcv, если он не указан явно
            # В будущем нужно будет добавить чтение символа из metrics.json или manifest.yaml SAC модели
            # Для этого требуется более глубокая интеграция с тем, как SAC модели сохраняют свои параметры
            pass # Пока оставляем symbol как model_id, если он был передан

        candles_per_day_5m = (24 * 60) // 5
        buffer_days = 5
        limit_candles = max(40000, (days + buffer_days) * candles_per_day_5m)

        if live:
            try:
                uni = normalize_symbol(symbol)
                ex_class = getattr(ccxt, exchange_req)
                ex = ex_class({'enableRateLimit': True, 'timeout': 30000})
                ex.load_markets()
                if uni not in getattr(ex, 'symbols', []):
                    # попытка по id
                    markets = getattr(ex, 'markets', {}) or {}
                    match = None
                    for k, m in markets.items():
                        if str(m.get('id','')).upper() == symbol.upper():
                            match = k; break
                    if match:
                        uni = match
                # 5m свечей ~ 288 в день, ccxt отдаёт батчами; для простоты берём лимит с запасом
                since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=int(max(1, days)))).timestamp()*1000)
                all_rows = []
                cursor = since_ms
                tf = '5m'
                tf_ms = ex.parse_timeframe(tf)*1000
                while True:
                    batch = ex.fetch_ohlcv(uni, tf, since=cursor, limit=1000)
                    if not batch:
                        break
                    all_rows.extend(batch)
                    last_ts = batch[-1][0]
                    next_cursor = last_ts + tf_ms
                    if next_cursor <= cursor:
                        break
                    cursor = next_cursor
                    if len(all_rows) >= limit_candles:
                        break
                    # небольшой троттлинг
                    ex.sleep(1000)
                if not all_rows:
                    return jsonify({'success': False, 'error': 'no candles (live)'}), 400
                ohlcv = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
            except Exception as le:
                return jsonify({'success': False, 'error': f'live fetch failed: {le}'}), 500
        else:
            primary_exchange = exchange_req
            fallback_exchange = 'binance' if primary_exchange == 'bybit' else 'bybit'
            df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=primary_exchange)
            if df is None or df.empty:
                df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=fallback_exchange)
            if df is None or df.empty:
                return jsonify({'success': False, 'error': f'no candles for {symbol}'}), 400

        try:
            if live:
                df = ohlcv
            df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
            cutoff = df['dt'].max().floor('5min') - pd.Timedelta(minutes=5)
            start_ts = cutoff - pd.Timedelta(days=int(max(1, days)))
            df = df[(df['dt'] > start_ts) & (df['dt'] <= cutoff)]
        except Exception:
            pass

        # --- DEBUG LOG: последние 2 дня, сводка и хвост 100 свечей ---
        try:
            debug_from = cutoff - pd.Timedelta(days=2)
            d2 = df[df['dt'] >= debug_from].copy()
            total = int(len(d2))
            if total > 0:
                nan_cnt = int(d2[['open','high','low','close']].isna().any(axis=1).sum())
                zero_spread_cnt = int(((d2['high'] - d2['low']).abs() < 1e-9).sum())
                high_lt_ocmax = int((d2['high'] < d2[['open','close']].max(axis=1)).sum())
                low_gt_ocmin = int((d2['low']  > d2[['open','close']].min(axis=1)).sum())
                print(f"[OOS/OHLCV-DEBUG] symbol={symbol} days={days} window=({start_ts}..{cutoff}) last2d_rows={total} nan={nan_cnt} zero_spread={zero_spread_cnt} high<max(oc)={high_lt_ocmax} low>min(oc)={low_gt_ocmin}")

                # Подробный список нулевых спредов (последние 60 записей)
                try:
                    zs_mask = (d2['high'] - d2['low']).abs() < 1e-9
                    flat_mask = zs_mask & (d2['open'].sub(d2['close']).abs() < 1e-9)
                    zs_df = d2.loc[zs_mask, ['dt','open','high','low','close']].tail(60)
                    flat_df = d2.loc[flat_mask, ['dt','open','high','low','close']].tail(60)
                    zs_list = [
                        {
                            'dt': str(row['dt']),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close'])
                        } for _, row in zs_df.iterrows()
                    ]
                    flat_list = [
                        {
                            'dt': str(row['dt']),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close'])
                        } for _, row in flat_df.iterrows()
                    ]
                    print(f"[OOS/OHLCV-DEBUG] zero_spread_tail60: {json.dumps(zs_list, ensure_ascii=False)}")
                    print(f"[OOS/OHLCV-DEBUG] flat_ohlc_tail60: {json.dumps(flat_list, ensure_ascii=False)}")
                except Exception as _ze:
                    print(f"[OOS/OHLCV-DEBUG] zero-spread listing error: {_ze}")
                tail = d2.tail(100)[['dt','open','high','low','close']]
                sample = [
                    {
                        'dt': str(row['dt']),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close'])
                    } for _, row in tail.iterrows()
                ]
                print(f"[OOS/OHLCV-DEBUG] tail100: {json.dumps(sample, ensure_ascii=False)}")
            else:
                print(f"[OOS/OHLCV-DEBUG] symbol={symbol} last2d_rows=0 (нет данных в последние 2 дня)")
        except Exception as e:
            print(f"[OOS/OHLCV-DEBUG] error while logging: {e}")

        ohlcv = df[['dt','open','high','low','close']].copy()
        ohlcv['dt'] = ohlcv['dt'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        data = [
            {
                'dt': str(r['dt']),
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close'])
            } for _, r in ohlcv.iterrows()
        ]
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@oos_bp.post('/oos_test_model')
def oos_test_model():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        code = (data.get('code') or '').strip()
        days = int(data.get('days') or 30)
        agent_type_raw = (data.get('agent_type') or '').strip().lower()
        agent_type = agent_type_raw if agent_type_raw in ('dqn', 'sac') else None
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
        candles_per_day_5m = (24 * 60) // 5
        buffer_days = 5
        limit_candles = max(40000, (days + buffer_days) * candles_per_day_5m)
        try:
            current_app.logger.info(f"[OOS] using exchange={primary_exchange} (fallback={fallback_exchange}) for {symbol}")
        except Exception:
            pass
        df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=primary_exchange)
        if df is None or df.empty:
            try:
                current_app.logger.warning(f"[OOS] no candles from {primary_exchange} for {symbol}, trying {fallback_exchange}")
            except Exception:
                pass
            df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=fallback_exchange)
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
            if df.empty:
                return jsonify({'success': False, 'error': 'no oos window'}), 400
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
            symbol_hint = symbol.replace('USDT', '')
            inferred_type = agent_type
            if not inferred_type:
                inferred_type = 'sac' if 'sac' in (_run_id or '').lower() else 'dqn'
            symbol_dir = resolve_symbol_dir(inferred_type, symbol_hint, create=True)
            run_dir = symbol_dir / 'runs' / _run_id if symbol_dir else None
            if run_dir:
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
                symbol_hint = symbol.replace('USDT', '')
                inferred_type = agent_type
                if not inferred_type:
                    inferred_type = 'sac' if 'sac' in (_run_id or '').lower() else 'dqn'
                symbol_dir = resolve_symbol_dir(inferred_type, symbol_hint, create=True)
                run_dir = symbol_dir / 'runs' / _run_id if symbol_dir else None

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


@oos_bp.post('/oos_ts_test_model')
def oos_ts_test_model():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        code = (data.get('code') or '').strip()
        days = int(data.get('days') or 30)
        episode_length = int(data.get('episode_length') or 500)

        # Определяем символ/путь аналогично обычному OOS
        symbol = 'BTCUSDT'
        run_dir = None
        if filename:
            p = Path(filename.replace('\\','/')).resolve()
            if p.exists() and p.is_file() and p.name == 'model.pth' and 'runs' in p.parts:
                runs_idx = list(p.parts).index('runs')
                if runs_idx-1 >= 0:
                    base = str(p.parts[runs_idx-1]).upper()
                    symbol = base if base.endswith('USDT') else (base + 'USDT')
                run_dir = p.parent
        if (run_dir is None) and code:
            # Поиск run по коду
            root = Path('result')
            if root.exists():
                for sym_dir in root.iterdir():
                    cand = sym_dir / 'runs' / code
                    if (cand / 'model.pth').exists():
                        run_dir = cand
                        symbol = (sym_dir.name.upper() if sym_dir.name.upper().endswith('USDT') else (sym_dir.name.upper()+'USDT'))
                        break
        if run_dir is None:
            return jsonify({'success': False, 'error': 'run directory not found (provide filename or code)'}), 400

        # Загружаем данные (тот же подход, что и в обычном OOS)
        primary_exchange = 'bybit'
        fallback_exchange = 'binance'
        candles_per_day_5m = (24 * 60) // 5
        limit_candles = max(40000, (days + 5) * candles_per_day_5m)
        df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=primary_exchange)
        if df is None or df.empty:
            df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id=fallback_exchange)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'no candles for {symbol}'}), 400
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        cutoff = df['dt'].max().floor('5min') - pd.Timedelta(minutes=5)
        start_ts = cutoff - pd.Timedelta(days=int(max(1, days)))
        df = df[(df['dt'] > start_ts) & (df['dt'] <= cutoff)]
        if df.empty:
            return jsonify({'success': False, 'error': 'no oos window'}), 400

        # Формируем dfs для env (как в обучении)
        df_5min = df.copy().reset_index(drop=True)
        df_15min = df_5min.resample('15min', on='dt').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        df_1h = df_5min.resample('1h', on='dt').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        dfs = {'df_5min': df_5min, 'df_15min': df_15min, 'df_1h': df_1h, 'symbol': symbol}

        # Создаём env через make_env_fn → TradingEnvWrapper (как в обучении)
        env = TradingEnvWrapper(make_env_fn(dfs, episode_length)())

        # Определяем размеры для сети
        obs_dim = getattr(env, 'observation_space_shape', None)
        if obs_dim is None and hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
            obs_dim = int(env.observation_space.shape[0])
        if obs_dim is None:
            return jsonify({'success': False, 'error': 'cannot resolve obs_dim'}), 500
        act_dim = env.action_space.n

        # Готовим сеть и загружаем веса
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = DQNN(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=(512,512,256),
            dropout_rate=0.1,
            layer_norm=True,
            dueling=True,
            activation='relu',
            use_residual=True,
            use_swiglu=False,
        ).to(device)
        ckpt_path = run_dir / 'model.pth'
        if not ckpt_path.exists():
            return jsonify({'success': False, 'error': 'model.pth not found in run_dir'}), 400
        sd = torch.load(str(ckpt_path), map_location=device)
        state = sd.get('model', sd) if isinstance(sd, dict) else sd
        net.load_state_dict(state, strict=False)
        net.eval()

        # Простой цикл инференса (без policy), greedy argmax
        obs, info = env.reset()
        total_reward = 0.0
        actions_stats = {0:0,1:0,2:0}
        trades_total = 0
        for _ in range(int(episode_length)):
            with torch.no_grad():
                logits, _ = net.forward(obs)
                if isinstance(logits, torch.Tensor):
                    act = int(torch.argmax(logits, dim=-1).item())
                else:
                    act = int(np.argmax(np.asarray(logits)))
            actions_stats[act] = actions_stats.get(act, 0) + 1
            obs, reward, terminated, truncated, info = env.step(act)
            total_reward += float(reward)
            if terminated or truncated:
                try:
                    if isinstance(info, dict) and 'trades_episode' in info and isinstance(info['trades_episode'], list):
                        trades_total += len(info['trades_episode'])
                except Exception:
                    pass
                break

        result = {
            'success': True,
            'symbol': symbol,
            'run_dir': str(run_dir),
            'episode_length': int(episode_length),
            'reward': float(total_reward),
            'actions': actions_stats,
            'trades': int(trades_total),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.post('/oos_test_model_async')
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
            'agent_type': (data.get('agent_type') or '').strip().lower(),
            'symbol': (data.get('symbol') or '').strip().upper(),
            'run_id': (data.get('run_id') or '').strip(),
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


@oos_bp.get('/oos_test_status')
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
    


