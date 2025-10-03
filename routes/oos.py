from flask import Blueprint, jsonify, request, render_template # type: ignore
from utils.db_utils import db_get_or_fetch_ohlcv
import ccxt # type: ignore
from utils.cctx_utils import normalize_symbol
from pathlib import Path
import json
import os

import pandas as pd # type: ignore
import pickle # type: ignore

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
        sac_runs = []
        sac_base = Path('result') / 'sac'
        if sac_base.exists():
            for d in sac_base.iterdir():
                if d.is_dir():
                    sac_runs.append(d.name)
        sac_runs.sort()

        dqn_symbols = []
        dqn_base = Path('result')
        if dqn_base.exists():
            for d in dqn_base.iterdir():
                # Исключаем папку sac, ищем только папки символов для DQN
                if d.is_dir() and d.name.lower() != 'sac':
                    # Проверяем, есть ли подпапка 'runs' внутри
                    if (d / 'runs').exists():
                        dqn_symbols.append(d.name)
        dqn_symbols.sort()

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
                # 90 дней 5m ~ 90*24*12 = 25920, ccxt отдаёт батчами; для простоты возьмём лимит ~40000 через since
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
                    if len(all_rows) >= 40000:
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
            df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id=primary_exchange)
            if df is None or df.empty:
                df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id=fallback_exchange)
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


