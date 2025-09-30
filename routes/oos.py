from flask import Blueprint, jsonify, request, render_template
from utils.db_utils import db_get_or_fetch_ohlcv
import ccxt
from utils.cctx_utils import normalize_symbol
from pathlib import Path
import json

import pandas as pd

oos_bp = Blueprint('oos', __name__)


@oos_bp.get('/oos_graph')
def oos_graph_page():
    try:
        return render_template('oos/oos_graph.html')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@oos_bp.get('/api/runs/trades_files')
def api_runs_trades_files():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()
        
        print(f"[TRADES_FILES] Request: symbol='{symbol}', run_id='{run_id}'")
        
        if not symbol or not run_id:
            print(f"[TRADES_FILES] Missing parameters: symbol={symbol}, run_id={run_id}")
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400

        # Используем символ как есть (как в models_admin.py)
        run_dir = Path('result') / symbol / 'runs' / run_id
        print(f"[TRADES_FILES] Looking for run_dir: {run_dir}")
        print(f"[TRADES_FILES] run_dir.exists(): {run_dir.exists()}")
        
        if not run_dir.exists():
            # Попробуем альтернативный путь с BTC вместо BTCUSDT
            if symbol.endswith('USDT'):
                alt_symbol = symbol.replace('USDT', '')
                alt_run_dir = Path('result') / alt_symbol / 'runs' / run_id
                print(f"[TRADES_FILES] Trying alternative path: {alt_run_dir}")
                print(f"[TRADES_FILES] alt_run_dir.exists(): {alt_run_dir.exists()}")
                if alt_run_dir.exists():
                    run_dir = alt_run_dir
                    print(f"[TRADES_FILES] Using alternative path: {run_dir}")
                else:
                    print(f"[TRADES_FILES] Alternative path also not found")
                    return jsonify({'success': False, 'error': f'run directory not found: {run_dir}'}), 404
            else:
                print(f"[TRADES_FILES] Path not found and no alternative to try")
                return jsonify({'success': False, 'error': f'run directory not found: {run_dir}'}), 404

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
        symbol = (request.args.get('symbol') or '').strip().upper()
        days = int(request.args.get('days') or 90)
        exchange_req = str(request.args.get('exchange') or 'bybit').lower().strip()
        live = str(request.args.get('live') or '0').strip() in ('1','true','yes')
        if exchange_req not in ('bybit', 'binance'):
            exchange_req = 'bybit'
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400

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


