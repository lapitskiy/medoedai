from flask import Blueprint, jsonify, request
from pathlib import Path
import os

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
                    import json as _json
                    manifest = _json.loads(mf.read_text(encoding='utf-8'))
                except Exception:
                    manifest = {}
            model_path = str(rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = str(rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = str(rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None
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
            })
        # Простая сортировка по created_at, затем по run_id
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        return jsonify({'success': True, 'runs': runs})
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
        for fname in ['model.pth', 'replay.pkl', 'train_result.pkl', 'manifest.json']:
            p = run_dir / fname
            try:
                if p.exists():
                    os.remove(p)
                    deleted.append(str(p))
            except Exception as e:
                errors.append(f'{fname}: {e}')
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
        import numpy as np
        import pandas as pd
        import json
        import requests

        # Символ попытаемся угадать из кода (prefix до "_")
        symbol = 'BTCUSDT'
        try:
            if code:
                base = code.split('_')[0].upper()
                if base and len(base) <= 6:
                    symbol = base + 'USDT'
        except Exception:
            pass

        df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id='bybit')
        if df is None or df.empty:
            return jsonify({'success': False, 'error': f'no candles for {symbol}'}), 400
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

        # Подготовка состояния (как в trade): последние 100 OHLCV нормализованных
        ohlcv_cols = ['open','high','low','close','volume']
        pnl_total = 0.0
        wins = 0
        losses = 0
        trades = 0
        peak = 0.0
        equity = 0.0
        max_dd = 0.0
        last_action = 'hold'
        position = None
        entry_price = None

        serving_url = os.environ.get('SERVING_URL', 'http://serving:8000/predict_ensemble')
        model_paths = []
        if filename:
            model_paths = [filename]
        else:
            # Попытаемся найти файл по коду в result/
            p = Path('result')/f'dqn_model_{code}.pth'
            if p.exists():
                model_paths = [str(p)]
        if not model_paths:
            return jsonify({'success': False, 'error': 'model file not found'}), 400

        def _state_from_tail(tail_df):
            arr = tail_df[ohlcv_cols].values.astype('float32')
            max_vals = np.maximum(arr.max(axis=0), 1e-9)
            norm = (arr / max_vals).flatten()
            if norm.size < 500:
                norm = np.pad(norm, (0, 500 - norm.size))
            elif norm.size > 500:
                norm = norm[:500]
            return norm.tolist()

        closes = df['close'].astype(float).values
        for i in range(120, len(df)):
            tail = df.iloc[i-100:i]
            if tail.shape[0] < 20:
                continue
            state = _state_from_tail(tail)
            payload = {"state": state, "model_paths": model_paths, "symbol": symbol, "consensus": {}, "market_regime": "flat", "market_regime_details": {}}
            try:
                resp = requests.post(serving_url, json=payload, timeout=15)
                pj = resp.json() if resp.ok else {"success": False}
            except Exception:
                pj = {"success": False}
            if not pj.get('success'):
                continue
            # Берём финальное решение или действие первой модели
            decision = pj.get('decision') or (pj.get('predictions') or [{}])[0].get('action') or 'hold'
            price = float(closes[i])
            # Простая симуляция: buy -> открыть лонг; sell -> закрыть, если был лонг (profit/loss)
            if decision == 'buy' and position is None:
                position = 'long'; entry_price = price
            elif decision == 'sell' and position == 'long':
                pl = price - float(entry_price)
                pnl_total += pl
                trades += 1
                wins += 1 if pl > 0 else 0
                losses += 1 if pl <= 0 else 0
                position = None; entry_price = None
            # Эквити и maxDD
            equity = pnl_total
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

        winrate = (wins / trades) if trades > 0 else None
        pl_ratio = None
        try:
            avg_win = pnl_total / max(1, wins)
            avg_loss = abs(pnl_total / max(1, losses)) if losses > 0 else None
            if avg_win is not None and avg_loss:
                pl_ratio = (avg_win / avg_loss) if avg_loss > 1e-12 else None
        except Exception:
            pl_ratio = None
        profit_factor = None
        try:
            gross_profit = max(0.0, pnl_total)
            gross_loss = abs(min(0.0, pnl_total))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else None
        except Exception:
            profit_factor = None

        return jsonify({
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
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


