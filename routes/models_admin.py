from flask import Blueprint, jsonify, request, current_app
from pathlib import Path
import os
import json

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

        stg_file = run_dir / 'strategies.json'
        payload = {
            'label': label,
            'created_at': __import__('datetime').datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'params': params,
            'metrics': metrics,
            'symbol': str(symbol),
            'run_id': str(run_id),
        }
        stg_id = f"stg_{__import__('datetime').datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        # Читаем/пишем файл
        doc = {'version': 1, 'current': None, 'items': {}}
        if stg_file.exists():
            try:
                doc = json.loads(stg_file.read_text(encoding='utf-8'))
                if not isinstance(doc, dict):
                    doc = {'version': 1, 'current': None, 'items': {}}
            except Exception:
                doc = {'version': 1, 'current': None, 'items': {}}
        if 'items' not in doc or not isinstance(doc['items'], dict):
            doc['items'] = {}
        doc['items'][stg_id] = payload
        if set_current:
            doc['current'] = stg_id
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
        stg_file = run_dir / 'strategies.json'
        if not stg_file.exists():
            return jsonify({'success': False, 'error': 'strategies.json not found'}), 404
        doc = json.loads(stg_file.read_text(encoding='utf-8'))
        item = (doc.get('items') or {}).get(stg_id)
        if not item:
            return jsonify({'success': False, 'error': 'strategy not found'}), 404
        # Обновим current
        doc['current'] = stg_id
        stg_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        # Применим qgate в Redis (если доступен)
        try:
            from redis import Redis as _Redis
            rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
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
        # Стартовый капитал для расчёта эквити/просадки
        try:
            start_capital = float(data.get('start_capital') or os.environ.get('OOS_START_CAPITAL', 10000))
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
        import numpy as np
        import pandas as pd
        import json
        import requests

        # Определяем символ максимально надёжно
        symbol = 'BTCUSDT'
        symbol_resolved = False
        try:
            # 1) Если передан путь runs/.../model.pth — читаем symbol из manifest.json рядом
            if filename:
                from pathlib import Path as _Path
                p = _Path(filename.replace('\\', '/'))
                try:
                    p = p.resolve()
                except Exception:
                    pass
                if p.exists() and p.is_file() and p.name == 'model.pth' and 'runs' in p.parts:
                    manifest_path = p.parent / 'manifest.json'
                    # 1a) Пытаемся взять из manifest.json
                    try:
                        if manifest_path.exists():
                            import json as _json
                            mf = _json.loads(manifest_path.read_text(encoding='utf-8'))
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
                    from pathlib import Path as _Path
                    runs_root = _Path('result')
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

        df = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=40000, exchange_id='bybit')
        if df is None or df.empty:
            try:
                current_app.logger.error(f"[OOS] no candles for {symbol}")
            except Exception:
                pass
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
        pnl_total = 0.0  # суммарный PnL NET (с учётом комиссий)
        wins = 0
        losses = 0
        trades = 0
        peak = float(start_capital)
        equity = float(start_capital)
        max_dd = 0.0
        last_action = 'hold'
        position = None
        entry_price = None
        entry_ts = None
        trades_details = []

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
            try:
                current_app.logger.error(f"[OOS] model file not found for code='{code}' filename='{filename}'")
            except Exception:
                pass
            return jsonify({'success': False, 'error': 'model file not found'}), 400

        # Преобразуем пути моделей в абсолютные пути внутри контейнера (например, /workspace/result/...)
        try:
            from pathlib import Path as _Path
            abs_list = []
            for mp in model_paths:
                pp = _Path(str(mp).replace('\\','/'))
                if not pp.is_absolute():
                    pp = (_Path.cwd() / pp).resolve()
                else:
                    pp = pp.resolve()
                abs_list.append(str(pp))
            model_paths = abs_list
            current_app.logger.info(f"[OOS] model_paths resolved → {model_paths}")
        except Exception as _res_err:
            try:
                current_app.logger.warning(f"[OOS] model_paths resolve error: {_res_err}")
            except Exception:
                pass

        # Загрузим консенсус и настройки Q‑gate как в бою
        consensus_cfg = None
        try:
            from redis import Redis as _Redis
            rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
            raw = rc.get('trading:consensus')
            if raw:
                import json as _json
                consensus_cfg = _json.loads(raw)
        except Exception:
            consensus_cfg = None

        # Порог Q‑gate (как в execute_trade)
        try:
            T1 = float(os.environ.get('QGATE_T1', '0.35'))
        except Exception:
            T1 = 0.35
        try:
            T2 = float(os.environ.get('QGATE_T2', '0.25'))
        except Exception:
            T2 = 0.25
        # Фактор для флэта
        try:
            flat_factor = float(os.environ.get('QGATE_FLAT', '1.0'))
        except Exception:
            flat_factor = 1.0
        # Применим пользовательский скейл/выключение
        if gate_disable:
            T1_user = 0.0; T2_user = 0.0
        else:
            if isinstance(gate_scale, (int, float)) and gate_scale >= 0:
                # Уменьшаем пороги на gate_scale * 100% (например, 0.5 → вдвое ниже)
                T1_user = max(0.0, T1 * (1.0 - float(gate_scale)))
                T2_user = max(0.0, T2 * (1.0 - float(gate_scale)))
            else:
                T1_user = T1; T2_user = T2

        # Простейший расчёт режима рынка (из последних закрытых свечей)
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
                # Majority
                counts = {'flat': labels.count('flat'), 'uptrend': labels.count('uptrend'), 'downtrend': labels.count('downtrend')}
                winner = max(counts, key=counts.get)
                return winner, {'windows':windows,'labels':labels,'weights':weights,'voting':'majority','tie_break':'flat'}
            except Exception:
                return 'flat', {'windows':[60,180,300],'labels':['flat','flat','flat'],'weights':[1,1,1],'voting':'majority','tie_break':'flat'}

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
                    try:
                        current_app.logger.error(f"[OOS] serving 500 status={resp.status_code} body={resp.text[:400]}")
                    except Exception:
                        pass
                pj = resp.json() if resp.ok else {"success": False}
            except Exception as _srv_err:
                try:
                    current_app.logger.error(f"[OOS] serving request error: {_srv_err}")
                except Exception:
                    pass
                pj = {"success": False}
            if not pj.get('success'):
                continue
            # Применим тот же консенсус и Q‑gate, что и в бою (упрощённо)
            decision = pj.get('decision') or 'hold'
            preds_list = pj.get('predictions') or []
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
            # Итоговое решение есть — симулируем сделку
            price = float(closes[i])
            ts_ms = int(df.iloc[i]['timestamp']) if 'timestamp' in df.columns else None
            try:
                ts_iso = pd.to_datetime(ts_ms, unit='ms').isoformat() if ts_ms else None
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
                'trades_details': trades_details,
                'gate_disable': bool(gate_disable),
                'gate_scale': (float(gate_scale) if isinstance(gate_scale, (int, float)) else None),
                't1_base': float(T1_user),
                't2_base': float(T2_user),
                'start_capital': float(start_capital),
                'equity_end': float(equity),
            }
        }
        try:
            current_app.logger.info(f"[OOS] done symbol={symbol} days={days} trades={trades} winrate={winrate} pf={profit_factor} maxDD={max_dd} pnl={pnl_total}")
        except Exception:
            pass
        return jsonify(result_payload)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


