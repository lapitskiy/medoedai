from flask import Blueprint, jsonify, request, current_app, render_template # type: ignore
from celery.result import AsyncResult # type: ignore
from tasks import celery # type: ignore
from pathlib import Path
import os
from utils.config_loader import get_config_value
import json
import time
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized # type: ignore
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes # type: ignore
from agents.vdqn.cfg.vconfig import vDqnConfig # type: ignore
import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
import requests # type: ignore
from redis import Redis # type: ignore
from datetime import datetime
import torch # type: ignore
import shutil

from utils.path import (
    resolve_run_dir,
    resolve_symbol_dir,
    list_symbol_dirs,
)

models_admin_bp = Blueprint('models_admin', __name__)

# Cache for /api/runs/list to avoid repeated filesystem reads and any accidental heavy loads
# Key: (model_type, resolved_symbol_dir_path)
_RUNS_LIST_CACHE: dict[tuple[str, str], dict] = {}

@models_admin_bp.route('/models')
def models_page():
    """Страница управления моделями (DQN)"""
    return render_template('models.html')


@models_admin_bp.get('/api/runs/symbols')
def api_runs_symbols():
    try:
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        if model_type not in {'dqn', 'sac'}:
            model_type = 'dqn'

        symbols = []
        for path in list_symbol_dirs(model_type):
            try:
                if not path.is_dir():
                    continue
                runs_dir = path / 'runs'
                if not runs_dir.exists():
                    continue
                if model_type == 'sac':
                    symbol_value = path.name
                else:
                    symbol_value = path.name.upper()
                if symbol_value not in symbols:
                    symbols.append(symbol_value)
            except Exception:
                continue

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
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        if model_type not in {'dqn', 'sac'}:
            model_type = 'dqn'

        symbol_raw = (request.args.get('symbol') or '').strip()
        if not symbol_raw:
            return jsonify({'success': False, 'error': 'symbol required'}), 400

        base = Path('result')
        symbol_candidates = []
        if model_type == 'sac':
            base = base / 'sac'
            core_symbol = symbol_raw.split(':',1)[-1]
            symbol_candidates = [core_symbol.lower(), core_symbol.upper(), core_symbol]
        else:
            core_symbol = symbol_raw
            symbol_candidates = [core_symbol.upper(), core_symbol.lower(), core_symbol]
        requested_value = core_symbol

        runs = []
        symbol_dir = resolve_symbol_dir(model_type, symbol_raw, create=False)
        if symbol_dir is None:
            return jsonify({'success': True, 'runs': []})

        runs_root = symbol_dir / 'runs'
        if not runs_root.exists():
            return jsonify({'success': True, 'runs': []})

        # Build a cheap signature from mtimes/sizes to decide cache hit.
        # This keeps the endpoint fast even when the UI refreshes frequently.
        cache_key = (model_type, symbol_dir.as_posix())
        sig_parts = []
        try:
            for rd in runs_root.iterdir():
                if not rd.is_dir():
                    continue
                mf = rd / 'manifest.json'
                metrics = rd / 'metrics.json'
                tr = rd / 'train_result.pkl'
                mp = rd / 'model.pth'
                # We only care about files that influence list output
                def _stat_key(p: Path):
                    try:
                        st = p.stat()
                        return (int(st.st_mtime), int(st.st_size))
                    except Exception:
                        return None
                sig_parts.append((
                    rd.name,
                    bool(mp.exists()),
                    _stat_key(tr) if tr.exists() else None,
                    _stat_key(mf) if mf.exists() else None,
                    _stat_key(metrics) if metrics.exists() else None,
                ))
        except Exception:
            sig_parts = []
        signature = tuple(sorted(sig_parts))
        cached = _RUNS_LIST_CACHE.get(cache_key)
        if cached and cached.get('signature') == signature:
            return jsonify({'success': True, 'runs': cached.get('runs', [])})

        for rd in runs_root.iterdir():
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
            model_path = (rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = (rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = (rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None
            # Попытаемся извлечь краткую статистику
            winrate = None; pl_ratio = None; roi = None; max_dd = None; episodes = None
            episodes_planned = None
            best_winrate = None
            episode_winrate_avg = None
            episode_winrate_last = None
            validation_avg = None
            validation_last = None
            seed_value = manifest.get('seed')
            # Направление сделки/режим обучения (long/short) — сохраняется в manifest.json как direction/trained_as
            direction_value = manifest.get('direction') or manifest.get('trained_as')
            # IMPORTANT: do NOT unpickle train_result.pkl here — it can be huge and makes /oos slow.
            # Use lightweight manifest.json for DQN, and metrics.json for SAC.
            try:
                if isinstance(manifest, dict):
                    bm = manifest.get('best_metrics') or {}
                    if isinstance(bm, dict):
                        if isinstance(bm.get('winrate'), (int, float)):
                            winrate = float(bm.get('winrate'))
                            # best_winrate historically ожидается как "лучший winrate"
                            best_winrate = float(bm.get('winrate'))
                    if isinstance(manifest.get('episodes_end'), int):
                        episodes = int(manifest.get('episodes_end'))
                    if isinstance(manifest.get('episodes_last'), int) and episodes is None:
                        episodes = int(manifest.get('episodes_last'))
            except Exception:
                pass

            if model_type == 'sac':
                metrics_file = rd / 'metrics.json'
                if metrics_file.exists():
                    try:
                        metrics_data = json.loads(metrics_file.read_text(encoding='utf-8'))
                        if isinstance(metrics_data, dict):
                            if isinstance(metrics_data.get('winrate'), (int, float)):
                                winrate = float(metrics_data.get('winrate'))
                            # SAC: metrics.json использует avg_roi и max_drawdown
                            if isinstance(metrics_data.get('avg_roi'), (int, float)):
                                roi = float(metrics_data.get('avg_roi'))
                            elif isinstance(metrics_data.get('roi'), (int, float)):
                                roi = float(metrics_data.get('roi'))
                            if isinstance(metrics_data.get('max_drawdown'), (int, float)):
                                max_dd = float(metrics_data.get('max_drawdown'))
                    except Exception:
                        pass
            else:
                # DQN: если метрики не попали в manifest.json, попробуем прочитать train_result.pkl (после миграции он обычно лёгкий).
                # Ограничим размер, чтобы не вернуть старые 100MB+ тормоза.
                allow_pkl_fallback = True
                max_mb = 32
                try:
                    v = str(get_config_value('UI_RUNS_LIST_PKL_FALLBACK_MAX_MB', str(max_mb)))
                    max_mb = int(float(v))
                except Exception:
                    max_mb = 32
                try:
                    # NOTE: winrate может быть записан в manifest.json как 0.0 (дефолт),
                    # но реальная метрика есть в train_result.pkl. Поэтому разрешаем override.
                    needs = (
                        (winrate is None) or (pl_ratio is None) or (max_dd is None) or (direction_value is None)
                        or (isinstance(winrate, (int, float)) and float(winrate) == 0.0)
                    )
                    if result_path and needs:
                        st = result_path.stat()
                        if st.st_size <= max_mb * 1024 * 1024 and allow_pkl_fallback:
                            try:
                                with open(result_path, 'rb') as _f:
                                    _res = pickle.load(_f)
                                if isinstance(_res, dict):
                                    _fs = _res.get('final_stats') or {}
                                    if isinstance(_fs, dict):
                                        # winrate: legacy key or tianshou key
                                        if winrate is None or (isinstance(winrate, (int, float)) and float(winrate) == 0.0):
                                            w = _fs.get('winrate')
                                            if isinstance(w, (int, float)):
                                                winrate = float(w)
                                            else:
                                                w2 = _fs.get('winrate_from_trades')
                                                if isinstance(w2, (int, float)):
                                                    winrate = float(w2)

                                        # pl_ratio: legacy key or compute from total_profit/total_loss (tianshou)
                                        if pl_ratio is None:
                                            pr = _fs.get('pl_ratio')
                                            if isinstance(pr, (int, float)):
                                                pl_ratio = float(pr)
                                            else:
                                                # alternate keys
                                                pf = _fs.get('profit_factor')
                                                if isinstance(pf, (int, float)):
                                                    pl_ratio = float(pf)
                                                else:
                                                    ap = _fs.get('avg_profit')
                                                    al = _fs.get('avg_loss')
                                                    if isinstance(ap, (int, float)) and isinstance(al, (int, float)) and float(al) < 0:
                                                        denom = abs(float(al))
                                                        if denom > 0:
                                                            pl_ratio = float(ap) / denom
                                                tp = _fs.get('total_profit')
                                                tl = _fs.get('total_loss')
                                                if isinstance(tp, (int, float)) and isinstance(tl, (int, float)) and float(tl) > 0:
                                                    pl_ratio = float(tp) / float(tl)

                                        if max_dd is None:
                                            dd = _fs.get('max_drawdown')
                                            if isinstance(dd, (int, float)):
                                                max_dd = float(dd)
                                    # best_winrate: синхронизируем с winrate, если он появился
                                    if best_winrate is None and isinstance(winrate, (int, float)):
                                        best_winrate = float(winrate)
                                    if direction_value is None:
                                        dv = _res.get('direction') or _res.get('trained_as')
                                        if isinstance(dv, str) and dv.strip():
                                            direction_value = dv.strip()
                            except Exception:
                                pass
                except Exception:
                    pass
            runs.append({
                'run_id': run_id,
                'parent_run_id': manifest.get('parent_run_id'),
                'root_id': manifest.get('root_id'),
                'seed': seed_value,
                'direction': (str(direction_value).strip().lower() if direction_value is not None else None),
                'episodes_end': manifest.get('episodes_end'),
                'created_at': manifest.get('created_at'),
                'model_path': model_path.as_posix() if model_path else None,
                'replay_path': replay_path.as_posix() if replay_path else None,
                'result_path': result_path.as_posix() if result_path else None,
                'winrate': winrate,
                'pl_ratio': pl_ratio,
                'roi': roi,
                'max_dd': max_dd,
                'episodes': episodes,
                'episodes_planned': episodes_planned,
                'best_winrate': best_winrate,
                'episode_winrate_avg': episode_winrate_avg,
                'episode_winrate_last': episode_winrate_last,
                'validation_avg': validation_avg,
                'validation_last': validation_last,
                'symbol_dir': symbol_dir.name if symbol_dir else requested_value,
                'train_result_exists': bool(result_path),
                'model_type': model_type,
                'agent_type': model_type,
            })
        # Простая сортировка по created_at, затем по run_id
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        _RUNS_LIST_CACHE[cache_key] = {'signature': signature, 'runs': runs, 'ts': time.time()}
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
        run_dir = resolve_run_dir('dqn', run_id, symbol_hint=symbol, create=False)
        if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
            return jsonify({'success': False, 'error': 'run directory not found'}), 404
        oos_file = run_dir / 'oos_results.json'
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
        model_type = (data.get('model_type') or data.get('agent_type') or 'dqn').strip().lower()
        symbol = (data.get('symbol') or '').strip().upper()
        run_id = (data.get('run_id') or '').strip()
        if not symbol or not run_id:
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400
        run_dir = resolve_run_dir(model_type, run_id, symbol_hint=symbol, create=False)
        if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
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


# === Encoders API ===
@models_admin_bp.get('/api/encoders')
def api_encoders_list():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400

        # Нормализуем базу символа: BTCUSDT -> btc, TONUSDT -> ton
        base_lower = symbol.split('USDT')[0].lower()
        base_upper = symbol.split('USDT')[0].upper()

        # Два корня поиска:
        # 1) Прод/релизы: models/<base>/encoder/unfrozen/vN
        # 2) Черновики/результаты: result/dqn/<BASE_UPPER>/encoder/unfrozen/vN
        roots = [
            Path('models') / base_lower / 'encoder' / 'unfrozen',
            Path('result') / 'dqn' / base_upper / 'encoder' / 'unfrozen',
            Path('result') / base_upper / 'encoder' / 'unfrozen',  # совместимость со старой структурой result/<SYMBOL>
        ]

        seen_ids = set()
        items = []

        def read_manifest(dir_path: Path):
            """Возвращает dict манифеста или None, если нет подходящих файлов."""
            cand_json = dir_path / 'encoder_manifest.json'
            cand_alt = dir_path / 'manifest.json'
            cand_yaml = dir_path / 'manifest.yaml'
            try:
                if cand_json.exists():
                    return json.loads(cand_json.read_text(encoding='utf-8'))
                if cand_alt.exists():
                    return json.loads(cand_alt.read_text(encoding='utf-8'))
                if cand_yaml.exists():
                    # Возвращаем raw YAML в удобной обёртке
                    txt = cand_yaml.read_text(encoding='utf-8')
                    return {'yaml': txt}
            except Exception:
                return None
            return None

        def _as_float(v):
            try:
                if isinstance(v, bool) or v is None:
                    return None
                return float(v)
            except Exception:
                return None

        def _as_int(v):
            try:
                if isinstance(v, bool) or v is None:
                    return None
                return int(v)
            except Exception:
                return None

        def extract_encoder_stats(manifest: dict, base_upper_symbol: str) -> dict:
            """
            Пытаемся вытащить метрики для UI:
            - winrate: обычно в encoder_manifest.json -> performance.avg_winrate (0..1)
            - episodes: training.cumulative_episodes_completed (или episodes_completed)
            - pnl: best-effort (если доступно из manifest/perf или train_result.pkl по training_run_dir/run_id)
            """
            stats = {'winrate': None, 'pnl': None, 'episodes': None}
            try:
                if not isinstance(manifest, dict):
                    return stats
                if 'yaml' in manifest:
                    return stats

                tr = manifest.get('training') or {}
                if isinstance(tr, dict):
                    stats['episodes'] = (
                        _as_int(tr.get('cumulative_episodes_completed'))
                        or _as_int(tr.get('episodes_completed'))
                        or _as_int(tr.get('episodes_end'))
                    )

                perf = manifest.get('performance') or {}
                if isinstance(perf, dict):
                    stats['winrate'] = _as_float(perf.get('avg_winrate')) or _as_float(perf.get('best_winrate'))
                    stats['pnl'] = (
                        _as_float(perf.get('total_profit'))
                        or _as_float(perf.get('pnl'))
                        or _as_float(perf.get('pnl_sum'))
                    )

                # Fallback для pnl: попробуем прочитать train_result.pkl по training_run_dir/run_id
                if stats.get('pnl') is None:
                    run_dir = None
                    trd = manifest.get('training_run_dir')
                    if isinstance(trd, str) and trd.strip():
                        p = Path(trd.strip())
                        if p.exists() and p.is_dir():
                            run_dir = p
                    if run_dir is None:
                        rid = manifest.get('run_id')
                        if isinstance(rid, str) and rid.strip():
                            rid = rid.strip()
                            candidates = [
                                Path('result') / 'dqn' / base_upper_symbol / 'runs' / rid,
                                Path('result') / base_upper_symbol / 'runs' / rid,
                            ]
                            for c in candidates:
                                if c.exists() and c.is_dir():
                                    run_dir = c
                                    break
                    if run_dir is not None:
                        pkl = run_dir / 'train_result.pkl'
                        if pkl.exists():
                            try:
                                st = pkl.stat()
                                if st.st_size <= 32 * 1024 * 1024:
                                    with open(pkl, 'rb') as f:
                                        doc = pickle.load(f)
                                    if isinstance(doc, dict):
                                        fs = doc.get('final_stats') or {}
                                        if isinstance(fs, dict):
                                            stats['pnl'] = (
                                                _as_float(fs.get('total_profit'))
                                                or _as_float(fs.get('pnl_sum'))
                                                or _as_float(fs.get('avg_roi'))
                                            )
                            except Exception:
                                pass
            except Exception:
                return stats
            return stats

        for root in roots:
            if not (root.exists() and root.is_dir()):
                continue
            # Перебираем только директории vN
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                enc_id = d.name
                if enc_id in seen_ids:
                    continue
                manifest = read_manifest(d)
                # Показываем только версии, у которых есть читаемый манифест
                if manifest is None:
                    continue
                stats = extract_encoder_stats(manifest, base_upper)
                seen_ids.add(enc_id)
                items.append({'id': enc_id, 'name': enc_id, 'manifest': manifest, 'stats': stats})

        # Стабильная сортировка по id (v1<v2<...)
        try:
            items.sort(key=lambda it: (it['id'][0] != 'v', int(it['id'][1:]) if (it['id'].startswith('v') and it['id'][1:].isdigit()) else it['id']))
        except Exception:
            items.sort(key=lambda it: it['id'])

        return jsonify({'success': True, 'encoders': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/api/encoders/delete')
def api_encoders_delete():
    """
    Удаление энкодеров (папки vN) из result/dqn/<SYMBOL>/encoder/<encoder_type>/.
    Payload: { symbol: "TON" | "TONUSDT", ids: ["v37","v38"], encoder_type?: "unfrozen" }
    """
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        ids = data.get('ids') or []
        encoder_type = (data.get('encoder_type') or 'unfrozen').strip() or 'unfrozen'

        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400
        if not isinstance(ids, list) or not ids:
            return jsonify({'success': False, 'error': 'ids list required'}), 400

        base_upper = symbol.split('USDT')[0].upper()
        root = Path('result') / 'dqn' / base_upper / 'encoder' / encoder_type
        if not (root.exists() and root.is_dir()):
            return jsonify({'success': False, 'error': f'encoder dir not found: {root.as_posix()}'}), 404

        # normalize ids: only v<digits>
        remove_set = set()
        bad = []
        for raw in ids:
            s = str(raw or '').strip()
            if s.startswith('v') and s[1:].isdigit():
                remove_set.add(s)
            else:
                bad.append(s)
        if not remove_set:
            return jsonify({'success': False, 'error': 'no valid ids (expected v<digits>)', 'bad': bad}), 400

        deleted = []
        errors = []
        for enc_id in sorted(remove_set, key=lambda x: int(x[1:])):
            target = (root / enc_id)
            try:
                # safety: only inside root and must be dir
                try:
                    target_res = target.resolve()
                    root_res = root.resolve()
                    if not str(target_res).startswith(str(root_res)):
                        raise RuntimeError('path outside root')
                except Exception:
                    pass
                if not (target.exists() and target.is_dir()):
                    continue
                shutil.rmtree(target, ignore_errors=False)
                deleted.append(enc_id)
            except Exception as e:
                errors.append({'id': enc_id, 'error': str(e)})

        # Try to update encoder_index.json and current.json (best-effort)
        try:
            index_path = root / 'encoder_index.json'
            if index_path.exists():
                try:
                    doc = json.loads(index_path.read_text(encoding='utf-8'))
                except Exception:
                    doc = None
                if isinstance(doc, list):
                    keep = []
                    for it in doc:
                        try:
                            v = it.get('version')
                            vid = f"v{int(v)}"
                            if vid in remove_set:
                                continue
                            keep.append(it)
                        except Exception:
                            keep.append(it)
                    tmp = index_path.with_suffix('.json.tmp')
                    tmp.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding='utf-8')
                    os.replace(tmp, index_path)

                    # update current.json if needed
                    cur_path = root / 'current.json'
                    if cur_path.exists():
                        try:
                            cur = json.loads(cur_path.read_text(encoding='utf-8'))
                        except Exception:
                            cur = {}
                        if isinstance(cur, dict):
                            cur_ver = str(cur.get('version') or '').strip()
                            if cur_ver in remove_set:
                                # pick latest from keep
                                latest = None
                                for it in keep:
                                    try:
                                        v = int(it.get('version'))
                                        latest = it if (latest is None or v > int(latest.get('version'))) else latest
                                    except Exception:
                                        continue
                                if latest is None:
                                    cur = {'version': None, 'sha256': None}
                                else:
                                    cur = {'version': f"v{int(latest.get('version'))}", 'sha256': latest.get('sha256')}
                                tmpc = cur_path.with_suffix('.json.tmp')
                                tmpc.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding='utf-8')
                                os.replace(tmpc, cur_path)
        except Exception:
            pass

        return jsonify({'success': True, 'deleted': deleted, 'errors': errors, 'bad': bad})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
