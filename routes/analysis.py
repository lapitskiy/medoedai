
@app.route('/api/analysis/qvalues_vs_pnl', methods=['GET'])
def analysis_qvalues_vs_pnl():
    """Возвращает датасет BUY→SELL пар с P&L и q_values для анализа порогов.

    Params:
      symbol: опционально, фильтр по символу
      limit_trades: максимум сделок (по умолчанию 400)
      limit_predictions: максимум предсказаний (по умолчанию 2000)
      tolerance_buckets: допуск по времени в 5м свечах (по умолчанию 1)
    """
    try:
        symbol = request.args.get('symbol')
        limit_trades = request.args.get('limit_trades', 400, type=int)
        limit_predictions = request.args.get('limit_predictions', 2000, type=int)
        tolerance_buckets = request.args.get('tolerance_buckets', 1, type=int)

        # Переиспользуем логику сопоставления
        with app.test_request_context(
            f"/api/trades/matched_full?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
        ):
            resp_raw = get_matched_full_trades()
        # Нормализуем (Response | (Response, status))
        if isinstance(resp_raw, tuple):
            resp_obj, status_code = resp_raw
        else:
            resp_obj, status_code = resp_raw, getattr(resp_raw, 'status_code', 200)
        if status_code != 200:
            return resp_obj, status_code
        payload = resp_obj.get_json() or {}
        if not payload.get('success'):
            return jsonify(payload), 200

        rows = []
        counters = {
            'pairs_total': 0,
            'pairs_with_buy_q': 0,
            'pairs_with_sell_q': 0
        }
        for p in payload.get('pairs', []):
            counters['pairs_total'] += 1
            pred_buy = p.get('pred_buy') or {}
            pred_sell = p.get('pred_sell') or {}
            q_vals_buy = pred_buy.get('q_values') or []
            q_vals_sell = pred_sell.get('q_values') or []
            # Явно берем Q(BUY) и разрыв относительно max(HOLD, SELL)
            q_buy = None
            q_buy_gap = None
            try:
                if isinstance(q_vals_buy, list) and len(q_vals_buy) >= 2:
                    q_buy = float(q_vals_buy[1])  # [hold, buy, sell]
                    other = []
                    if len(q_vals_buy) >= 1:
                        other.append(float(q_vals_buy[0]))
                    if len(q_vals_buy) >= 3:
                        other.append(float(q_vals_buy[2]))
                    if other:
                        q_buy_gap = q_buy - max(other)
                    counters['pairs_with_buy_q'] += 1
            except Exception:
                q_buy = None
                q_buy_gap = None

            # Для SELL: Q(SELL) и разрыв относительно max(HOLD, BUY)
            q_sell = None
            q_sell_gap = None
            try:
                if isinstance(q_vals_sell, list) and len(q_vals_sell) >= 3:
                    q_sell = float(q_vals_sell[2])
                    other_s = []
                    if len(q_vals_sell) >= 1:
                        other_s.append(float(q_vals_sell[0]))
                    if len(q_vals_sell) >= 2:
                        other_s.append(float(q_vals_sell[1]))
                    if other_s:
                        q_sell_gap = q_sell - max(other_s)
                    counters['pairs_with_sell_q'] += 1
            except Exception:
                q_sell = None
                q_sell_gap = None

            rows.append({
                'symbol': p.get('symbol'),
                'entry_time': p.get('entry_time'),
                'exit_time': p.get('exit_time'),
                'pnl_abs': p.get('pnl_abs'),
                'pnl_pct': p.get('pnl_pct'),
                'buy_confidence': pred_buy.get('confidence'),
                'sell_confidence': pred_sell.get('confidence'),
                'buy_q_values': q_vals_buy,
                'sell_q_values': pred_sell.get('q_values') or [],
                'buy_max_q': q_buy,
                'buy_gap_q': q_buy_gap,
                'sell_max_q': q_sell,
                'sell_gap_q': q_sell_gap,
            })

        return jsonify({ 'success': True, 'rows': rows, 'total': len(rows), 'counters': counters }), 200

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

@app.route('/api/analysis/qgate_suggest', methods=['GET'])
def analysis_qgate_suggest():
    """Подбирает пороги T1/T2 (maxQ/gapQ) по сетке квантилей без внешних зависимостей.

    Params: такие же, как у /api/analysis/qvalues_vs_pnl
    Возвращает: { T1, T2, hit_rate, n, score }
    """
    try:
        symbol = request.args.get('symbol')
        limit_trades = request.args.get('limit_trades', 400, type=int)
        limit_predictions = request.args.get('limit_predictions', 2000, type=int)
        tolerance_buckets = request.args.get('tolerance_buckets', 1, type=int)
        metric = request.args.get('metric', 'hit_rate')  # hit_rate | pnl_sum | pnl_per_trade
        min_n = request.args.get('min_n', 20, type=int)
        grid_points = request.args.get('grid_points', 15, type=int)
        side = request.args.get('side', 'buy')  # buy | sell

        # Получаем датасет
        with app.test_request_context(
            f"/api/analysis/qvalues_vs_pnl?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
        ):
            resp_raw = analysis_qvalues_vs_pnl()
        # Нормализуем (Response | (Response, status))
        if isinstance(resp_raw, tuple):
            resp_obj, status_code = resp_raw
        else:
            resp_obj, status_code = resp_raw, getattr(resp_raw, 'status_code', 200)
        if status_code != 200:
            return resp_obj, status_code
        js = resp_obj.get_json() or {}
        if not js.get('success'):
            return jsonify(js), 200

        rows = js.get('rows', [])
        # Преобразуем в простые массивы
        maxqs = []
        gapqs = []
        wins = []
        for r in rows:
            if side == 'sell':
                max_q = r.get('sell_max_q')
                gap_q = r.get('sell_gap_q')
            else:
                max_q = r.get('buy_max_q')
                gap_q = r.get('buy_gap_q')
            pnl_abs = r.get('pnl_abs')
            if max_q is None:
                continue
            maxqs.append(float(max_q))
            gapqs.append(float(gap_q) if gap_q is not None else float('nan'))
            wins.append(1.0 if (pnl_abs is not None and float(pnl_abs) > 0.0) else 0.0)

        if not maxqs:
            return jsonify({ 'success': False, 'error': 'Недостаточно данных' }), 200

        # Квантильные сетки
        def quantiles(arr, qs):
            a = sorted([x for x in arr if not (x is None or (isinstance(x, float) and (x != x)))])
            if not a:
                return []
            res = []
            n = len(a)
            for q in qs:
                if q <= 0:
                    res.append(a[0])
                elif q >= 1:
                    res.append(a[-1])
                else:
                    idx = int(q * (n - 1))
                    res.append(a[idx])
            return res

        gp = max(5, grid_points)
        qs = [0.2 + i*(0.7/(gp-1)) for i in range(gp)]  # 0.2..0.9, gp точек
        maxq_vals = quantiles(maxqs, qs)
        has_gap = any(not (g != g) for g in gapqs)  # есть хотя бы одно не-NaN
        gapq_clean = [g for g in gapqs if not (isinstance(g, float) and (g != g))]
        gapq_vals = quantiles(gapq_clean, qs) if gapq_clean else [0.0]

        best = None
        total = len(maxqs)
        for t1 in maxq_vals:
            for t2 in gapq_vals:
                selected = 0
                wins_sel = 0
                pnl_sum = 0.0
                for i in range(total):
                    ok = (maxqs[i] >= t1)
                    if has_gap and i < len(gapqs) and not (isinstance(gapqs[i], float) and (gapqs[i] != gapqs[i])):
                        ok = ok and (gapqs[i] >= t2)
                    if ok:
                        selected += 1
                        wins_sel += wins[i]
                        # добавляем pnl, если доступен
                        try:
                            pnl_val = float(rows[i].get('pnl_abs') or 0.0)
                        except Exception:
                            pnl_val = 0.0
                        pnl_sum += pnl_val
                if selected < min_n:
                    continue
                hit = wins_sel / selected if selected > 0 else 0.0
                if metric == 'hit_rate':
                    score = hit * (selected / total)
                elif metric == 'pnl_sum':
                    score = pnl_sum
                else:  # pnl_per_trade
                    score = (pnl_sum / selected) if selected else -1e9
                if (best is None) or (score > best['score']):
                    best = { 'T1': float(t1), 'T2': float(t2), 'hit_rate': float(hit), 'n': int(selected), 'score': float(score), 'pnl_sum': float(pnl_sum), 'pnl_per_trade': float((pnl_sum/selected) if selected else 0.0) }

        if not best:
            # Эвристический фолбэк по квантилям, чтобы вернуть разумные пороги даже на малой выборке
            def quantile_one(arr, q):
                a = sorted([x for x in arr if not (isinstance(x, float) and (x != x))])
                if not a:
                    return None
                if q <= 0:
                    return a[0]
                if q >= 1:
                    return a[-1]
                idx = int(q * (len(a) - 1))
                return a[idx]

            t1_fb = quantile_one(maxqs, 0.7) or maxqs[0]
            t2_fb = quantile_one(gapq_clean if gapq_clean else [0.0], 0.6) if has_gap else 0.0

            # Оценим метрики для фолбэка
            selected = 0
            wins_sel = 0
            pnl_sum = 0.0
            for i in range(total):
                ok = (maxqs[i] >= t1_fb)
                if has_gap and i < len(gapqs) and not (isinstance(gapqs[i], float) and (gapqs[i] != gapqs[i])):
                    ok = ok and (gapqs[i] >= t2_fb)
                if ok:
                    selected += 1
                    wins_sel += wins[i]
                    try:
                        pnl_val = float(rows[i].get('pnl_abs') or 0.0)
                    except Exception:
                        pnl_val = 0.0
                    pnl_sum += pnl_val
            hit = wins_sel / selected if selected else 0.0
            best = { 'T1': float(t1_fb), 'T2': float(t2_fb), 'hit_rate': float(hit), 'n': int(selected), 'score': float(hit * (selected/total) if total else 0.0), 'pnl_sum': float(pnl_sum), 'pnl_per_trade': float((pnl_sum/selected) if selected else 0.0) }
            approx = True
        else:
            approx = False

        # Сводки распределений по win/loss
        def summarize(arr, mask):
            vals = [float(arr[i]) for i in range(len(arr)) if mask[i] and not (isinstance(arr[i], float) and (arr[i] != arr[i]))]
            if not vals:
                return { 'n': 0 }
            vals_sorted = sorted(vals)
            def q(p):
                if not vals_sorted:
                    return None
                idx = int(p * (len(vals_sorted)-1))
                return vals_sorted[idx]
            return {
                'n': len(vals_sorted),
                'mean': sum(vals_sorted)/len(vals_sorted),
                'q10': q(0.1), 'q50': q(0.5), 'q90': q(0.9)
            }

        mask_win = [w == 1.0 for w in wins]
        mask_loss = [w == 0.0 for w in wins]
        summary = {
            'q_buy_win': summarize(maxqs, mask_win),
            'q_buy_loss': summarize(maxqs, mask_loss),
            'gap_win': summarize(gapqs, mask_win),
            'gap_loss': summarize(gapqs, mask_loss)
        }

        env_str = (f"QGATE_{'SELL_' if side=='sell' else ''}MAXQ={best['T1']:.6f} "
                   f"QGATE_{'SELL_' if side=='sell' else ''}GAPQ={best['T2']:.6f}")
        # Дополнительные счетчики доступности q-значений
        counters = None
        try:
            with app.test_request_context(
                f"/api/analysis/qvalues_vs_pnl?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
            ):
                resp_raw2 = analysis_qvalues_vs_pnl()
            if isinstance(resp_raw2, tuple):
                resp_obj2, status_code2 = resp_raw2
            else:
                resp_obj2, status_code2 = resp_raw2, getattr(resp_raw2, 'status_code', 200)
            if status_code2 == 200:
                js2 = resp_obj2.get_json() or {}
                counters = js2.get('counters')
        except Exception:
            counters = None

        return jsonify({ 'success': True, 'suggestion': best, 'env': env_str, 'summary': summary, 'side': side, 'approx': approx, 'counters': counters }), 200

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

@app.route('/analyze_bad_trades', methods=['POST'])
def analyze_bad_trades():
    """Анализирует плохие сделки из результатов обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения в папке result
        results_dir = "result"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена. Сначала запустите обучение.',
                'success': False
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Выбор конкретного файла из тела запроса (если указан)
        data = request.get_json(silent=True) or {}
        requested_file = data.get('file')
        selected_file = None
        if requested_file:
            safe_path = os.path.abspath(requested_file)
            base_path = os.path.abspath(results_dir)
            if safe_path.startswith(base_path) and os.path.exists(safe_path):
                selected_file = safe_path
        if not selected_file:
            selected_file = max(result_files, key=os.path.getctime)
        
        # Импортируем функцию анализа плохих сделок (с fallback без зависимостей)
        try:
            from analyze_bad_trades import analyze_bad_trades_detailed, print_bad_trades_analysis, print_detailed_recommendations
        except ImportError as e:
            app.logger.warning(f"Fallback analyze_bad_trades (ImportError: {e})")
            import numpy as _np
            def analyze_bad_trades_detailed(trades):
                if not trades:
                    return {
                        'bad_trades': [], 'bad_trades_count': 0,
                        'bad_trades_percentage': 0.0, 'avg_bad_roi': 0.0,
                        'avg_bad_duration': 0.0, 'loss_distribution': {},
                    }
                total = len(trades)
                bad = [t for t in trades if float(t.get('roi', 0.0)) < 0.0]
                bad_rois = [float(t.get('roi', 0.0)) for t in bad]
                bad_durs = [float(t.get('duration', 0.0)) for t in bad if t.get('duration') is not None]
                return {
                    'bad_trades': bad,
                    'bad_trades_count': len(bad),
                    'bad_trades_percentage': (len(bad)/total*100.0) if total else 0.0,
                    'avg_bad_roi': float(_np.mean(bad_rois)) if bad_rois else 0.0,
                    'avg_bad_duration': float(_np.mean(bad_durs)) if bad_durs else 0.0,
                    'loss_distribution': {
                        'very_small_losses': sum(1 for r in bad_rois if -0.002 <= r < 0),
                        'small_losses':      sum(1 for r in bad_rois if -0.01  <= r < -0.002),
                        'medium_losses':     sum(1 for r in bad_rois if -0.03  <= r < -0.01),
                        'large_losses':      sum(1 for r in bad_rois if r < -0.03),
                    }
                }
            def print_bad_trades_analysis(analysis):
                print("📉 АНАЛИЗ ПЛОХИХ СДЕЛОК")
                print(f"Всего плохих сделок: {analysis.get('bad_trades_count', 0)}")
                print(f"Процент плохих сделок: {analysis.get('bad_trades_percentage', 0):.2f}%")
                print(f"Средний ROI плохих сделок: {analysis.get('avg_bad_roi', 0.0)*100:.4f}%")
                print(f"Средняя длительность плохих сделок: {analysis.get('avg_bad_duration', 0.0):.1f} мин")
            def print_detailed_recommendations(analysis):
                print("🧠 РЕКОМЕНДАЦИИ: ")
        
        # Загружаем результаты для анализа
        try:
            import pickle
            with open(selected_file, 'rb') as f:
                results = pickle.load(f)
            
            # Проверяем наличие сделок
            if 'all_trades' not in results:
                return jsonify({
                    'status': 'error',
                    'message': 'В файле нет данных о сделках',
                    'success': False
                }), 404
            
            trades = results['all_trades']
            
            # Анализируем плохие сделки
            bad_trades_analysis = analyze_bad_trades_detailed(trades)
            
            # Добавляем все сделки для сравнения
            bad_trades_analysis['all_trades'] = trades
            
            # Временно перенаправляем stdout для захвата вывода
            import io
            import sys
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                print_bad_trades_analysis(bad_trades_analysis)
                print_detailed_recommendations(bad_trades_analysis)
            
            analysis_output = output.getvalue()
            
            # Подготавливаем ответ
            response_data = {
                'status': 'success',
                'message': 'Анализ плохих сделок завершен успешно',
                'success': True,
                'file_analyzed': selected_file,
                'output': analysis_output,
                'bad_trades_count': bad_trades_analysis.get('bad_trades_count', 0),
                'bad_trades_percentage': bad_trades_analysis.get('bad_trades_percentage', 0),
                'analysis_summary': {
                    'total_trades': len(trades),
                    'bad_trades': bad_trades_analysis.get('bad_trades_count', 0),
                    'avg_bad_roi': bad_trades_analysis.get('avg_bad_roi', 0),
                    'avg_bad_duration': bad_trades_analysis.get('avg_bad_duration', 0),
                    'loss_distribution': bad_trades_analysis.get('loss_distribution', {})
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка при анализе файла: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при анализе плохих сделок: {str(e)}',
            'success': False
        }), 500

@app.route('/analyze_training_results', methods=['POST'])
def analyze_training_results():
    """Анализирует результаты обучения DQN модели"""
    try:
        # Новая логика: сначала пробуем конкретный файл из запроса (поддержка runs/*/train_result.pkl)
        results_dir = "result"
        data = request.get_json(silent=True) or {}
        requested_file = (data.get('file') or '').strip()
        selected_file = None
        if requested_file:
            # Нормализуем путь; поддерживаем Windows-разделители
            req = requested_file.replace('\\', '/')
            if not os.path.isabs(req):
                # Если уже начинается с result/ — трактуем как относительный к корню проекта
                if req.startswith('result/'):
                    cand = os.path.abspath(req)
                else:
                    # Иначе считаем относительным к каталогу result/
                    cand = os.path.abspath(os.path.join(results_dir, req))
            else:
                cand = os.path.abspath(req)
            # Принимаем только пути внутри result/
            base_path = os.path.abspath(results_dir)
            if cand.startswith(base_path) and os.path.exists(cand):
                selected_file = cand
        # Если файл не указан/не найден — ищем новый формат result/<SYMBOL>/runs/*/train_result.pkl
        if not selected_file:
            run_results = []
            if os.path.exists(results_dir):
                for sym in os.listdir(results_dir):
                    rdir = os.path.join(results_dir, sym, 'runs')
                    if not os.path.isdir(rdir):
                        continue
                    for run_id in os.listdir(rdir):
                        p = os.path.join(rdir, run_id, 'train_result.pkl')
                        if os.path.exists(p):
                            run_results.append(p)
            # Фоллбек на старый плоский формат
            flat_results = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
            all_candidates = (run_results or []) + (flat_results or [])
            if not all_candidates:
                return jsonify({'status': 'error','message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.','success': False}), 404
            selected_file = max(all_candidates, key=os.path.getctime)
        
        # Импортируем функцию анализа
        try:
            from analyze_training_results import analyze_training_results as analyze_func
        except ImportError:
            # Если модуль не найден, создаем простую функцию анализа
            def analyze_func(filename):
                print(f"📊 Анализ файла: {filename}")
                print("⚠️ Модуль анализа не найден. Установите matplotlib и numpy.")
                print("💡 Для полного анализа используйте: pip install matplotlib numpy")
                return "Анализ недоступен - установите зависимости"
        
        # Загружаем результаты для дополнительного анализа
        try:
            import pickle
            with open(selected_file, 'rb') as f:
                results = pickle.load(f)
            
            # Добавляем информацию об actual_episodes
            if 'actual_episodes' in results:
                actual_episodes = results['actual_episodes']
                planned_episodes = results['episodes']
                
                if actual_episodes < planned_episodes:
                    print(f"⚠️ Early Stopping сработал! Обучение остановлено на {actual_episodes} эпизоде из {planned_episodes}")
                else:
                    print(f"✅ Обучение завершено полностью: {actual_episodes} эпизодов")                    
        except Exception as e:
            print(f"⚠️ Не удалось загрузить детали результатов: {e}")
        
        # Запускаем анализ
        print(f"📊 Анализирую результаты из файла: {selected_file}")
        
        # Временно перенаправляем stdout для захвата вывода
        import io
        import sys
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            analyze_func(selected_file)
        
        analysis_output = output.getvalue()
        
        # Добавляем информацию об actual_episodes в ответ
        response_data = {
            'status': 'success',
            'message': 'Анализ результатов завершен успешно',
            'success': True,
            'file_analyzed': selected_file,
            'output': analysis_output,
            'available_files': []
        }
        
        # Добавляем информацию об эпизодах если доступна
        try:
            # Добавляем episode_winrates_count для правильного определения early stopping
            if 'episode_winrates' in results:
                response_data['episode_winrates_count'] = len(results['episode_winrates'])
                print(f"🔍 episode_winrates_count: {response_data['episode_winrates_count']}")
            
            if 'actual_episodes' in results:
                response_data['actual_episodes'] = results['actual_episodes']
                response_data['episodes'] = results['episodes']
                
                # Проверяем несоответствие actual_episodes и episode_winrates_count
                if 'episode_winrates_count' in response_data:
                    if response_data['actual_episodes'] != response_data['episode_winrates_count']:
                        print(f"⚠️ НЕСООТВЕТСТВИЕ: actual_episodes={response_data['actual_episodes']}, episode_winrates_count={response_data['episode_winrates_count']}")
                        # Исправляем actual_episodes на правильное значение
                        response_data['actual_episodes'] = response_data['episode_winrates_count']
                        print(f"🔧 Исправлено: actual_episodes = {response_data['actual_episodes']}")
            else:
                # Если actual_episodes не найден, пытаемся извлечь из логов
                if 'output' in response_data:
                    output_text = response_data['output']
                    # Ищем Early stopping в логах
                    if 'Early stopping triggered after' in output_text:
                        import re
                        early_stopping_match = re.search(r'Early stopping triggered after (\d+) episodes', output_text)
                        if early_stopping_match:
                            actual_episodes = int(early_stopping_match.group(1))
                            # Ищем планируемое количество эпизодов
                            episodes_match = re.search(r'Количество эпизодов: (\d+)', output_text)
                            if episodes_match:
                                planned_episodes = int(episodes_match.group(1))
                                response_data['actual_episodes'] = actual_episodes
                                response_data['episodes'] = planned_episodes
                                print(f"🔍 Извлечено из логов: actual_episodes={actual_episodes}, episodes={planned_episodes}")
        except Exception as e:
            print(f"⚠️ Ошибка при извлечении actual_episodes: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при анализе результатов: {str(e)}',
            'success': False
        }), 500
