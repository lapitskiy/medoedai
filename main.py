#!/usr/bin/env python3
"""
🚀 MedoedAI - Flask веб-приложение для управления DQN торговым ботом

Запуск:
    python main.py              # Автоматический запуск
    flask run                   # Альтернатива
    FLASK_APP=main.py flask run # Альтернатива
"""

import requests
from flask import Flask, request, jsonify, render_template
from flask import redirect, url_for

import redis

from tasks.celery_tasks import celery
from celery.result import AsyncResult

import os
from tasks.celery_tasks import search_lstm_task, train_dqn, train_dqn_multi_crypto, trade_step
from utils.db_utils import clean_ohlcv_data, delete_ohlcv_for_symbol_timeframe, load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library

import logging
from flask import Response
import json

import time

import glob
import os

logging.basicConfig(level=logging.INFO)

# Создаем Flask приложение
app = Flask(__name__)

# Функция очистки Redis при запуске
def clear_redis_on_startup():
    """Очищает Redis при запуске приложения"""
    try:
        # Подключаемся к Redis (пробуем разные хосты)
        redis_hosts = ['localhost', 'redis', '127.0.0.1']
        r = None
        
        for host in redis_hosts:
            try:
                r = redis.Redis(host=host, port=6379, db=0, socket_connect_timeout=5)
                r.ping()  # Проверяем соединение
                print(f"✅ Подключились к Redis на {host}")
                break
            except Exception:
                continue
        
        if r is None:
            print("⚠️ Не удалось подключиться к Redis")
            return None
        
        # Очищаем все данные
        r.flushall()
        print("✅ Redis очищен при запуске")
        
        # Проверяем, что очистка прошла успешно
        if r.dbsize() == 0:
            print("✅ Redis пуст, готов к работе")
        else:
            print(f"⚠️ В Redis осталось {r.dbsize()} ключей")
            
        return r
            
    except Exception as e:
        print(f"⚠️ Не удалось очистить Redis: {e}")
        print("Продолжаем работу без очистки Redis")
        return None

# Инициализируем Redis клиент и очищаем при запуске
redis_client = clear_redis_on_startup()
if redis_client is None:
    # Fallback - создаем клиент без очистки
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
    except:
        redis_client = redis.Redis(host='redis', port=6379, db=0)



@app.before_request
def log_request_info():
    logging.info(f"Request from {request.remote_addr}: {request.method} {request.path}")
    logging.info("Headers: %s", dict(request.headers))  # Логируем все заголовки

@app.route("/")
def index():
    """Возвращает список всех задач Celery и их состояние"""
    task_ids = redis_client.keys("celery-task-meta-*")  # Ищем все задачи
    print(f"task_ids {print(task_ids)}")
    tasks = []

    for task_id in task_ids:
        task_id = task_id.decode("utf-8").replace("celery-task-meta-", "")
        task = AsyncResult(task_id, app=celery)

        tasks.append({
            "task_id": task_id,
            "state": task.state,
            "result": task.result if task.successful() else None
        })

    return render_template("index.html", tasks=tasks)


@app.route("/task-start-search", methods=["POST"])
def start_task():
    """Запускает задачу Celery и делает редирект на главную страницу"""
    query = request.form.get("query", "")  # Берём query из формы

    if not query:
        return redirect(url_for("index"))  # Перенаправление на главную

    task = search_lstm_task.apply_async(args=[query])  # Запускаем в Celery

    return redirect(url_for("index"))  # Перенаправление на главную


@app.route("/task-status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Проверяет статус задачи по task_id"""
    task = long_running_task.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": "PENDING", "status": "Задача в очереди"}
    elif task.state == "IN_PROGRESS":
        response = {"state": "IN_PROGRESS", "status": "Задача выполняется", "progress": task.info}
    elif task.state == "SUCCESS":
        response = {"state": "SUCCESS", "status": "Задача завершена", "result": task.result}
    elif task.state == "FAILURE":
        response = {"state": "FAILURE", "status": "Ошибка", "error": str(task.info)}
    else:
        response = {"state": task.state, "status": "Неизвестное состояние"}

    return jsonify(response)


@app.route("/start-search", methods=["POST"])
def start_parameter_search():
    try:
        response = requests.post(
            "http://parameter-search:5052/run-search",
            json={"query": "some_value"},  # Здесь передаём нужный query
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train_dqn', methods=['POST'])
def train():
    task = train_dqn.apply_async(queue="train")
    return redirect(url_for("index"))

@app.route('/train_dqn_multi_crypto', methods=['POST'])
def train_multi_crypto():
    """Запускает мультивалютное обучение DQN"""
    task = train_dqn_multi_crypto.apply_async(queue="train")
    return redirect(url_for("index"))





@app.route('/trade_dqn', methods=['POST'])
def trade():
    task = trade_step.apply_async()
    return redirect(url_for("index"))

# Новый маршрут для запуска очистки данных
@app.route('/clean_data', methods=['POST'])
def clean_data():
    timeframes_to_clean = ['5m', '15m', '1h']
    symbol_name ='BTC/USDT'
    
    max_close_change_percent = 15.0
    max_hl_range_percent = 20.0
    volume_multiplier = 10.0

    results = []
    for tf in timeframes_to_clean:
        try:
            # Вызываем функцию очистки напрямую (она больше не Celery-задача)
            result = clean_ohlcv_data(tf, symbol_name,
                                      max_close_change_percent,
                                      max_hl_range_percent,
                                      volume_multiplier)
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "message": f"Ошибка при очистке {tf} для {symbol_name}: {str(e)}"})

    return jsonify({'status': 'Очистка данных завершена для всех указанных таймфреймов.', 'results': results})


# Новый маршрут для запуска очистки данных
@app.route('/clean_db', methods=['POST'])
def clean_db():
    timeframes_to_clean = '5m'
    symbol_name ='BTC/USDT'            

    results = []
    try:
        # Вызываем функцию очистки напрямую (она больше не Celery-задача)
        delete_ohlcv_for_symbol_timeframe('BTC/USDT', timeframes_to_clean)
        delete_ohlcv_for_symbol_timeframe('BTCUSDT', timeframes_to_clean)
    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка при очистке для {symbol_name}: {str(e)}"})


    return jsonify({'status': 'Очистка базы от всех свечей завершена указанных таймфреймов.', 'results': results})

@app.route('/analyze_training_results', methods=['POST'])
def analyze_training_results():
    """Анализирует результаты обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения в папке temp/train_results
        results_dir = "temp/train_results"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена. Сначала запустите обучение.',
                'success': False
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'training_results_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Берем самый свежий файл
        latest_file = max(result_files, key=os.path.getctime)
        
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
            with open(latest_file, 'rb') as f:
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
        print(f"📊 Анализирую результаты из файла: {latest_file}")
        
        # Временно перенаправляем stdout для захвата вывода
        import io
        import sys
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            analyze_func(latest_file)
        
        analysis_output = output.getvalue()
        
        # Добавляем информацию об actual_episodes в ответ
        response_data = {
            'status': 'success',
            'message': 'Анализ результатов завершен успешно',
            'success': True,
            'file_analyzed': latest_file,
            'output': analysis_output,
            'available_files': result_files
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

@app.route('/list_training_results', methods=['GET'])
def list_training_results():
    """Возвращает список доступных файлов с результатами обучения"""
    try:
        # Ищем файлы с результатами обучения в папке temp/train_results
        results_dir = "temp/train_results"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена',
                'success': False,
                'files': []
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'training_results_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены',
                'success': False,
                'files': []
            }), 404
        
        # Получаем информацию о файлах
        files_info = []
        for file in result_files:
            stat = os.stat(file)
            files_info.append({
                'filename': file,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            })
        
        # Сортируем по дате создания (новые первыми)
        files_info.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'message': f'Найдено {len(result_files)} файлов результатов',
            'success': True,
            'files': files_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при получении списка файлов: {str(e)}',
            'success': False
        }), 500

@app.route('/analyze_bad_trades', methods=['POST'])
def analyze_bad_trades():
    """Анализирует плохие сделки из результатов обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения в папке temp/train_results
        results_dir = "temp/train_results"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена. Сначала запустите обучение.',
                'success': False
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'training_results_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Берем самый свежий файл
        latest_file = max(result_files, key=os.path.getctime)
        
        # Импортируем функцию анализа плохих сделок
        try:
            from analyze_bad_trades import analyze_bad_trades_detailed, print_bad_trades_analysis, print_detailed_recommendations
        except ImportError:
            return jsonify({
                'status': 'error',
                'message': 'Модуль анализа плохих сделок не найден. Установите зависимости.',
                'success': False
            }), 500
        
        # Загружаем результаты для анализа
        try:
            import pickle
            with open(latest_file, 'rb') as f:
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
                'file_analyzed': latest_file,
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

# Новый маршрут для запуска очистки данных
@app.route('/parser', methods=['POST'])
def parser():
    results = []
    symbol = 'BTCUSDT'
    interval = '5m'
    desired_candles = 100000
    csv_file_path = None

    try:
        # 1. Вызываем внешнюю функцию, которая создает CSV с 100 000 свечами
        csv_file_path = parser_download_and_combine_with_library(
            symbol='BTCUSDT',
            interval=interval,
            months_to_fetch=12, # Это параметр для parser_download_and_combine_with_library
            desired_candles=desired_candles
        )
        results.append({"status": "success", "message": f"CSV файл создан: {csv_file_path}"})

        # 2. Загружаем эти 100 000 свечей из CSV в базу данных
        # Эта функция теперь отвечает за перезапись/обновление данных в БД
        loaded_count = load_latest_candles_from_csv_to_db(
            file_path=csv_file_path,
            symbol_name=symbol,
            timeframe=interval
        )
        if loaded_count > 0:
            results.append({"status": "success", "message": f"Загружено {loaded_count} свечей из CSV в БД."})
        else:
            results.append({"status": "warning", "message": "Не удалось загрузить свечи из CSV в БД."})

    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка в процессе парсинга: {str(e)}"})

    response = {'status': 'Парсинг завершен', 'results': results}
    return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')

# Новый маршрут для мультивалютного скачивания свечей
@app.route('/parser_multi_crypto', methods=['POST'])
def parser_multi_crypto():
    """Скачивает свечи для всех криптовалют одновременно"""
    results = []
    interval = '5m'
    desired_candles = 100000
    
    # Список всех криптовалют для скачивания
    crypto_symbols = [
        'BTCUSDT',  # Биткоин
        'TONUSDT',  # TON
        'ETHUSDT',  # Эфириум
        'SOLUSDT',  # Solana
        'ADAUSDT',  # Cardano
        'BNBUSDT'   # Binance Coin
    ]
    
    print(f"🚀 Начинаю скачивание свечей для {len(crypto_symbols)} криптовалют...")
    print(f"📊 Таймфрейм: {interval}, Целевое количество: {desired_candles}")
    
    for i, symbol in enumerate(crypto_symbols, 1):
        try:
            print(f"\n📥 [{i}/{len(crypto_symbols)}] Скачиваю {symbol}...")
            
            # 1. Скачиваем свечи для текущей криптовалюты
            csv_file_path = parser_download_and_combine_with_library(
                symbol=symbol,
                interval=interval,
                months_to_fetch=12,
                desired_candles=desired_candles
            )
            
            if csv_file_path:
                results.append({
                    "status": "success", 
                    "symbol": symbol,
                    "message": f"CSV файл создан: {csv_file_path}"
                })
                
                # 2. Загружаем свечи в базу данных
                try:
                    loaded_count = load_latest_candles_from_csv_to_db(
                        file_path=csv_file_path,
                        symbol_name=symbol,
                        timeframe=interval
                    )
                    
                    if loaded_count > 0:
                        results.append({
                            "status": "success",
                            "symbol": symbol,
                            "message": f"Загружено {loaded_count} свечей в БД"
                        })
                        print(f"  ✅ {symbol}: {loaded_count} свечей загружено в БД")
                    else:
                        results.append({
                            "status": "warning",
                            "symbol": symbol,
                            "message": "Не удалось загрузить свечи в БД"
                        })
                        print(f"  ⚠️ {symbol}: ошибка загрузки в БД")
                        
                except Exception as db_error:
                    results.append({
                        "status": "error",
                        "symbol": symbol,
                        "message": f"Ошибка загрузки в БД: {str(db_error)}"
                    })
                    print(f"  ❌ {symbol}: ошибка БД - {db_error}")
                    
            else:
                results.append({
                    "status": "error",
                    "symbol": symbol,
                    "message": "Не удалось создать CSV файл"
                })
                print(f"  ❌ {symbol}: не удалось создать CSV")
                
        except Exception as e:
            error_msg = f"Ошибка при скачивании {symbol}: {str(e)}"
            results.append({
                "status": "error",
                "symbol": symbol,
                "message": error_msg
            })
            print(f"  ❌ {symbol}: {error_msg}")
            continue
    
    # Сводка по всем криптовалютам
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    warnings = sum(1 for r in results if r['status'] == 'warning')
    
    print(f"\n{'='*60}")
    print(f"📊 СВОДКА СКАЧИВАНИЯ")
    print(f"{'='*60}")
    print(f"✅ Успешно: {successful}")
    print(f"⚠️ Предупреждения: {warnings}")
    print(f"❌ Ошибки: {failed}")
    print(f"📈 Всего криптовалют: {len(crypto_symbols)}")
    
    response = {
        'status': 'Мультивалютное скачивание завершено',
        'summary': {
            'total_cryptos': len(crypto_symbols),
            'successful': successful,
            'warnings': warnings,
            'failed': failed
        },
        'results': results
    }
    
    return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')

@app.route('/clear_redis', methods=['POST'])
def clear_redis():
    """Очищает Redis вручную"""
    try:
        global redis_client
        redis_client.flushall()
        return jsonify({
            "success": True,
            "message": "Redis очищен успешно"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models')
def models_page():
    """Страница управления моделями"""
    return render_template('models.html')

@app.route('/create_model_version', methods=['POST'])
def create_model_version():
    """Создает новую версию модели с уникальным ID"""
    import shutil
    import uuid
    from datetime import datetime
    from pathlib import Path
    
    try:
        # Создаем папку good_models если её нет
        good_models_dir = Path('good_models')
        good_models_dir.mkdir(exist_ok=True)
        
        # Генерируем уникальный ID (4 символа)
        model_id = str(uuid.uuid4())[:4].upper()
        
        # Ищем последний файл результатов обучения
        train_results_dir = Path('temp/train_results')
        if not train_results_dir.exists():
            return jsonify({
                "success": False,
                "error": "Папка temp/train_results не найдена"
            })
        
        result_files = list(train_results_dir.glob('training_results_*.pkl'))
        if not result_files:
            return jsonify({
                "success": False,
                "error": "Файлы результатов обучения не найдены"
            })
        
        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        # Проверяем наличие файлов модели
        model_file = Path('dqn_model.pth')
        replay_file = Path('replay_buffer.pkl')
        
        if not model_file.exists():
            return jsonify({
                "success": False,
                "error": "Файл dqn_model.pth не найден"
            })
        
        if not replay_file.exists():
            return jsonify({
                "success": False,
                "error": "Файл replay_buffer.pkl не найден"
            })
        
        # Копируем файлы с новыми именами
        new_model_file = good_models_dir / f'dqn_model_{model_id}.pth'
        new_replay_file = good_models_dir / f'replay_buffer_{model_id}.pkl'
        new_result_file = good_models_dir / f'train_result_{model_id}.pkl'
        
        shutil.copy2(model_file, new_model_file)
        shutil.copy2(replay_file, new_replay_file)
        shutil.copy2(latest_result_file, new_result_file)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "files": [
                f'dqn_model_{model_id}.pth',
                f'replay_buffer_{model_id}.pkl',
                f'train_result_{model_id}.pkl'
            ]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get_models_list')
def get_models_list():
    """Возвращает список всех сохраненных моделей"""
    from pathlib import Path
    import pickle
    from datetime import datetime
    
    try:
        good_models_dir = Path('good_models')
        if not good_models_dir.exists():
            return jsonify({
                "success": True,
                "models": []
            })
        
        models = []
        
        # Ищем все файлы моделей
        model_files = list(good_models_dir.glob('dqn_model_*.pth'))
        
        for model_file in model_files:
            # Извлекаем ID модели из имени файла
            model_id = model_file.stem.replace('dqn_model_', '')
            
            # Проверяем наличие связанных файлов
            replay_file = good_models_dir / f'replay_buffer_{model_id}.pkl'
            result_file = good_models_dir / f'train_result_{model_id}.pkl'
            
            if not replay_file.exists() or not result_file.exists():
                continue  # Пропускаем неполные модели
            
            # Получаем размеры файлов
            model_size = f"{model_file.stat().st_size / 1024 / 1024:.1f} MB"
            replay_size = f"{replay_file.stat().st_size / 1024 / 1024:.1f} MB"
            result_size = f"{result_file.stat().st_size / 1024:.1f} KB"
            
            # Получаем дату создания
            creation_time = datetime.fromtimestamp(model_file.stat().st_ctime)
            date_str = creation_time.strftime('%d.%m.%Y %H:%M')
            
            # Пытаемся загрузить статистику из файла результатов
            stats = {}
            try:
                with open(result_file, 'rb') as f:
                    results = pickle.load(f)
                    if 'final_stats' in results:
                        stats = results['final_stats']
            except:
                pass  # Если не удалось загрузить статистику, оставляем пустую
            
            models.append({
                "id": model_id,
                "date": date_str,
                "files": {
                    "model": f'dqn_model_{model_id}.pth',
                    "model_size": model_size,
                    "replay": f'replay_buffer_{model_id}.pkl',
                    "replay_size": replay_size,
                    "results": f'train_result_{model_id}.pkl',
                    "results_size": result_size
                },
                "stats": stats
            })
        
        # Сортируем по дате создания (новые сначала)
        models.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            "success": True,
            "models": models
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze_model', methods=['POST'])
def analyze_model():
    """Анализирует конкретную модель"""
    import pickle
    from pathlib import Path
    
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                "success": False,
                "error": "ID модели не указан"
            })
        
        result_file = Path(f'good_models/train_result_{model_id}.pkl')
        if not result_file.exists():
            return jsonify({
                "success": False,
                "error": f"Файл результатов для модели {model_id} не найден"
            })
        
        # Загружаем результаты
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        
        # Формируем анализ
        analysis = f"""📊 АНАЛИЗ МОДЕЛИ {model_id}
{'='*50}

📅 Дата обучения: {results.get('training_date', 'Неизвестно')}
⏱️ Время обучения: {results.get('total_training_time', 0) / 3600:.1f} часов
🎯 Эпизодов: {results.get('actual_episodes', results.get('episodes', 'Неизвестно'))}

📈 СТАТИСТИКА:
"""
        
        if 'final_stats' in results:
            stats = results['final_stats']
            analysis += f"""
• Winrate: {stats.get('winrate', 0) * 100:.1f}%
• P/L Ratio: {stats.get('pl_ratio', 0):.2f}
• Сделок: {stats.get('trades_count', 0)}
• Средняя прибыль: {stats.get('avg_profit', 0) * 100:.2f}%
• Средний убыток: {stats.get('avg_loss', 0) * 100:.2f}%
• Плохих сделок: {stats.get('bad_trades_count', 0)}
"""
        
        if 'all_trades' in results:
            trades = results['all_trades']
            analysis += f"""
📊 ДЕТАЛЬНАЯ СТАТИСТИКА СДЕЛОК:
• Всего сделок: {len(trades)}
• Прибыльных: {sum(1 for t in trades if t.get('roi', 0) > 0)}
• Убыточных: {sum(1 for t in trades if t.get('roi', 0) < 0)}
• Нейтральных: {sum(1 for t in trades if abs(t.get('roi', 0)) < 0.001)}
"""
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_model', methods=['POST'])
def delete_model():
    """Удаляет модель и все связанные файлы"""
    from pathlib import Path
    import os
    
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                "success": False,
                "error": "ID модели не указан"
            })
        
        good_models_dir = Path('good_models')
        
        # Удаляем все файлы модели
        files_to_delete = [
            good_models_dir / f'dqn_model_{model_id}.pth',
            good_models_dir / f'replay_buffer_{model_id}.pkl',
            good_models_dir / f'train_result_{model_id}.pkl'
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                os.remove(file_path)
                deleted_files.append(file_path.name)
        
        if not deleted_files:
            return jsonify({
                "success": False,
                "error": f"Файлы модели {model_id} не найдены"
            })
        
        return jsonify({
            "success": True,
            "message": f"Модель {model_id} удалена",
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Автоматический запуск Flask сервера
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Получаем порт из переменной окружения
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 Запускаю Flask сервер на порту {port}...")
    print(f"🌐 Откройте: http://localhost:{port}")
    print(f"🔧 Debug режим: {'ВКЛЮЧЕН' if debug_mode else 'ОТКЛЮЧЕН'}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
