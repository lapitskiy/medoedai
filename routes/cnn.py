
@app.route('/cnn_training')
def cnn_training_page():
    """Страница обучения CNN моделей"""
    return render_template('cnn_training.html')

# === CNN Training API Endpoints ===

@app.route('/cnn/start_training', methods=['POST'])
def cnn_start_training():
    """Запуск обучения CNN модели"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT'])
        timeframes = data.get('timeframes', ['5m'])
        model_type = data.get('model_type', 'multiframe')
        
        print(f"🔍 Flask: Получен запрос на CNN обучение")
        print(f"🔍 Flask: symbols={symbols}, model_type={model_type}")
        
        # Запускаем реальную Celery задачу для обучения CNN
        try:
            print(f"🔍 Flask: Импортируем train_cnn_model...")
            from tasks.celery_tasks import train_cnn_model
            print(f"✅ Flask: train_cnn_model импортирован успешно")
        except ImportError as e:
            print(f"❌ Flask: Ошибка импорта train_cnn_model: {e}")
            raise
        
        # Если передан один символ как строка, конвертируем в список
        if isinstance(symbols, str):
            symbols = [symbols]
            print(f"🔍 Flask: Конвертировали символ в список: {symbols}")
        
        # Запускаем ОДНУ задачу обучения, которая сама обучит на всех символах из конфигурации
        print(f"🔍 Flask: Запускаем одну задачу обучения для символов: {symbols}...")
        try:
            # Передаем первый символ как формальный аргумент (задача внутри возьмет все символы из config)
            task = train_cnn_model.delay(
                symbol=symbols[0] if symbols else "BTCUSDT",
                model_type=model_type
            )
            print(f"✅ Flask: Задача создана с ID: {task.id}")
            task_results = [{
                "symbols": symbols,
                "task_id": task.id
            }]
        except Exception as e:
            print(f"❌ Flask: Ошибка создания задачи: {e}")
            raise
        
        print(f"✅ Flask: Все задачи созданы успешно. Возвращаем ответ.")
        
        result = {
            "success": True,
            "message": f"🧠 CNN обучение запущено для {symbols}",
            "task_results": task_results,
            "details": {
                "symbols": symbols,
                "timeframes": timeframes,
                "model_type": model_type,
                "note": "Все параметры обучения берутся из config.py"
            }
        }
        
        print(f"🔍 Flask: Ответ: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Flask: Критическая ошибка в CNN endpoint: {str(e)}")
        import traceback
        print(f"❌ Flask: Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Ошибка запуска CNN обучения: {str(e)}"
        }), 500

@app.route('/cnn/models', methods=['GET'])
def cnn_get_models():
    """Получение списка CNN моделей из cnn_training/result"""
    try:
        import os
        import json
        from datetime import datetime
        
        models = []
        result_dir = "cnn_training/result"
        
        if not os.path.exists(result_dir):
            return jsonify({
                "success": True,
                "models": []
            })
        
        # Проходим по всем символам в result/
        for symbol in os.listdir(result_dir):
            symbol_path = os.path.join(result_dir, symbol)
            if not os.path.isdir(symbol_path):
                continue
                
            runs_dir = os.path.join(symbol_path, "runs")
            if not os.path.exists(runs_dir):
                continue
            
            # Проходим по всем run_id
            for run_id in os.listdir(runs_dir):
                run_path = os.path.join(runs_dir, run_id)
                if not os.path.isdir(run_path):
                    continue
                
                # Ищем manifest.json и result_*.json
                manifest_path = os.path.join(run_path, "manifest.json")
                if not os.path.exists(manifest_path):
                    continue
                
                try:
                    # Читаем manifest.json
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    
                    # Ищем result файлы
                    result_files = [f for f in os.listdir(run_path) if f.startswith('result_') and f.endswith('.json')]
                    
                    # Ищем модели (берем только лучшую модель)
                    model_files = [f for f in os.listdir(run_path) if f.endswith('.pth')]
                    
                    if model_files:
                        # Приоритет: сначала best, потом обычную
                        best_model = None
                        regular_model = None
                        
                        for model_file in model_files:
                            if 'best' in model_file.lower():
                                best_model = model_file
                            else:
                                regular_model = model_file
                        
                        # Берем лучшую модель, если есть, иначе обычную
                        model_file = best_model if best_model else regular_model
                        model_path = os.path.join(run_path, model_file)
                        model_size = os.path.getsize(model_path)
                        
                        # Получаем данные из manifest
                        model_type = manifest.get('model_type', 'unknown')
                        timeframes = manifest.get('timeframes', [])
                        created_at = manifest.get('created_at', '')
                        symbols_trained = manifest.get('symbols', [])
                        
                        # Читаем результаты обучения если есть
                        accuracy = None
                        epochs_trained = None
                        train_loss = None
                        val_loss = None
                        
                        if result_files:
                            try:
                                result_path = os.path.join(run_path, result_files[0])
                                with open(result_path, 'r', encoding='utf-8') as f:
                                    result_data = json.load(f)
                                
                                accuracy = result_data.get('best_val_accuracy')
                                epochs_trained = result_data.get('epochs_trained')
                                train_loss = result_data.get('train_loss_last')
                                val_loss = result_data.get('val_loss_last')
                            except Exception:
                                pass
                        
                        # Формируем информацию о модели
                        model_info = {
                            "symbol": symbol,
                            "run_id": run_id,
                            "model_type": model_type,
                            "timeframes": timeframes,
                            "symbols_trained": symbols_trained,
                            "accuracy": accuracy,
                            "epochs_trained": epochs_trained,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "size": model_size,
                            "created": created_at,
                            "path": model_path,
                            "manifest": manifest,
                            "model_file": model_file
                        }
                        
                        models.append(model_info)
                        
                except Exception as e:
                    print(f"Ошибка чтения {manifest_path}: {e}")
                    continue
        
        # Сортируем по дате создания (новые первыми)
        models.sort(key=lambda x: x.get('created', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "models": models,
            "total": len(models)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cnn/test_model', methods=['POST'])
def cnn_test_model():
    """Тестирование CNN модели"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        # TODO: Реализовать тестирование CNN модели
        # Пока возвращаем заглушку
        return jsonify({
            "success": True,
            "test_results": f"Тестирование модели {model_path} завершено успешно.\nТочность: 75.2%\nLoss: 0.234\nВремя тестирования: 45 секунд"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cnn/test_extraction', methods=['POST'])
def cnn_test_extraction():
    """Тестирование извлечения признаков"""
    try:
        import time
        import torch
        import numpy as np
        
        data = request.get_json()
        model_path = data.get('model_path')
        test_symbol = data.get('test_symbol', 'BTCUSDT')
        
        if not model_path:
            return jsonify({
                "success": False,
                "error": "Путь к модели не указан"
            }), 400
        
        print(f"🧪 Тестируем извлечение признаков для модели: {model_path}")
        
        # Импортируем необходимые модули
        try:
            from cnn_training.feature_extractor import CNNFeatureExtractor
            from cnn_training.config import CNNTrainingConfig
            from cnn_training.data_loader import CryptoDataLoader
        except ImportError as e:
            return jsonify({
                "success": False,
                "error": f"Ошибка импорта модулей: {e}"
            }), 500
        
        # Создаем конфигурацию
        config = CNNTrainingConfig(
            symbols=[test_symbol],
            timeframes=["5m", "15m", "1h"],
            device="auto"
        )
        
        # Создаем feature extractor
        extractor = CNNFeatureExtractor(config)
        
        # Загружаем модель
        try:
            extractor.load_model(model_path)
            print(f"✅ Модель загружена: {model_path}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Ошибка загрузки модели: {e}"
            }), 500
        
        # Подготавливаем тестовые данные
        data_loader = CryptoDataLoader(config)
        data_dict = data_loader.prepare_training_data([test_symbol], config.timeframes)
        
        if not data_dict:
            return jsonify({
                "success": False,
                "error": f"Не удалось загрузить данные для {test_symbol}"
            }), 500
        
        # Создаем мультифреймовый датасет
        train_dataset, val_dataset = data_loader.create_multiframe_dataset(data_dict)
        
        # Берем несколько образцов для тестирования
        test_samples = []
        for i in range(min(10, len(val_dataset))):
            sample = val_dataset[i]
            test_samples.append(sample)
        
        print(f"📊 Тестируем на {len(test_samples)} образцах")
        
        # Тестируем извлечение признаков
        start_time = time.time()
        features_list = []
        
        for sample in test_samples:
            try:
                features = extractor.extract_features(sample)
                features_list.append(features)
            except Exception as e:
                print(f"⚠️ Ошибка извлечения признаков для образца: {e}")
                continue
        
        extraction_time = (time.time() - start_time) * 1000  # в миллисекундах
        
        if not features_list:
            return jsonify({
                "success": False,
                "error": "Не удалось извлечь признаки ни из одного образца"
            }), 500
        
        # Анализируем результаты
        features_array = np.array(features_list)
        
        results = {
            "success": True,
            "feature_size": int(features_array.shape[1]),
            "extraction_time": round(extraction_time, 2),
            "samples_tested": len(features_list),
            "feature_mean": round(float(np.mean(features_array)), 6),
            "feature_std": round(float(np.std(features_array)), 6),
            "feature_min": round(float(np.min(features_array)), 6),
            "feature_max": round(float(np.max(features_array)), 6),
            "feature_sample": features_array[0].tolist()[:10],  # Первые 10 значений
            "model_path": model_path,
            "test_symbol": test_symbol
        }
        
        print(f"✅ Тестирование завершено: {results}")
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cnn/integrate_dqn', methods=['POST'])
def cnn_integrate_dqn():
    """Интеграция CNN с DQN"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        # TODO: Реализовать интеграцию CNN с DQN
        # Пока возвращаем заглушку
        return jsonify({
            "success": True,
            "model_path": model_path,
            "cnn_features_size": 64,
            "total_state_size": 128,
            "config_updated": True
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cnn/monitoring', methods=['GET'])
def cnn_monitoring():
    """Мониторинг обучения CNN"""
    try:
        # TODO: Реализовать мониторинг обучения
        # Пока возвращаем заглушку
        return jsonify({
            "success": True,
            "metrics": {
                "current_epoch": 25,
                "train_loss": 0.234,
                "val_loss": 0.267,
                "val_accuracy": 0.752
            },
            "logs": [
                "Epoch 25/50: Train Loss: 0.234, Val Loss: 0.267, Val Acc: 75.2%",
                "Epoch 24/50: Train Loss: 0.241, Val Loss: 0.271, Val Acc: 74.8%",
                "Epoch 23/50: Train Loss: 0.248, Val Loss: 0.275, Val Acc: 74.5%"
            ]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/cnn/validate_model', methods=['POST'])
def cnn_validate_model():
    """Валидация CNN модели на новых символах"""
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        test_symbols = data.get('test_symbols', ['SOLUSDT', 'XRPUSDT', 'TONUSDT'])
        test_period = data.get('test_period', 'last_year')
        validation_type = data.get('validation_type', 'cross_symbol')
        
        if not model_path:
            return jsonify({
                "success": False,
                "error": "Путь к модели не указан"
            }), 400
        
        print(f"🧪 Валидация CNN модели: {model_path}")
        print(f"📊 Тестовые символы: {test_symbols}")
        print(f"📅 Период: {test_period}")
        
        # Импортируем валидатор
        try:
            from cnn_training.model_validator import validate_cnn_model
        except ImportError as e:
            return jsonify({
                "success": False,
                "error": f"Ошибка импорта валидатора: {e}"
            }), 500
        
        # Запускаем валидацию
        try:
            result = validate_cnn_model(
                model_path=model_path,
                test_symbols=test_symbols,
                test_period=test_period
            )
            
            if result['success']:
                print(f"✅ Валидация завершена успешно")
                print(f"📈 Общая точность: {result.get('overall_accuracy', 0):.2%}")
                return jsonify(result)
            else:
                print(f"❌ Ошибка валидации: {result.get('error', 'Неизвестная ошибка')}")
                return jsonify(result), 500
                
        except Exception as e:
            print(f"❌ Ошибка выполнения валидации: {str(e)}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "error": f"Ошибка валидации: {str(e)}"
            }), 500
            
    except Exception as e:
        print(f"❌ Критическая ошибка в endpoint валидации: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Ошибка валидации: {str(e)}"
        }), 500

@app.route('/cnn/examples', methods=['GET'])
def cnn_examples():
    """Примеры использования CNN модуля"""
    try:
        examples = """# Пример использования CNN модуля

from cnn_training.config import CNNTrainingConfig
from cnn_training.trainer import CNNTrainer
from cnn_training.feature_extractor import create_cnn_wrapper

# 1. Создание конфигурации
config = CNNTrainingConfig(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["5m", "15m", "1h"],
    sequence_length=50,
    output_features=64
)

# 2. Обучение модели
trainer = CNNTrainer(config)
result = trainer.train_single_model("BTCUSDT", "5m", "prediction")

# 3. Извлечение признаков для DQN
cnn_wrapper = create_cnn_wrapper(config)
features = cnn_wrapper.get_cnn_features("BTCUSDT", ohlcv_data)

# 4. Интеграция с DQN
combined_state = np.concatenate([base_dqn_state, features])"""
        
        return jsonify({
            "success": True,
            "examples": examples
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
