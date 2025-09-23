"""
Модуль валидации CNN моделей на новых символах
Проверяет обобщающую способность моделей на данных, не участвовавших в обучении
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ccxt

from .models import TradingCNN, MultiTimeframeCNN
from .data_loader import CryptoDataLoader
from .feature_extractor import CNNFeatureExtractor

logger = logging.getLogger(__name__)

class CNNModelValidator:
    """Валидатор CNN моделей для проверки на новых символах"""
    
    def __init__(self, config_path: str = None):
        """
        Инициализация валидатора
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config_path = config_path
        
        # Символы для валидации (не участвовавшие в обучении)
        self.validation_symbols = ['SOLUSDT', 'XRPUSDT', 'TONUSDT']
        
        # Временные фреймы для тестирования
        self.timeframes = ['5m', '15m', '1h']
        
        # Создаем конфигурацию для data loader
        from .config import CNNTrainingConfig
        data_config = CNNTrainingConfig(
            symbols=self.validation_symbols,
            timeframes=self.timeframes,
            device="cpu"
        )
        self.data_loader = CryptoDataLoader(data_config)
        
        # Создаем feature extractor с той же конфигурацией
        self.feature_extractor = CNNFeatureExtractor(data_config)
        
        # Периоды для тестирования
        self.test_periods = {
            'last_year': 365,  # Последний год
            'last_6_months': 180,  # Последние 6 месяцев
            'last_3_months': 90,   # Последние 3 месяца
            'last_month': 30       # Последний месяц
        }
    
    def validate_model(self, model_path: str, test_symbols: List[str] = None, 
                      test_period: str = 'last_year') -> Dict[str, Any]:
        """
        Валидация CNN модели на новых символах
        
        Args:
            model_path: Путь к модели
            test_symbols: Символы для тестирования
            test_period: Период тестирования
            
        Returns:
            Результаты валидации
        """
        try:
            logger.info(f"🧪 Начинаем валидацию модели: {model_path}")
            
            # Загружаем модель
            model = self._load_model(model_path)
            if not model:
                return {'success': False, 'error': 'Не удалось загрузить модель'}
            
            # Определяем символы для тестирования
            if not test_symbols:
                test_symbols = self.validation_symbols
            
            # Получаем период тестирования
            days_back = self.test_periods.get(test_period, 365)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"📊 Тестируем на символах: {test_symbols}")
            logger.info(f"📅 Период: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            
            # Валидируем на каждом символе
            symbol_results = []
            total_samples = 0
            total_correct = 0
            
            for symbol in test_symbols:
                logger.info(f"🔍 Тестируем символ: {symbol}")
                
                symbol_result = self._validate_symbol(
                    model, symbol, start_date, end_date
                )
                
                if symbol_result['success']:
                    symbol_results.append(symbol_result)
                    total_samples += symbol_result['samples_tested']
                    total_correct += symbol_result['correct_predictions']
                else:
                    logger.warning(f"❌ Ошибка тестирования {symbol}: {symbol_result['error']}")
            
            # Вычисляем общую статистику
            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
            
            # Генерируем рекомендации
            recommendation = self._generate_recommendation(symbol_results, overall_accuracy)
            
            # Детальный анализ результатов
            try:
                from .validation_analyzer import analyze_validation_results
                detailed_analysis = analyze_validation_results({
                    'overall_accuracy': overall_accuracy,
                    'total_samples': total_samples,
                    'symbol_results': symbol_results,
                    'test_symbols': test_symbols,
                    'test_period': test_period
                })
            except Exception as e:
                logger.warning(f"⚠️ Не удалось выполнить детальный анализ: {str(e)}")
                detailed_analysis = {}
            
            result = {
                'success': True,
                'model_path': model_path,
                'test_symbols': test_symbols,
                'test_period': test_period,
                'total_samples': total_samples,
                'overall_accuracy': overall_accuracy,
                'symbol_results': symbol_results,
                'recommendation': recommendation,
                'detailed_analysis': detailed_analysis,
                'validation_date': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Валидация завершена. Общая точность: {overall_accuracy:.2%}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _load_model(self, model_path: str) -> Optional[Union[TradingCNN, MultiTimeframeCNN]]:
        """Загрузка CNN модели"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"❌ Модель не найдена: {model_path}")
                return None
            
            # Загружаем конфигурацию модели
            manifest_path = os.path.join(os.path.dirname(model_path), 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            else:
                manifest = {}
            
            # Загружаем checkpoint для определения архитектуры
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Определяем тип модели по ключам в state_dict
            model_type = manifest.get('model_type', 'unknown')
            is_multiframe = any(key.startswith('cnn_models.') for key in state_dict.keys())
            
            logger.info(f"🔍 Тип модели: {model_type}, Мультифреймовая: {is_multiframe}")
            
            # Создаем модель с правильной архитектурой
            config = manifest.get('config', {})
            
            if is_multiframe or model_type == 'multiframe':
                # Мультифреймовая модель
                timeframes = config.get('timeframes', ['5m', '15m', '1h'])
                sequence_lengths = {
                    '5m': 50,
                    '15m': 40, 
                    '1h': 30
                }
                
                model = MultiTimeframeCNN(
                    input_channels=config.get('input_channels', 4),
                    hidden_channels=config.get('hidden_channels', [32, 64, 128]),
                    output_features=config.get('output_features', 64),
                    sequence_lengths=sequence_lengths,
                    dropout_rate=config.get('dropout_rate', 0.2),
                    use_batch_norm=config.get('use_batch_norm', True)
                )
                logger.info("✅ Создана мультифреймовая модель")
            else:
                # Обычная модель
                model = TradingCNN(
                    input_channels=config.get('input_channels', 4),
                    hidden_channels=config.get('hidden_channels', [32, 64, 128]),
                    output_features=config.get('output_features', 64),
                    dropout_rate=config.get('dropout_rate', 0.2),
                    use_batch_norm=config.get('use_batch_norm', True)
                )
                logger.info("✅ Создана обычная модель")
            
            # Загружаем веса
            try:
                model.load_state_dict(state_dict)
                logger.info("✅ Веса модели загружены успешно")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки весов: {str(e)}")
                return None
            
            model.eval()
            logger.info(f"✅ Модель загружена: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
            return None
    
    def _validate_symbol(self, model: Union[TradingCNN, MultiTimeframeCNN], symbol: str, 
                        start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Валидация модели на конкретном символе"""
        try:
            # Загружаем данные для символа
            data = self._load_symbol_data(symbol, start_date, end_date)
            if data is None or len(data) < 100:
                return {
                    'success': False,
                    'error': f'Недостаточно данных для {symbol}'
                }
            
            # Подготавливаем данные для тестирования
            test_data = self._prepare_test_data(data, symbol)
            if not test_data:
                return {
                    'success': False,
                    'error': f'Не удалось подготовить данные для {symbol}'
                }
            
            # Тестируем модель (возвращает также метки для оценки)
            predictions, confidences, patterns, labels_eval = self._test_model_on_data(
                model, test_data
            )
            
            # Вычисляем метрики
            accuracy = self._calculate_accuracy(labels_eval, predictions)
            
            # Анализируем ошибки (features могут отсутствовать для мультифрейма)
            error_analysis = self._analyze_errors(
                test_data['labels'], predictions, test_data.get('features', np.array([]))
            )
            
            result = {
                'success': True,
                'symbol': symbol,
                'samples_tested': len(predictions),
                'correct_predictions': int(accuracy * len(predictions)),
                'accuracy': accuracy,
                'avg_confidence': np.mean(confidences),
                'test_period': f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
                'patterns_detected': patterns,
                'error_analysis': error_analysis
            }
            
            logger.info(f"✅ {symbol}: точность {accuracy:.2%}, образцов {len(predictions)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_symbol_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime) -> Optional[pd.DataFrame]:
        """Загрузка данных для символа теми же функциями, что и при обучении"""
        try:
            all_data = []

            for timeframe in self.timeframes:
                try:
                    # 1) Загружаем сырые данные как при обучении
                    df_raw = self.data_loader.load_symbol_data(symbol, timeframe)
                    if df_raw is None or df_raw.empty:
                        continue

                    # 2) Фильтруем по периоду ДО препроцессинга, используя доступные временные колонки
                    df = df_raw.copy()
                    ts_col = None
                    for cand_col in [
                        'timestamp', 'open_time', 'start_time', 'date', 'datetime'
                    ]:
                        if cand_col in df.columns:
                            ts_col = cand_col
                            break

                    if ts_col is not None:
                        try:
                            sample = df[ts_col].dropna().iloc[0]
                            unit = None
                            # Определяем unit по масштабу (сек/мс)
                            if isinstance(sample, (int, float, np.integer, np.floating)):
                                val = float(sample)
                                if val > 1e11:
                                    unit = 'ms'   # миллисекунды
                                elif val > 1e9:
                                    unit = 's'    # секунды (за запас возьмем)
                                else:
                                    unit = 's'
                                df[ts_col] = pd.to_datetime(df[ts_col], unit=unit, errors='coerce')
                            else:
                                # Пытаемся парсить как строку
                                df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce', utc=False)
                            # Фильтруем по периоду
                            df = df[(df[ts_col] >= start_date) & (df[ts_col] <= end_date)]
                        except Exception:
                            # В случае любой ошибки не фильтруем по времени
                            pass

                    if df is None or df.empty:
                        continue

                    # 3) Препроцессинг как в обучении (удалит timestamp колонки)
                    df = self.data_loader.preprocess_dataframe(df)

                    if df is not None and not df.empty:
                        df['timeframe'] = timeframe
                        all_data.append(df)

                except Exception as e:
                    logger.warning(f"⚠️ Не удалось загрузить {symbol} {timeframe}: {str(e)}")
                    continue

            if not all_data:
                return None

            # Объединяем без сортировки по времени (временные колонки уже удалены препроцессингом)
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных {symbol}: {str(e)}")
            return None
    
    def _prepare_test_data(self, data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Подготовка данных для тестирования (последовательности OHLCV как при обучении)"""
        try:
            # Разбиваем объединенный DataFrame на фреймы
            data_by_tf: Dict[str, pd.DataFrame] = {}
            for tf in self.timeframes:
                df_tf = data[data.get('timeframe') == tf] if 'timeframe' in data.columns else data
                if df_tf is not None and not df_tf.empty:
                    # Держим только OHLCV
                    df_tf = df_tf[['open', 'high', 'low', 'close', 'volume']]
                    data_by_tf[tf] = df_tf

            if not data_by_tf:
                return None

            # Длины последовательностей как в MultiTimeframeCNN
            seq_len_map = {'5m': 50, '15m': 40, '1h': 30}

            # Собираем последовательности для каждого фрейма в формате [N, seq_len, 5]
            sequences_by_tf: Dict[str, List[np.ndarray]] = {}
            num_samples_per_tf: Dict[str, int] = {}

            for tf, df_tf in data_by_tf.items():
                seq_len = seq_len_map.get(tf, 50)
                arr = df_tf.values.astype(float)
                if len(arr) < seq_len + 1:
                    continue
                seq_list: List[np.ndarray] = []
                for i in range(len(arr) - seq_len):
                    window = arr[i:i + seq_len]            # [seq_len, 5]
                    # TradingCNN ожидает [batch, seq_len, channels]
                    seq_list.append(window)
                if seq_list:
                    sequences_by_tf[tf] = seq_list
                    num_samples_per_tf[tf] = len(seq_list)

            if not sequences_by_tf:
                return None

            # Кол-во общих образцов = минимум по фреймам, если доступны несколько
            common_samples = min(num_samples_per_tf.values()) if len(num_samples_per_tf) > 0 else 0
            if common_samples == 0:
                return None

            # Метки считаем по базовому фрейму 5m, если он есть, иначе берем первый доступный
            base_tf = '5m' if '5m' in data_by_tf else list(data_by_tf.keys())[0]
            base_seq_len = seq_len_map.get(base_tf, 50)
            base_close = data_by_tf[base_tf]['close'].values.astype(float)
            # Метки как в _create_labels (рост/падение на следующий шаг)
            base_labels_full = (np.diff(base_close) > 0).astype(int)
            base_labels_full = np.append(base_labels_full, 0)
            # Выравниваем по окну последовательности
            labels_for_samples = base_labels_full[base_seq_len - 1 : base_seq_len - 1 + common_samples]
            if len(labels_for_samples) < common_samples:
                return None

            # Собираем финальные массивы
            features_multiframe: Dict[str, np.ndarray] = {}
            for tf, seq_list in sequences_by_tf.items():
                features_multiframe[tf] = np.stack(seq_list[:common_samples], axis=0)  # [N, seq_len, 5]

            return {
                'features_multiframe': features_multiframe,
                'labels': labels_for_samples,
                'symbol': symbol
            }

        except Exception as e:
            logger.error(f"❌ Ошибка подготовки данных: {str(e)}")
            return None
    
    def _extract_simple_features(self, data: pd.DataFrame) -> np.ndarray:
        """Простое извлечение признаков для валидации"""
        try:
            # Используем простые технические индикаторы
            features = []
            
            # OHLCV данные
            ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Простые признаки
            for i in range(len(ohlcv)):
                if i < 10:  # Пропускаем первые 10 свечей
                    continue
                    
                # Берем последние 10 свечей для контекста
                window = ohlcv[i-10:i+1]
                
                # Простые признаки
                feature_vector = []
                
                # Текущие значения
                current = window[-1]
                feature_vector.extend([
                    current[0],  # open
                    current[1],  # high
                    current[2],  # low
                    current[3],  # close
                    current[4]   # volume
                ])
                
                # Относительные изменения
                if len(window) > 1:
                    prev = window[-2]
                    feature_vector.extend([
                        (current[3] - prev[3]) / prev[3],  # price change
                        (current[1] - current[2]) / current[3],  # high-low ratio
                        current[4] / np.mean(window[:, 4]) if np.mean(window[:, 4]) > 0 else 0  # volume ratio
                    ])
                else:
                    feature_vector.extend([0, 0, 0])
                
                # Простые скользящие средние
                if len(window) >= 5:
                    sma5 = np.mean(window[-5:, 3])  # 5-period SMA
                    feature_vector.append((current[3] - sma5) / sma5 if sma5 > 0 else 0)
                else:
                    feature_vector.append(0)
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения простых признаков: {str(e)}")
            return np.array([])
    
    def _create_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Создание меток для классификации"""
        try:
            # Простая стратегия: рост/падение цены на следующий период
            prices = data['close'].values
            
            # Вычисляем изменение цены
            price_changes = np.diff(prices)
            
            # Создаем бинарные метки: 1 - рост, 0 - падение
            labels = (price_changes > 0).astype(int)
            
            # Добавляем последний элемент (не можем предсказать)
            labels = np.append(labels, 0)
            
            return labels
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания меток: {str(e)}")
            return np.zeros(len(data))
    
    def _test_model_on_data(self, model: Union[TradingCNN, MultiTimeframeCNN], test_data: Dict) -> Tuple:
        """Тестирование модели: извлекаем признаки и обучаем лёгкий линейный классификатор (linear probe)."""
        try:
            labels_all = test_data['labels']

            # Сбор признаков X
            X_features: List[np.ndarray] = []
            is_multiframe = isinstance(model, MultiTimeframeCNN)

            with torch.no_grad():
                if is_multiframe and 'features_multiframe' in test_data:
                    features_by_tf: Dict[str, np.ndarray] = test_data['features_multiframe']
                    num_samples = min(arr.shape[0] for arr in features_by_tf.values())
                    for i in range(num_samples):
                        try:
                            multi_data = {}
                            for tf, arr in features_by_tf.items():
                                multi_data[tf] = torch.tensor(arr[i], dtype=torch.float32).unsqueeze(0)
                            output = model(multi_data)
                            if isinstance(output, dict):
                                output = output.get('prediction', output.get('output', list(output.values())[0]))
                            X_features.append(output.squeeze(0).cpu().numpy())
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка извлечения признаков образца {i}: {str(e)}")
                else:
                    features = test_data.get('features')
                    if features is None:
                        return [], [], [], np.array([])
                    for i in range(len(features)):
                        try:
                            feature_tensor = torch.FloatTensor(features[i]).unsqueeze(0)
                            output = model(feature_tensor)
                            if isinstance(output, dict):
                                output = output.get('prediction', output.get('output', list(output.values())[0]))
                            X_features.append(output.squeeze(0).cpu().numpy())
                        except Exception as e:
                            logger.warning(f"⚠️ Ошибка извлечения признаков образца {i}: {str(e)}")

            if not X_features:
                return [], [], [], np.array([])

            X = np.vstack(X_features)
            y = np.asarray(labels_all[: len(X)])

            # Временной split (без утечки будущего)
            if len(X) < 50:
                return [], [], [], np.array([])
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Linear probe
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, n_jobs=1)
            try:
                clf.fit(X_train, y_train)
                proba = clf.predict_proba(X_test)  # [:, 1] если бинарная
                if proba.shape[1] == 2:
                    conf = proba[:, 1]
                    preds = (conf >= 0.5).astype(int)
                    confidences = conf.tolist()
                else:
                    preds = np.argmax(proba, axis=1)
                    confidences = np.max(proba, axis=1).tolist()
            except Exception as e:
                logger.error(f"❌ Ошибка обучения linear probe: {str(e)}")
                return [], [], [], np.array([])

            patterns = ["linear_probe"]
            return preds.tolist(), confidences, list(set(patterns)), y_test
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования модели: {str(e)}")
            return [], [], [], np.array([])
    
    def _calculate_accuracy(self, true_labels: np.ndarray, 
                           predictions: List[int]) -> float:
        """Вычисление точности"""
        try:
            if len(true_labels) != len(predictions):
                min_length = min(len(true_labels), len(predictions))
                true_labels = true_labels[:min_length]
                predictions = predictions[:min_length]
            
            return accuracy_score(true_labels, predictions)
            
        except Exception as e:
            logger.error(f"❌ Ошибка вычисления точности: {str(e)}")
            return 0.0
    
    def _analyze_errors(self, true_labels: np.ndarray, predictions: List[int], 
                       features: np.ndarray = None) -> Dict[str, Any]:
        """Детальный анализ ошибок модели"""
        try:
            if len(true_labels) != len(predictions):
                min_length = min(len(true_labels), len(predictions))
                true_labels = true_labels[:min_length]
                predictions = predictions[:min_length]
            
            # Матрица ошибок
            cm = confusion_matrix(true_labels, predictions)
            
            # Классификационный отчет
            report = classification_report(
                true_labels, predictions, 
                output_dict=True, zero_division=0
            )
            
            # Детальный анализ ошибок
            error_analysis = {
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'error_rate': 1 - accuracy_score(true_labels, predictions),
                'error_patterns': self._identify_error_patterns(true_labels, predictions),
                'class_balance': self._analyze_class_balance(true_labels),
                'prediction_confidence': self._analyze_prediction_confidence(predictions, true_labels)
            }
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа ошибок: {str(e)}")
            return {}
    
    def _identify_error_patterns(self, true_labels: np.ndarray, predictions: List[int]) -> Dict[str, Any]:
        """Выявление паттернов ошибок"""
        try:
            errors = []
            correct = []
            
            for i, (true, pred) in enumerate(zip(true_labels, predictions)):
                if true != pred:
                    errors.append({
                        'index': i,
                        'true_label': int(true),
                        'predicted_label': int(pred),
                        'error_type': f"True_{int(true)}_Pred_{int(pred)}"
                    })
                else:
                    correct.append({
                        'index': i,
                        'true_label': int(true),
                        'predicted_label': int(pred)
                    })
            
            # Анализ типов ошибок
            error_types = {}
            for error in errors:
                error_type = error['error_type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                'total_errors': len(errors),
                'total_correct': len(correct),
                'error_types': error_types,
                'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None,
                'error_rate_by_type': {k: v/len(errors) for k, v in error_types.items()} if errors else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка выявления паттернов ошибок: {str(e)}")
            return {}
    
    def _analyze_class_balance(self, labels: np.ndarray) -> Dict[str, Any]:
        """Анализ баланса классов"""
        try:
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            
            balance = {}
            for label, count in zip(unique, counts):
                balance[int(label)] = {
                    'count': int(count),
                    'percentage': float(count / total * 100)
                }
            
            # Проверяем дисбаланс
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            return {
                'class_distribution': balance,
                'imbalance_ratio': imbalance_ratio,
                'is_balanced': imbalance_ratio < 2.0,
                'dominant_class': int(unique[np.argmax(counts)]),
                'minority_class': int(unique[np.argmin(counts)])
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа баланса классов: {str(e)}")
            return {}
    
    def _analyze_prediction_confidence(self, predictions: List[int], true_labels: np.ndarray) -> Dict[str, Any]:
        """Анализ уверенности предсказаний"""
        try:
            # Простой анализ - считаем, что предсказания с одинаковыми значениями подряд более уверенные
            confidence_patterns = []
            current_pattern = 1
            current_pred = predictions[0]
            
            for i in range(1, len(predictions)):
                if predictions[i] == current_pred:
                    current_pattern += 1
                else:
                    confidence_patterns.append({
                        'prediction': int(current_pred),
                        'length': current_pattern,
                        'confidence_level': 'high' if current_pattern >= 3 else 'medium' if current_pattern >= 2 else 'low'
                    })
                    current_pred = predictions[i]
                    current_pattern = 1
            
            # Добавляем последний паттерн
            confidence_patterns.append({
                'prediction': int(current_pred),
                'length': current_pattern,
                'confidence_level': 'high' if current_pattern >= 3 else 'medium' if current_pattern >= 2 else 'low'
            })
            
            # Статистика по уверенности
            high_conf = sum(1 for p in confidence_patterns if p['confidence_level'] == 'high')
            medium_conf = sum(1 for p in confidence_patterns if p['confidence_level'] == 'medium')
            low_conf = sum(1 for p in confidence_patterns if p['confidence_level'] == 'low')
            
            return {
                'confidence_patterns': confidence_patterns,
                'high_confidence_count': high_conf,
                'medium_confidence_count': medium_conf,
                'low_confidence_count': low_conf,
                'avg_pattern_length': np.mean([p['length'] for p in confidence_patterns]),
                'confidence_distribution': {
                    'high': high_conf / len(confidence_patterns) * 100,
                    'medium': medium_conf / len(confidence_patterns) * 100,
                    'low': low_conf / len(confidence_patterns) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа уверенности: {str(e)}")
            return {}
    
    def _generate_recommendation(self, symbol_results: List[Dict], 
                               overall_accuracy: float) -> str:
        """Генерация рекомендаций на основе результатов"""
        try:
            if overall_accuracy >= 0.7:
                return "✅ Отличная обобщающая способность! Модель готова к использованию в торговле."
            elif overall_accuracy >= 0.6:
                return "⚠️ Хорошая обобщающая способность, но рекомендуется дополнительное тестирование."
            elif overall_accuracy >= 0.5:
                return "❌ Слабая обобщающая способность. Рекомендуется дообучение на новых данных."
            else:
                return "❌ Плохая обобщающая способность. Модель требует переобучения."
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации рекомендаций: {str(e)}")
            return "Не удалось сгенерировать рекомендации."


def validate_cnn_model(model_path: str, test_symbols: List[str] = None, 
                      test_period: str = 'last_year') -> Dict[str, Any]:
    """
    Удобная функция для валидации CNN модели
    
    Args:
        model_path: Путь к модели
        test_symbols: Символы для тестирования
        test_period: Период тестирования
        
    Returns:
        Результаты валидации
    """
    validator = CNNModelValidator()
    return validator.validate_model(model_path, test_symbols, test_period)
