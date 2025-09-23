"""
Загрузчик данных для обучения CNN моделей
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Any
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class CryptoDataset(Dataset):
    """Датасет для обучения CNN на данных криптовалют"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, sequence_length: int = 50):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        
        assert len(data) == len(labels), "Данные и метки должны иметь одинаковую длину"
        assert len(data) >= sequence_length, f"Недостаточно данных для sequence_length={sequence_length}"
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Получаем последовательность данных
        sequence = self.data[idx:idx + self.sequence_length]
        
        # Получаем соответствующую метку
        label = self.labels[idx + self.sequence_length - 1]
        
        # Конвертируем в тензоры
        sequence_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.LongTensor([label])
        
        return sequence_tensor, label_tensor


class MultiTimeframeDataset(Dataset):
    """Датасет для мультифреймовой CNN"""
    
    def __init__(self, data_dict: Dict[str, np.ndarray], labels: np.ndarray, 
                 sequence_lengths: Dict[str, int]):
        self.data_dict = data_dict
        self.labels = labels
        self.sequence_lengths = sequence_lengths
        self.timeframes = list(data_dict.keys())
        
        # Проверяем совместимость данных
        min_length = min(len(data) for data in data_dict.values())
        # Используем метки от самого короткого фрейма, но не требуем точного соответствия
        if len(labels) > min_length:
            labels = labels[:min_length]  # Обрезаем метки до минимальной длины
        
        # Определяем максимальную длину последовательности
        self.max_sequence_length = max(sequence_lengths.values())
        
        # Проверяем, что у нас достаточно данных
        for timeframe, seq_len in sequence_lengths.items():
            assert len(data_dict[timeframe]) >= seq_len, f"Недостаточно данных для {timeframe}"
    
    def __len__(self) -> int:
        # Используем длину меток как основу для количества образцов
        return len(self.labels) - self.max_sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Получаем последовательности для каждого временного фрейма
        sequences = {}
        
        for timeframe in self.timeframes:
            seq_len = self.sequence_lengths[timeframe]
            data = self.data_dict[timeframe]
            
            # Убеждаемся, что у нас достаточно данных
            if idx + seq_len <= len(data):
                sequence = data[idx:idx + seq_len]
            else:
                # Если данных недостаточно, берем последние доступные
                sequence = data[-seq_len:]
            
            sequences[timeframe] = torch.FloatTensor(sequence)
        
        # Получаем соответствующую метку
        if idx + self.max_sequence_length - 1 < len(self.labels):
            label = self.labels[idx + self.max_sequence_length - 1]
        else:
            # Если метка выходит за границы, берем последнюю
            label = self.labels[-1]
        
        label_tensor = torch.LongTensor([label])
        
        return sequences, label_tensor


class CryptoDataLoader:
    """Основной класс для загрузки и подготовки данных криптовалют"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_means = {}
        self.feature_stds = {}
    
    def load_symbol_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Загрузка данных для конкретного символа и временного фрейма"""
        data_dir = self.config.data_dir
        
        # Возможные пути к данным
        possible_paths = [
            os.path.join(data_dir, f"{symbol.lower()}_{timeframe}.csv"),
            os.path.join(data_dir, symbol, timeframe, "data.csv"),
            os.path.join(data_dir, "spot", symbol, f"{timeframe}.csv"),
            os.path.join(data_dir, symbol, f"{timeframe}.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"✅ Загружены данные {symbol} {timeframe}: {len(df)} записей")
                    return df
                except Exception as e:
                    print(f"❌ Ошибка загрузки {path}: {e}")
                    continue

        # Если локальные файлы не найдены — пробуем загрузить как в DQN через БД/биржу
        try:
            from utils.db_utils import db_get_or_fetch_ohlcv
            candles_limit = self._estimate_required_candles(timeframe)
            print(f"🔄 Пытаемся загрузить из БД/биржи {symbol} {timeframe} ~ {candles_limit} свечей...")
            df = db_get_or_fetch_ohlcv(
                symbol_name=symbol,
                timeframe=timeframe,
                limit_candles=candles_limit,
                dry_run=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"✅ Данные получены из БД/биржи: {symbol} {timeframe}: {len(df)} записей")
                return df
            else:
                print(f"⚠️ Не удалось получить данные из БД/биржи: пустой DataFrame для {symbol} {timeframe}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке из БД/биржи для {symbol} {timeframe}: {e}")

        print(f"⚠️ Данные для {symbol} {timeframe} не найдены")
        return None

    def _estimate_required_candles(self, timeframe: str) -> int:
        """Оценка необходимого количества свечей по датам из конфигурации"""
        try:
            from datetime import datetime
            start = datetime.fromisoformat(getattr(self.config, 'train_start_date', '2023-01-01'))
            end = datetime.fromisoformat(getattr(self.config, 'test_end_date', '2024-12-01'))
            total_minutes = max(1, int((end - start).total_seconds() // 60))
            minutes_per_candle = {
                '1m': 1,
                '3m': 3,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440,
            }.get(timeframe, 5)
            candles = total_minutes // minutes_per_candle
            # Ограничим разумным верхним пределом
            return max(1000, min(candles, 200_000))
        except Exception:
            # Бэкап значение
            return 50_000
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Предобработка DataFrame"""
        # Удаляем строки с NaN
        df = df.dropna()
        
        # Проверяем наличие необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Если есть timestamp колонка, убираем ее
        if 'timestamp' in df.columns or 'open_time' in df.columns:
            timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
            df = df.drop(columns=timestamp_cols)
        
        # Проверяем, что у нас есть нужные колонки
        available_columns = df.columns.tolist()
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            print(f"⚠️ Отсутствуют колонки: {missing_columns}")
            # Пытаемся найти альтернативные названия
            column_mapping = {}
            for col in required_columns:
                if col not in available_columns:
                    # Ищем похожие названия
                    for available_col in available_columns:
                        if col in available_col.lower() or available_col.lower() in col:
                            column_mapping[available_col] = col
                            break
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"✅ Переименованы колонки: {column_mapping}")
            else:
                raise ValueError(f"Не удалось найти необходимые колонки: {missing_columns}")
        
        # Оставляем только нужные колонки в правильном порядке
        df = df[required_columns]
        
        # Конвертируем в float
        df = df.astype(float)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.01) -> np.ndarray:
        """Создание меток для предсказания движения цены.
        Поддержка binary (0/1) и ternary (0/1/2) схем через config.label_scheme."""
        close_prices = df['close'].values
        
        # Вычисляем будущие цены через horizon шагов
        future_prices = np.roll(close_prices, -horizon)
        future_prices[-horizon:] = close_prices[-horizon:]
        
        # Процентное изменение
        price_changes = (future_prices - close_prices) / np.clip(close_prices, 1e-8, None)
        
        if getattr(self.config, 'label_scheme', 'binary') == 'binary':
            # Бинарные метки: 1 — рост, 0 — падение (как в валидаторе)
            labels = (price_changes > 0).astype(int)
        else:
            # Тернарные: 0 — падение, 1 — боковое, 2 — рост
            labels = np.ones(len(price_changes), dtype=int)
            labels[price_changes < -threshold] = 0
            labels[price_changes > threshold] = 2
            
            # По желанию: небольшой шум
            label_noise = getattr(self.config, 'label_noise', 0.0)
            if label_noise and label_noise > 0:
                noise_mask = np.random.random(len(labels)) < float(label_noise)
                labels[noise_mask] = np.random.randint(0, 3, size=int(np.sum(noise_mask)))
        
        return labels
    
    def normalize_data(self, data: np.ndarray, symbol: str, timeframe: str, 
                      fit_scaler: bool = True) -> np.ndarray:
        """Нормализация данных"""
        scaler_key = f"{symbol}_{timeframe}"
        
        if fit_scaler:
            # Используем RobustScaler для устойчивости к выбросам
            scaler = RobustScaler()
            normalized_data = scaler.fit_transform(data)
            self.scalers[scaler_key] = scaler
            
            # Сохраняем статистики
            self.feature_means[scaler_key] = np.mean(data, axis=0)
            self.feature_stds[scaler_key] = np.std(data, axis=0)
        else:
            # Используем уже обученный scaler
            if scaler_key in self.scalers:
                scaler = self.scalers[scaler_key]
                normalized_data = scaler.transform(data)
            else:
                print(f"⚠️ Scaler для {scaler_key} не найден, используем StandardScaler")
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(data)
                self.scalers[scaler_key] = scaler
        
        return normalized_data
    
    def prepare_training_data(self, symbols: List[str], timeframes: List[str]) -> Dict[str, Any]:
        """Подготовка данных для обучения"""
        all_data = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Загружаем данные
                df = self.load_symbol_data(symbol, timeframe)
                if df is None:
                    continue
                
                # Предобрабатываем
                df = self.preprocess_dataframe(df)
                
                # Создаем метки
                labels = self.create_labels(df, self.config.prediction_horizon, 
                                          self.config.prediction_threshold)
                
                # Нормализуем данные
                data_array = df.values
                normalized_data = self.normalize_data(data_array, symbol, timeframe, fit_scaler=True)
                
                # Сохраняем данные
                key = f"{symbol}_{timeframe}"
                all_data[key] = {
                    'data': normalized_data,
                    'labels': labels,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'scaler': self.scalers[key]
                }
                
                print(f"📊 {key}: {len(normalized_data)} образцов, "
                      f"метки: {np.bincount(labels)}")
        
        return all_data
    
    def create_datasets(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Создание датасетов для обучения"""
        datasets = {}
        
        for key, data_info in data_dict.items():
            symbol = data_info['symbol']
            timeframe = data_info['timeframe']
            data = data_info['data']
            labels = data_info['labels']
            
            # Определяем длину последовательности для данного фрейма
            if timeframe == "5m":
                seq_len = self.config.sequence_length
            elif timeframe == "15m":
                seq_len = self.config.sequence_length // 3
            elif timeframe == "1h":
                seq_len = self.config.sequence_length // 12
            else:
                seq_len = self.config.sequence_length // 2
            
            # Создаем датасет
            dataset = CryptoDataset(data, labels, seq_len)
            datasets[key] = dataset
            
            print(f"📦 Создан датасет {key}: {len(dataset)} образцов")
        
        return datasets
    
    def create_multiframe_dataset(self, data_dict: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
        """Создание мультифреймового датасета"""
        # Группируем данные по символам
        symbols_data = {}
        
        for key, data_info in data_dict.items():
            symbol = data_info['symbol']
            timeframe = data_info['timeframe']
            data = data_info['data']
            labels = data_info['labels']
            
            if symbol not in symbols_data:
                symbols_data[symbol] = {
                    'data': {},
                    'labels_per_tf': {}
                }

            symbols_data[symbol]['data'][timeframe] = data
            symbols_data[symbol]['labels_per_tf'][timeframe] = labels
        
        # Создаем мультифреймовые датасеты
        train_datasets = []
        val_datasets = []
        
        for symbol, symbol_data in symbols_data.items():
            data_dict_multi = symbol_data['data']
            labels_per_tf = symbol_data['labels_per_tf']
            
            # Определяем длины последовательностей для каждого фрейма
            sequence_lengths = {}
            for timeframe in data_dict_multi.keys():
                if timeframe == "5m":
                    seq_len = self.config.sequence_length
                elif timeframe == "15m":
                    seq_len = self.config.sequence_length // 3
                elif timeframe == "1h":
                    seq_len = self.config.sequence_length // 12
                else:
                    seq_len = self.config.sequence_length // 2
                
                sequence_lengths[timeframe] = seq_len
            
            # Выбираем опорные метки: используем самый длинный фрейм (5m) для максимального количества образцов
            if '5m' in labels_per_tf:
                base_labels = labels_per_tf['5m']
                print(f"📊 Используем метки от 5m фрейма: {len(base_labels)} образцов")
            else:
                # выбираем фрейм с максимальной длиной данных
                tf_max = max(data_dict_multi.keys(), key=lambda tf: len(data_dict_multi[tf]))
                base_labels = labels_per_tf[tf_max]
                print(f"📊 Используем метки от {tf_max} фрейма: {len(base_labels)} образцов")

            # НЕ обрезаем данные - используем все доступные данные каждого фрейма
            # MultiTimeframeDataset сам будет брать нужные срезы для каждого образца
            
            # Создаем мультифреймовый датасет со всеми данными
            dataset = MultiTimeframeDataset(data_dict_multi, base_labels, sequence_lengths)
            
            print(f"📊 Датасет {symbol}: {len(dataset)} образцов")
            print(f"📊 Размеры данных по фреймам: {[(tf, len(data_dict_multi[tf])) for tf in data_dict_multi.keys()]}")
            print(f"📊 Размер меток: {len(base_labels)}")
            print(f"📊 Распределение меток: {np.bincount(base_labels)}")
            
            # Разделяем на train/val
            train_size = int(len(dataset) * (1 - self.config.validation_split))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            print(f"📊 Train: {train_size} образцов, Val: {val_size} образцов")
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        
        # Объединяем все датасеты
        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        else:
            train_dataset = train_datasets[0]
            val_dataset = val_datasets[0]
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
        """Создание DataLoader'ов"""
        # Балансировка классов через WeightedRandomSampler при необходимости
        sampler = None
        if getattr(self.config, 'class_balance', 'auto') == 'auto':
            try:
                # Оценим распределение классов
                targets = []
                if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'labels'):
                    # Subset из random_split
                    labels_full = train_dataset.dataset.labels
                    indices = train_dataset.indices if hasattr(train_dataset, 'indices') else range(len(train_dataset))
                    max_seq = getattr(train_dataset.dataset, 'sequence_length', 1)
                    for idx in indices:
                        label_idx = min(idx + max_seq - 1, len(labels_full) - 1)
                        targets.append(int(labels_full[label_idx]))
                elif hasattr(train_dataset, 'labels'):
                    labels_full = train_dataset.labels
                    max_seq = getattr(train_dataset, 'max_sequence_length', 1)
                    for idx in range(len(train_dataset)):
                        label_idx = min(idx + max_seq - 1, len(labels_full) - 1)
                        targets.append(int(labels_full[label_idx]))
                
                if targets:
                    counts = np.bincount(np.array(targets), minlength=self.config.num_classes)
                    class_weights = 1.0 / np.clip(counts, 1, None)
                    sample_weights = [class_weights[t] for t in targets]
                    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            except Exception:
                sampler = None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader
