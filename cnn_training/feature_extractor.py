"""
Извлечение признаков из предобученных CNN моделей для использования в DQN
"""

import torch
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
import logging

from .models import CNNFeatureExtractor, TradingCNN, MultiTimeframeCNN
from .config import CNNTrainingConfig


class CNNFeatureExtractorWrapper:
    """Обертка для извлечения CNN признаков в DQN среде"""
    
    def __init__(self, config: CNNTrainingConfig, model_paths: Dict[str, str] = None):
        self.config = config
        self.device = torch.device(config.device)
        self.model_paths = model_paths or {}
        self.extractors = {}
        self.logger = logging.getLogger(__name__)
        
        # Загружаем предобученные модели
        self._load_models()
    
    def _load_models(self):
        """Загрузка предобученных CNN моделей"""
        model_dir = self.config.model_save_dir
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                # Определяем путь к модели
                if f"{symbol}_{timeframe}" in self.model_paths:
                    model_path = self.model_paths[f"{symbol}_{timeframe}"]
                else:
                    model_path = self.config.get_model_path(symbol, timeframe)
                
                # Проверяем существование файла
                if not os.path.exists(model_path):
                    self.logger.warning(f"⚠️ Модель не найдена: {model_path}")
                    continue
                
                try:
                    # Загружаем checkpoint
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Создаем модель
                    if "multiframe" in model_path:
                        # Мультифреймовая модель
                        sequence_lengths = {
                            "5m": self.config.sequence_length,
                            "15m": self.config.sequence_length // 3,
                            "1h": self.config.sequence_length // 12
                        }
                        
                        model = MultiTimeframeCNN(
                            input_channels=self.config.input_channels,
                            hidden_channels=self.config.hidden_channels,
                            output_features=self.config.output_features,
                            sequence_lengths=sequence_lengths,
                            dropout_rate=self.config.dropout_rate,
                            use_batch_norm=self.config.use_batch_norm
                        ).to(self.device)
                    else:
                        # Обычная модель
                        model = TradingCNN(
                            input_channels=self.config.input_channels,
                            hidden_channels=self.config.hidden_channels,
                            output_features=self.config.output_features,
                            sequence_length=self.config.sequence_length,
                            dropout_rate=self.config.dropout_rate,
                            use_batch_norm=self.config.use_batch_norm
                        ).to(self.device)
                    
                    # Загружаем веса
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Замораживаем параметры
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    model.eval()
                    
                    # Сохраняем экстрактор
                    key = f"{symbol}_{timeframe}"
                    self.extractors[key] = model
                    
                    self.logger.info(f"✅ Загружена модель {key}: точность {checkpoint.get('val_accuracy', 0):.2f}%")
                
                except Exception as e:
                    self.logger.error(f"❌ Ошибка загрузки модели {symbol}_{timeframe}: {e}")
    
    def extract_features_single(self, symbol: str, timeframe: str, 
                               data: np.ndarray) -> Optional[np.ndarray]:
        """Извлечение признаков для одного символа и фрейма"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.extractors:
            self.logger.warning(f"⚠️ Экстрактор для {key} не найден")
            return None
        
        try:
            # Подготавливаем данные
            if data.ndim == 2:
                data = data[np.newaxis, :, :]  # Добавляем batch dimension
            
            # Конвертируем в тензор
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            # Извлекаем признаки
            with torch.no_grad():
                features = self.extractors[key](data_tensor)
                features_np = features.cpu().numpy()
            
            return features_np
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка извлечения признаков для {key}: {e}")
            return None
    
    def extract_features_multiframe(self, symbol: str, 
                                   data_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Извлечение признаков для мультифреймовой модели"""
        key = f"{symbol}_multiframe"
        
        # Ищем мультифреймовую модель или создаем из отдельных
        if key in self.extractors:
            model = self.extractors[key]
        else:
            # Создаем мультифреймовую модель из отдельных экстракторов
            return self._extract_features_combined(symbol, data_dict)
        
        try:
            # Подготавливаем данные
            prepared_data = {}
            for timeframe, data in data_dict.items():
                if data.ndim == 2:
                    data = data[np.newaxis, :, :]
                prepared_data[timeframe] = torch.FloatTensor(data).to(self.device)
            
            # Извлекаем признаки
            with torch.no_grad():
                features = model(prepared_data)
                features_np = features.cpu().numpy()
            
            return features_np
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка извлечения мультифреймовых признаков для {symbol}: {e}")
            return None
    
    def _extract_features_combined(self, symbol: str, 
                                  data_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Объединение признаков от отдельных моделей"""
        features_list = []
        
        for timeframe, data in data_dict.items():
            features = self.extract_features_single(symbol, timeframe, data)
            if features is not None:
                features_list.append(features)
        
        if not features_list:
            return None
        
        # Объединяем признаки
        combined_features = np.concatenate(features_list, axis=1)
        
        # Применяем простую нормализацию
        combined_features = np.tanh(combined_features)
        
        return combined_features
    
    def get_feature_size(self, symbol: str, timeframe: str = None) -> int:
        """Получение размера признаков"""
        if timeframe:
            key = f"{symbol}_{timeframe}"
            if key in self.extractors:
                return self.config.output_features
        else:
            # Для мультифреймовых признаков
            return self.config.output_features * len(self.config.timeframes)
        
        return self.config.output_features
    
    def is_available(self, symbol: str, timeframe: str = None) -> bool:
        """Проверка доступности экстрактора"""
        if timeframe:
            key = f"{symbol}_{timeframe}"
            return key in self.extractors
        else:
            # Проверяем доступность для всех фреймов
            for tf in self.config.timeframes:
                key = f"{symbol}_{tf}"
                if key not in self.extractors:
                    return False
            return True


class DQNCNNWrapper:
    """Обертка для интеграции CNN признаков в DQN среду"""
    
    def __init__(self, feature_extractor: CNNFeatureExtractorWrapper):
        self.feature_extractor = feature_extractor
        self.logger = logging.getLogger(__name__)
    
    def get_cnn_features(self, symbol: str, ohlcv_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Получение CNN признаков для DQN состояния"""
        try:
            # Извлекаем мультифреймовые признаки
            features = self.feature_extractor.extract_features_multiframe(symbol, ohlcv_data)
            
            if features is None:
                # Fallback: создаем нулевые признаки
                feature_size = self.feature_extractor.get_feature_size(symbol)
                features = np.zeros((1, feature_size), dtype=np.float32)
                self.logger.warning(f"⚠️ Используем нулевые CNN признаки для {symbol}")
            
            return features.flatten()  # Возвращаем 1D массив
        
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения CNN признаков для {symbol}: {e}")
            # Fallback: нулевые признаки
            feature_size = self.feature_extractor.get_feature_size(symbol)
            return np.zeros(feature_size, dtype=np.float32)
    
    def get_state_with_cnn(self, symbol: str, base_state: np.ndarray, 
                          ohlcv_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Получение объединенного состояния (обычные признаки + CNN признаки)"""
        # Получаем CNN признаки
        cnn_features = self.get_cnn_features(symbol, ohlcv_data)
        
        # Объединяем с обычными признаками
        combined_state = np.concatenate([base_state, cnn_features])
        
        return combined_state.astype(np.float32)


def create_cnn_wrapper(config: CNNTrainingConfig, 
                      model_paths: Dict[str, str] = None) -> DQNCNNWrapper:
    """Фабричная функция для создания CNN обертки"""
    feature_extractor = CNNFeatureExtractorWrapper(config, model_paths)
    return DQNCNNWrapper(feature_extractor)


# Пример использования
if __name__ == "__main__":
    # Создаем конфигурацию
    config = CNNTrainingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframes=["5m", "15m", "1h"],
        output_features=64
    )
    
    # Создаем обертку
    cnn_wrapper = create_cnn_wrapper(config)
    
    # Пример извлечения признаков
    symbol = "BTCUSDT"
    ohlcv_data = {
        "5m": np.random.randn(50, 5),
        "15m": np.random.randn(30, 5),
        "1h": np.random.randn(20, 5)
    }
    
    features = cnn_wrapper.get_cnn_features(symbol, ohlcv_data)
    print(f"CNN признаки для {symbol}: {features.shape}")


class CNNFeatureExtractor:
    """Простой класс для тестирования извлечения признаков"""
    
    def __init__(self, config: CNNTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str):
        """Загрузка предобученной CNN модели"""
        try:
            # Загружаем checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Создаем модель
            from .models import MultiTimeframeCNN
            self.model = MultiTimeframeCNN(
                input_channels=self.config.input_channels,
                hidden_channels=self.config.hidden_channels,
                output_features=self.config.output_features,
                dropout_rate=self.config.dropout_rate,
                use_batch_norm=self.config.use_batch_norm,
                sequence_lengths={"5m": 50, "15m": 40, "1h": 30}  # Фиксированные длины
            )
            
            # Загружаем веса
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Режим инференса
            
            self.logger.info(f"✅ Модель загружена: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели {model_path}: {e}")
            raise
    
    def extract_features(self, sample):
        """Извлечение признаков из образца данных"""
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")
        
        try:
            with torch.no_grad():
                # Если sample - это кортеж (data, labels), извлекаем только данные
                if isinstance(sample, tuple):
                    data = sample[0]  # Берем только данные, игнорируем labels
                else:
                    data = sample
                
                # Подготавливаем данные
                if isinstance(data, dict):
                    # Мультифреймовые данные
                    inputs = {}
                    for tf in ["5m", "15m", "1h"]:
                        if tf in data:
                            inputs[tf] = data[tf].unsqueeze(0).to(self.device)
                    
                    # Извлекаем признаки
                    features = self.model(inputs)
                    
                else:
                    # Одиночные данные
                    inputs = data.unsqueeze(0).to(self.device)
                    features = self.model(inputs)
                
                # Конвертируем в numpy
                features = features.cpu().numpy().flatten()
                
                return features
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка извлечения признаков: {e}")
            raise
