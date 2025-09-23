"""
CNN Training Module для анализа торговых паттернов криптовалют

Этот модуль содержит современную архитектуру CNN для:
- Обучения CNN на исторических данных криптовалют
- Извлечения латентных признаков для DQN агента
- Предсказания движения цен на основе OHLCV паттернов
"""

from .models import TradingCNN, MultiTimeframeCNN, CNNFeatureExtractor
from .trainer import CNNTrainer
from .data_loader import CryptoDataLoader
from .config import CNNTrainingConfig
from .model_validator import CNNModelValidator, validate_cnn_model
from .validation_analyzer import ValidationAnalyzer, analyze_validation_results

__version__ = "1.1.0"
__all__ = [
    "TradingCNN",
    "MultiTimeframeCNN", 
    "CNNFeatureExtractor",
    "CNNTrainer",
    "CryptoDataLoader",
    "CNNTrainingConfig",
    "CNNModelValidator",
    "validate_cnn_model",
    "ValidationAnalyzer",
    "analyze_validation_results"
]
