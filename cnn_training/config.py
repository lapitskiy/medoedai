"""
Конфигурация для обучения CNN моделей
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
from datetime import datetime


@dataclass
class CNNTrainingConfig:
    """Конфигурация для обучения CNN моделей"""
    
    # === Основные параметры ===
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"])  # Больше символов для увеличения данных
    timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    sequence_length: int = 50  # Длина последовательности для CNN (базовое значение для 5-минутного фрейма)
    
    # === Данные ===
    train_start_date: str = "2023-01-01"
    train_end_date: str = "2024-06-01"
    test_start_date: str = "2024-06-01"
    test_end_date: str = "2024-12-01"
    
    # === Архитектура CNN ===
    input_channels: int = 5  # OHLCV
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    output_features: int = 64  # Размер латентного вектора для DQN
    # Схема меток: 'binary' (падение/рост) или 'ternary' (падение/боковое/рост)
    label_scheme: str = "binary"
    num_classes: int = 2  # 2 для binary, 3 для ternary
    dropout_rate: float = 0.4  # Усиленная регуляризация
    use_batch_norm: bool = True
    
    # === Обучение ===
    batch_size: int = 16  # Еще больше уменьшили для лучшего обобщения
    learning_rate: float = 0.00005  # Меньший learning rate
    num_epochs: int = 300
    early_stopping_patience: int = 100  # Еще больше терпения
    weight_decay: float = 5e-3  # Сильная регуляризация
    label_smoothing: float = 0.05  # Снижаем излишнюю уверенность
    class_balance: str = "auto"  # auto | none (веса классов и/или WeightedRandomSampler)
    
    # === Оптимизатор ===
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # === Предсказание ===
    prediction_horizon: int = 1  # Горизонт предсказания (для binary ближе к валидации)
    prediction_threshold: float = 0.01  # Порог для ternary; для binary игнорируется
    label_noise: float = 0.0  # Доля случайного шума в метках (0 — отключено)
    
    # === Данные и пути ===
    data_dir: str = "temp/binance_data"
    model_save_dir: str = "cnn_training/result"  # Аналогично result/ для DQN
    
    # === Даты для обучения ===
    train_start_date: str = "2022-01-01"  # Начинаем с 2022 года для большего объема данных
    test_end_date: str = "2024-12-01"     # До конца 2024 года
    
    # === Устройство ===
    device: str = "auto"  # auto, cuda, cpu
    
    # === Логирование ===
    use_wandb: bool = False
    wandb_project: str = "crypto-cnn-training"
    log_interval: int = 100
    
    # === Валидация ===
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        """Проверка и настройка конфигурации после инициализации"""
        # Автоматическое определение устройства
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        
        # Согласуем количество классов с выбранной схемой меток
        if self.label_scheme not in ("binary", "ternary"):
            self.label_scheme = "binary"
        self.num_classes = 2 if self.label_scheme == "binary" else 3

        # Создание директорий
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь"""
        return {
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'sequence_length': self.sequence_length,
            'input_channels': self.input_channels,
            'hidden_channels': self.hidden_channels,
            'output_features': self.output_features,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'prediction_horizon': self.prediction_horizon,
            'prediction_threshold': self.prediction_threshold,
            'validation_split': self.validation_split,
            'cross_validation_folds': self.cross_validation_folds,
            'device': self.device,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CNNTrainingConfig":
        """Создание конфигурации из словаря"""
        return cls(**config_dict)
    
    def get_model_path(self, symbol: str, timeframe: str, run_id: str = None) -> str:
        """Получение пути для сохранения модели в структуре как DQN"""
        if run_id is None:
            import time
            run_id = hex(int(time.time()))[-4:]  # Генерируем короткий ID как в DQN
        
        # Создаем структуру: cnn_training/result/SYMBOL/runs/RUN_ID/
        symbol_dir = os.path.join(self.model_save_dir, symbol)
        runs_dir = os.path.join(symbol_dir, "runs")
        run_dir = os.path.join(runs_dir, run_id)
        
        os.makedirs(run_dir, exist_ok=True)
        
        filename = f"cnn_model_{timeframe}.pth"
        return os.path.join(run_dir, filename)
    
    def get_run_dir(self, symbol: str, run_id: str = None) -> str:
        """Получение директории для run"""
        if run_id is None:
            import time
            run_id = hex(int(time.time()))[-4:]
        
        symbol_dir = os.path.join(self.model_save_dir, symbol)
        runs_dir = os.path.join(symbol_dir, "runs")
        run_dir = os.path.join(runs_dir, run_id)
        
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def get_log_path(self, symbol: str, timeframe: str, run_id: str = None) -> str:
        """Получение пути для логирования"""
        run_dir = self.get_run_dir(symbol, run_id)
        filename = f"training_{timeframe}.log"
        return os.path.join(run_dir, filename)
    
    def save_manifest(self, symbol: str, run_id: str, model_type: str, 
                      timeframes: list, config_dict: dict, symbols: Optional[List[str]] = None) -> str:
        """Сохранение manifest.json как в DQN"""
        run_dir = self.get_run_dir(symbol, run_id)
        manifest_path = os.path.join(run_dir, "manifest.json")
        
        manifest = {
            "symbol": symbol,
            "run_id": run_id,
            "model_type": model_type,
            "timeframes": timeframes,
            "created_at": datetime.now().isoformat(),
            "config": config_dict,
            "architecture": "MultiTimeframeCNN",
            "version": "1.0"
        }
        if symbols is not None:
            manifest["symbols"] = symbols
        
        import json
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return manifest_path

    def save_results(self, symbol: str, run_id: str, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Сохранение результатов обучения в JSON в папке ран-а, как в DQN"""
        run_dir = self.get_run_dir(symbol, run_id)
        if not filename:
            filename = 'result_multiframe.json' if symbol == 'multi' else 'result_single.json'
        path = os.path.join(run_dir, filename)
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return path