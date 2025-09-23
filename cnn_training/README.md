# CNN Training Module

Модуль для обучения CNN моделей на данных криптовалют и извлечения признаков для DQN агента.

## Структура модуля

```
cnn_training/
├── __init__.py              # Инициализация модуля
├── config.py                # Конфигурация обучения
├── models.py                # CNN архитектуры
├── data_loader.py           # Загрузка и подготовка данных
├── trainer.py               # Тренер для обучения моделей
├── feature_extractor.py     # Извлечение признаков для DQN
├── train_cnn.py            # Скрипт для обучения
└── README.md               # Документация
```

## Возможности

### 🤖 CNN Модели
- **TradingCNN**: CNN для анализа одного временного фрейма
- **MultiTimeframeCNN**: CNN для анализа нескольких временных фреймов
- **PricePredictionCNN**: CNN для предсказания движения цены
- **CNNFeatureExtractor**: Замороженная CNN для извлечения признаков

### 📊 Подготовка данных
- Автоматическая загрузка данных из различных форматов
- Нормализация OHLCV данных
- Создание меток для предсказания движения цены
- Поддержка различных временных фреймов

### 🎯 Обучение
- Обучение отдельных моделей для каждого символа/фрейма
- Обучение мультифреймовых моделей
- Early stopping и валидация
- Поддержка различных оптимизаторов и планировщиков
- Интеграция с Weights & Biases

### 🔧 Извлечение признаков
- Извлечение латентных признаков из предобученных CNN
- Интеграция с DQN средой
- Поддержка мультифреймовых признаков

## Быстрый старт

### 1. Обучение CNN модели

```bash
# Обучение модели предсказания для BTCUSDT
python cnn_training/train_cnn.py --symbol BTCUSDT --timeframe 5m --model_type prediction

# Обучение мультифреймовой модели
python cnn_training/train_cnn.py --symbols BTCUSDT,ETHUSDT --timeframes 5m,15m,1h --model_type multiframe

# Обучение с пользовательскими параметрами
python cnn_training/train_cnn.py \
    --symbols BTCUSDT,ETHUSDT,TONUSDT \
    --timeframes 5m,15m,1h \
    --model_type prediction \
    --batch_size 128 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --sequence_length 100
```

### 2. Использование в коде

```python
from cnn_training.config import CNNTrainingConfig
from cnn_training.trainer import CNNTrainer
from cnn_training.feature_extractor import create_cnn_wrapper

# Создание конфигурации
config = CNNTrainingConfig(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["5m", "15m", "1h"],
    sequence_length=50,
    output_features=64
)

# Обучение модели
trainer = CNNTrainer(config)
result = trainer.train_single_model("BTCUSDT", "5m", "prediction")

# Извлечение признаков для DQN
cnn_wrapper = create_cnn_wrapper(config)
features = cnn_wrapper.get_cnn_features("BTCUSDT", ohlcv_data)
```

## Конфигурация

### Основные параметры

```python
@dataclass
class CNNTrainingConfig:
    # Данные
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "TONUSDT"]
    timeframes: List[str] = ["5m", "15m", "1h"]
    sequence_length: int = 50
    
    # Архитектура
    input_channels: int = 5  # OHLCV
    hidden_channels: List[int] = [32, 64, 128]
    output_features: int = 64
    
    # Обучение
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    
    # Предсказание
    prediction_horizon: int = 5
    prediction_threshold: float = 0.01
```

### Параметры командной строки

```bash
--symbols              # Символы криптовалют (через запятую)
--timeframes           # Временные фреймы (через запятую)
--model_type           # Тип модели: single, multiframe, prediction
--batch_size           # Размер батча
--learning_rate        # Скорость обучения
--num_epochs           # Количество эпох
--sequence_length      # Длина последовательности
--prediction_horizon   # Горизонт предсказания
--prediction_threshold # Порог значимого движения
--device               # Устройство: auto, cuda, cpu
--use_wandb           # Использовать Weights & Biases
```

## Архитектура моделей

### TradingCNN
```
Input: [batch, sequence_length, 5] (OHLCV)
├── Conv1D Block 1 (5→32, kernel=3)
├── Conv1D Block 2 (32→64, kernel=5)
├── Conv1D Block 3 (64→128, kernel=7)
├── Global Average Pooling
└── Feature Extractor (128→64)
Output: [batch, 64] (латентные признаки)
```

### MultiTimeframeCNN
```
Input: {"5m": [batch, 50, 5], "15m": [batch, 30, 5], "1h": [batch, 20, 5]}
├── TradingCNN для 5m → [batch, 21]
├── TradingCNN для 15m → [batch, 21]
├── TradingCNN для 1h → [batch, 21]
├── Concatenate → [batch, 63]
└── Fusion Layer → [batch, 64]
Output: [batch, 64] (объединенные признаки)
```

## Интеграция с DQN

### 1. Обучение CNN
```python
# Обучаем CNN на исторических данных
trainer = CNNTrainer(config)
trainer.train_single_model("BTCUSDT", "5m", "prediction")
```

### 2. Извлечение признаков
```python
# Создаем обертку для извлечения признаков
cnn_wrapper = create_cnn_wrapper(config)

# В DQN среде получаем CNN признаки
cnn_features = cnn_wrapper.get_cnn_features(symbol, ohlcv_data)

# Объединяем с обычными признаками состояния
combined_state = np.concatenate([base_state, cnn_features])
```

### 3. Обновление DQN состояния
```python
# В crypto_trading_env.py
def _get_state_with_cnn(self):
    # Обычное состояние
    base_state = self._get_base_state()
    
    # CNN признаки
    ohlcv_data = {
        "5m": self.cnn_data_5min[start:end],
        "15m": self.cnn_data_15min[start:end],
        "1h": self.cnn_data_1h[start:end]
    }
    cnn_features = self.cnn_wrapper.get_cnn_features(self.symbol, ohlcv_data)
    
    # Объединенное состояние
    return np.concatenate([base_state, cnn_features])
```

## Структура данных

### Входные данные
- **OHLCV**: Open, High, Low, Close, Volume
- **Форма**: [sequence_length, 5]
- **Нормализация**: RobustScaler для устойчивости к выбросам

### Метки
- **0**: Падение цены > threshold
- **1**: Боковое движение (в пределах threshold)
- **2**: Рост цены > threshold

### Выходные признаки
- **Размер**: config.output_features (по умолчанию 64)
- **Нормализация**: Tanh активация
- **Использование**: Замороженные признаки для DQN

## Мониторинг обучения

### Логирование
- Автоматическое логирование в файлы
- Метрики: loss, accuracy, learning rate
- Сохранение лучших моделей

### Weights & Biases
```bash
# Включить Wandb логирование
python cnn_training/train_cnn.py --use_wandb --wandb_project "crypto-cnn"
```

### Метрики
- Train/Validation Loss
- Train/Validation Accuracy
- Learning Rate Schedule
- Model Weights Distribution

## Требования

### Зависимости
```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
wandb>=0.13.0 (опционально)
```

### Данные
- CSV файлы с OHLCV данными
- Поддерживаемые форматы: timestamp, open, high, low, close, volume
- Минимальная длина: sequence_length + prediction_horizon

## Примеры использования

### Обучение модели для одного символа
```python
config = CNNTrainingConfig(
    symbols=["BTCUSDT"],
    timeframes=["5m"],
    sequence_length=50,
    output_features=64
)

trainer = CNNTrainer(config)
result = trainer.train_single_model("BTCUSDT", "5m", "prediction")
```

### Обучение мультифреймовой модели
```python
config = CNNTrainingConfig(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["5m", "15m", "1h"],
    sequence_length=50
)

trainer = CNNTrainer(config)
result = trainer.train_multiframe_model(["BTCUSDT", "ETHUSDT"])
```

### Извлечение признаков для DQN
```python
cnn_wrapper = create_cnn_wrapper(config)

# В DQN среде
ohlcv_data = {
    "5m": self.get_ohlcv_sequence("5m"),
    "15m": self.get_ohlcv_sequence("15m"),
    "1h": self.get_ohlcv_sequence("1h")
}

cnn_features = cnn_wrapper.get_cnn_features("BTCUSDT", ohlcv_data)
```

## Troubleshooting

### Частые проблемы

1. **Недостаточно данных**
   - Увеличить sequence_length
   - Использовать больше исторических данных

2. **Низкая точность**
   - Увеличить hidden_channels
   - Настроить prediction_threshold
   - Использовать data augmentation

3. **Медленное обучение**
   - Увеличить batch_size
   - Использовать GPU
   - Уменьшить sequence_length

4. **Ошибки загрузки данных**
   - Проверить пути к файлам
   - Проверить формат CSV
   - Убедиться в наличии всех колонок OHLCV

### Логи и отладка

```python
# Включить подробное логирование
import logging
logging.basicConfig(level=logging.DEBUG)

# Проверить доступность данных
from cnn_training.data_loader import CryptoDataLoader
loader = CryptoDataLoader(config)
df = loader.load_symbol_data("BTCUSDT", "5m")
print(f"Загружено {len(df)} записей")
```

## Лицензия

Этот модуль является частью проекта medoedai и следует той же лицензии.
