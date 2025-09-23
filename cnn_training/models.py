"""
CNN модели для анализа торговых паттернов криптовалют
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


class Conv1DBlock(nn.Module):
    """Блок Conv1D с BatchNorm и Dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TradingCNN(nn.Module):
    """CNN для анализа торговых паттернов на одном временном фрейме"""
    
    def __init__(self, input_channels: int = 5, hidden_channels: List[int] = None,
                 output_features: int = 64, sequence_length: int = 50,
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.output_features = output_features
        
        # Сверточные блоки
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_channels):
            kernel_size = 3 + i * 2  # Увеличиваем размер ядра для более глубоких слоев
            padding = kernel_size // 2
            
            block = Conv1DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        # Глобальное усреднение
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Классификатор для извлечения признаков
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels[-1] // 2, output_features)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, sequence_length, input_channels] -> [batch_size, features, sequence_length]
        """
        # Транспонируем для Conv1d: [batch, seq_len, channels] -> [batch, channels, seq_len]
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # Проходим через сверточные блоки
        for block in self.conv_blocks:
            x = block(x)
        
        # Глобальное усреднение
        x = self.global_avg_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # [batch, channels]
        
        # Извлечение признаков
        features = self.feature_extractor(x)  # [batch, output_features]
        
        return features


class MultiTimeframeCNN(nn.Module):
    """CNN для анализа нескольких временных фреймов одновременно"""
    
    def __init__(self, input_channels: int = 5, hidden_channels: List[int] = None,
                 output_features: int = 64, sequence_lengths: Dict[str, int] = None,
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        if sequence_lengths is None:
            sequence_lengths = {"5m": 50, "15m": 40, "1h": 30}  # Оптимально для анализа паттернов            
        
        self.timeframes = list(sequence_lengths.keys())
        self.output_features = output_features
        
        # Отдельные CNN для каждого временного фрейма
        self.cnn_models = nn.ModuleDict()
        for timeframe, seq_len in sequence_lengths.items():
            self.cnn_models[timeframe] = TradingCNN(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_features=output_features // len(sequence_lengths),
                sequence_length=seq_len,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
        
        # Фьюжн слой для объединения признаков от разных фреймов
        total_features = (output_features // len(sequence_lengths)) * len(sequence_lengths)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, output_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_features, output_features),
            nn.LayerNorm(output_features)
        )
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        data: словарь с данными для каждого временного фрейма
        {"5m": tensor, "15m": tensor, "1h": tensor}
        """
        features = []
        
        for timeframe in self.timeframes:
            if timeframe in data:
                frame_features = self.cnn_models[timeframe](data[timeframe])
                features.append(frame_features)
            else:
                # Если данных нет, создаем нулевые признаки
                zero_features = torch.zeros(
                    data[list(data.keys())[0]].size(0), 
                    self.output_features // len(self.timeframes),
                    device=data[list(data.keys())[0]].device
                )
                features.append(zero_features)
        
        # Объединяем признаки от всех фреймов
        combined_features = torch.cat(features, dim=1)
        
        # Финальная обработка
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features


class CNNFeatureExtractor(nn.Module):
    """Замороженный CNN для извлечения признаков в DQN"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        
        self.device = device
        
        # Загружаем предобученную модель
        checkpoint = torch.load(model_path, map_location=device)
        
        # Создаем модель с теми же параметрами
        self.cnn_model = MultiTimeframeCNN(
            input_channels=checkpoint['config']['input_channels'],
            hidden_channels=checkpoint['config']['hidden_channels'],
            output_features=checkpoint['config']['output_features'],
            sequence_lengths=checkpoint['config']['sequence_lengths'],
            dropout_rate=checkpoint['config']['dropout_rate'],
            use_batch_norm=checkpoint['config']['use_batch_norm']
        )
        
        # Загружаем веса
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Замораживаем все параметры
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        
        self.cnn_model.eval()  # Переводим в режим оценки
        
        self.output_size = checkpoint['config']['output_features']
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Извлечение признаков без градиентов"""
        with torch.no_grad():
            return self.cnn_model(data)
    
    def extract_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Публичный метод для извлечения признаков"""
        return self.forward(data)


class PricePredictionCNN(nn.Module):
    """CNN для предсказания движения цены (обучение)"""
    
    def __init__(self, input_channels: int = 5, hidden_channels: List[int] = None,
                 sequence_length: int = 50, dropout_rate: float = 0.3,
                 use_batch_norm: bool = True, num_classes: int = 3):
        super().__init__()
        
        # Используем TradingCNN как backbone
        self.backbone = TradingCNN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_features=128,  # Больше признаков для предсказания
            sequence_length=sequence_length,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        # Классификатор для предсказания движения цены
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)  # 0: down, 1: sideways, 2: up
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание класса движения цены"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков для DQN (без классификатора)"""
        return self.backbone(x)
