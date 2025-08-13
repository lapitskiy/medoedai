import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DuelingDQN(nn.Module):
    """Dueling DQN архитектура для лучшего обучения Q-значений"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], 
                 dropout_rate: float = 0.2, layer_norm: bool = True):
        super().__init__()
        
        self.act_dim = act_dim
        self.layer_norm = layer_norm
        
        # Общие слои для извлечения признаков
        self.feature_layers = nn.ModuleList()
        in_dim = obs_dim
        
        for i, h in enumerate(hidden_sizes[:-1]):
            layer = nn.Linear(in_dim, h)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            self.feature_layers.append(layer)
            
            if layer_norm:
                self.feature_layers.append(nn.LayerNorm(h))
            
            self.feature_layers.append(nn.ReLU())
            self.feature_layers.append(nn.Dropout(dropout_rate))
            in_dim = h
        
        # Последний общий слой
        self.feature_layers.append(nn.Linear(in_dim, hidden_sizes[-1]))
        nn.init.kaiming_uniform_(self.feature_layers[-1].weight, nonlinearity='relu')
        nn.init.zeros_(self.feature_layers[-1].bias)
        
        if layer_norm:
            self.feature_layers.append(nn.LayerNorm(hidden_sizes[-1]))
        
        self.feature_layers.append(nn.ReLU())
        self.feature_layers.append(nn.Dropout(dropout_rate))
        
        # Dueling DQN: разделяем на Value и Advantage
        feature_dim = hidden_sizes[-1]
        
        # Value stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Advantage stream (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, act_dim)
        )
        
        # Инициализация весов
        for module in self.value_stream.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
        for module in self.advantage_stream.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Проходим через общие слои
        for layer in self.feature_layers:
            x = layer(x)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Центрируем advantage для стабильности
        advantage_centered = advantage - advantage.mean(dim=-1, keepdim=True)
        
        q_values = value + advantage_centered
        return q_values

class DQNN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], 
                 dropout_rate: float = 0.2, layer_norm: bool = True, dueling: bool = True):
        """
        obs_dim  — размер входного вектора состояния
        act_dim  — число дискретных действий
        hidden_sizes — (512, 256, 128, …) любой кортеж размеров скрытых слоёв
        dropout_rate — вероятность dropout
        layer_norm — использовать ли Layer Normalization
        dueling — использовать ли Dueling DQN архитектуру
        """
        super().__init__()
        
        if dueling:
            self.net = DuelingDQN(obs_dim, act_dim, hidden_sizes, dropout_rate, layer_norm)
        else:
            # Классическая архитектура с улучшениями
            layers = []
            in_dim = obs_dim
            
            for i, h in enumerate(hidden_sizes):
                layers.append(nn.Linear(in_dim, h))
                
                if layer_norm:
                    layers.append(nn.LayerNorm(h))
                
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_dim = h
            
            layers.append(nn.Linear(in_dim, act_dim))
            self.net = nn.Sequential(*layers)
            
            # Инициализация весов
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)