import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Отключаем компиляцию PyTorch для Noisy Networks
torch._dynamo.config.suppress_errors = True


class NoisyLinear(nn.Module):
    """Noisy Linear слой для лучшего exploration без ε-greedy"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        # Отключаем компиляцию для этого слоя
        super().__init__()
        self._no_compile = True  # Флаг для отключения компиляции
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Детерминированные веса и смещения
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Инициализация
        self._reset_parameters()
        self._reset_noise()
    
    def _reset_parameters(self):
        """Инициализация параметров"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _reset_noise(self):
        """Сброс шума"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.register_buffer('weight_epsilon', epsilon_out.outer(epsilon_in))
        self.register_buffer('bias_epsilon', epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Масштабирование шума"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход с шумом"""
        if self.training:
            # Убеждаемся, что все тензоры на одном устройстве
            device = x.device
            weight_epsilon = self.weight_epsilon.to(device)
            bias_epsilon = self.bias_epsilon.to(device)
            
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_noise(self):
        """Сброс шума (вызывать после каждого обновления)"""
        self._reset_noise()

class DuelingDQN(nn.Module):
    """Dueling DQN архитектура для лучшего обучения Q-значений"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], 
                 dropout_rate: float = 0.2, layer_norm: bool = True):
        super().__init__()
        self._no_compile = True  # Отключаем компиляцию для Dueling DQN
        
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


class NoisyDuelingDQN(nn.Module):
    """Dueling DQN с Noisy Linear слоями для лучшего exploration"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], 
                 dropout_rate: float = 0.2, layer_norm: bool = True):
        super().__init__()
        self._no_compile = True  # Отключаем компиляцию для Noisy Networks
        
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
        
        # Value stream (V(s)) - используем обычные слои
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Advantage stream (A(s,a)) - используем Noisy Linear для exploration
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            NoisyLinear(feature_dim // 2, act_dim)
        )
        
        # Инициализация весов
        for module in self.value_stream.modules():
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
    
    def reset_noise(self):
        """Сброс шума во всех Noisy Linear слоях"""
        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DistributionalDQN(nn.Module):
    """Distributional DQN - предсказывает распределение Q-значений вместо среднего"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...], 
                 n_atoms: int = 51, v_min: float = -10.0, v_max: float = 10.0,
                 dropout_rate: float = 0.2, layer_norm: bool = True):
        super().__init__()
        self._no_compile = True  # Отключаем компиляцию для Distributional DQN
        
        self.act_dim = act_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)
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
        
        # Distributional head: предсказываем распределение для каждого действия
        feature_dim = hidden_sizes[-1]
        self.distribution_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, act_dim * n_atoms)
        )
        
        # Инициализация весов
        for module in self.distribution_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Проходим через общие слои
        for layer in self.feature_layers:
            x = layer(x)
        
        # Получаем распределения для всех действий
        batch_size = x.size(0)
        distribution_logits = self.distribution_head(x)
        
        # Reshape: [batch_size, act_dim * n_atoms] -> [batch_size, act_dim, n_atoms]
        distribution_logits = distribution_logits.view(batch_size, self.act_dim, self.n_atoms)
        
        # Применяем softmax для получения вероятностей
        distributions = F.softmax(distribution_logits, dim=-1)
        
        # Вычисляем ожидаемые Q-значения для обратной совместимости
        q_values = torch.sum(distributions * self.support.to(x.device), dim=-1)
        
        return q_values, distributions
    
    def get_q_values(self, x):
        """Возвращает только Q-значения для обратной совместимости"""
        q_values, _ = self.forward(x)
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
        self._no_compile = True  # Отключаем компиляцию для DQNN
        
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