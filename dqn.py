# Импорты PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Ваши импорты
from model.dqn_model.gym.crypto_trading_env import CryptoTradingEnv
from utils.env import dqncfg # Предполагаем, что dqncfg содержит ваши константы

import numpy as np
import random
import os # Для проверки существования файла модели
import pickle
from collections import deque
import gym
from gym import spaces
from gym.utils import seeding

import wandb

wandb.init(
    project="medoedai-medoedai",
    name="medoedai-medoedai",
    config={
        "learning_rate": dqncfg.LEARNING_RATE,
        "batch_size": dqncfg.BATCH_SIZE,
        "exploration_max": dqncfg.EXPLORATION_MAX,
        "exploration_min": dqncfg.EXPLORATION_MIN,
        "exploration_decay": dqncfg.EXPLORATION_DECAY,
        "memory_size": dqncfg.MEMORY_SIZE,
        "gamma": dqncfg.GAMMA
    }
)

# --- Константы из dqncfg (предполагаем, что они определены там) ---
# Если dqncfg - это модуль или объект, убедитесь, что эти переменные доступны.
# Например, если они в dqncfg.py:
EXPLORATION_MAX = dqncfg.EXPLORATION_MAX
EXPLORATION_MIN = dqncfg.EXPLORATION_MIN
EXPLORATION_DECAY = dqncfg.EXPLORATION_DECAY
MEMORY_SIZE = dqncfg.MEMORY_SIZE
BATCH_SIZE = dqncfg.BATCH_SIZE
LEARNING_RATE = dqncfg.LEARNING_RATE
GAMMA = dqncfg.GAMMA
# --- Конец констант ---

# Определение устройства (GPU или CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство для PyTorch: {DEVICE}")

# --- Модель Q-сети на PyTorch ---
class DQNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(DQNN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- DQNSolver адаптированный под PyTorch ---
class DQNSolver:
    def __init__(self, observation_space, action_space, load=False, model_path="dqn_model.pth", buffer_path="replay_buffer.pkl"):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Инициализация модели PyTorch
        self.model = DQNN(observation_space, action_space).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss() # Для вычисления Q-значений

        if load:
        # Загрузка модели
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                self.model.eval()
                print("✅ Модель загружена из", model_path)
            else:
                print("⚠️ Файл модели не найден. Создана новая модель.")

            # Загрузка replay buffer
            if os.path.exists(buffer_path):
                try:
                    print("Загрузка буфера...")
                    with open(buffer_path, "rb") as f:
                        self.memory = pickle.load(f)
                    print(f"✅ Replay buffer загружен из {buffer_path}, {len(self.memory)} записей.")
                except Exception as e:
                    print("⚠️ Ошибка при загрузке replay buffer:", e)
            else:
                print("⚠️ Файл replay buffer не найден. Память не восстановлена.")

        self.model_path = model_path
        self.buffer_path = buffer_path
        
        # Target Network (часто используется в DQN для стабильности)
        # Это копия основной модели, параметры которой обновляются реже
        self.target_model = DQNN(observation_space, action_space).to(DEVICE)
        self.update_target_model() # Инициализируем целевую сеть
        self.target_model.eval() # Целевая сеть всегда в режиме оценки


    def update_target_model(self):
        """Копирует веса из основной модели в целевую модель."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Преобразуем numpy массивы в тензоры PyTorch
        state_t = torch.tensor(state, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.long) # Действие - это индекс
        reward_t = torch.tensor(reward, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        done_t = torch.tensor(done, dtype=torch.bool)
        
        self.memory.append((state_t, action_t, reward_t, next_state_t, done_t))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        
        # Преобразование состояния в тензор и отправка на устройство
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Отключаем вычисление градиентов для предсказания
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Получаем действие с максимальным Q-значением
        return torch.argmax(q_values[0]).item()
          
    def save(self):
        # Сохраняем модель
        torch.save(self.model.state_dict(), self.model_path)
        # Сохраняем replay buffer во временный файл
        tmp_path = self.buffer_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(self.memory, f)
        # Перемещаем временный файл на место основного
        os.replace(tmp_path, self.buffer_path)       

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        # Разделяем батч на отдельные тензоры
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(DEVICE)
        actions = torch.stack(actions).to(DEVICE)
        rewards = torch.stack(rewards).to(DEVICE)
        next_states = torch.stack(next_states).to(DEVICE)
        dones = torch.stack(dones).to(DEVICE)

        # Вычисление текущих Q-значений (для выбранных действий)
        # model(states) возвращает Q-значения для всех действий
        # .gather(1, actions.unsqueeze(1)) выбирает Q-значение для фактического действия
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Вычисление целевых Q-значений (используем target_model для стабильности)
        # detached() останавливает градиенты от распространения обратно в target_model
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0] # max(1)[0] берет максимальное Q-значение
            
        # Q-update: Q*(s,a) = r + gamma * max_a' Q(s',a')
        # Если эпизод завершен (done), следующее состояние не имеет значения, поэтому Q_update = reward
        target_q_values = rewards + (GAMMA * next_q_values * (~dones))

        # Вычисление функции потерь
        loss = self.criterion(current_q_values, target_q_values)

        # Обнуление градиентов, обратное распространение, обновление весов
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_model(dfs: dict, load_previous: bool = False, episodes: int = 10000, model_path: str = "dqn_model.pth"):
    """
    Обучает модель DQN для торговли криптовалютой.

    Args:
        dfs (dict): Словарь с Pandas DataFrames для разных таймфреймов (df_5min, df_15min, df_1h).
        load_previous (bool): Загружать ли ранее сохраненную модель.
        episodes (int): Количество эпизодов для обучения.
        model_path (str): Путь для сохранения/загрузки модели.
    Returns:
        str: Сообщение о завершении обучения.
    """
    
    previous_cumulative_reward = -float('inf')
    
    # Теперь CryptoTradingEnv принимает словарь с DataFrame'ами
    env = CryptoTradingEnv(dfs=dfs) 
    
    observation_space_dim = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    dqn_solver = DQNSolver(observation_space_dim, action_space, load=load_previous, model_path=model_path)

    target_update_frequency = 10 # Обновлять целевую сеть каждые N эпизодов

    global_step = 0
    successful_episodes = 0

    for episode in range(episodes):
        # Переводим модель в режим обучения
        dqn_solver.model.train() 
        state = env.reset() # env.reset() теперь возвращает начальное состояние
        step_in_episode = 0
        
        # state уже в правильной форме (NumPy массив), не нужно reshape здесь

        while True:
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            
            dqn_solver.experience_replay() # Вызываем replay на каждом шаге или с определенной частотой
            
            global_step += 1                        
            
            wandb.log({
                            "step": global_step,
                            "episode": episode + 1,
                            #"action": action,
                            "reward": reward,
                            #"crypto_held": info.get('crypto_held', 0),
                            "cumulative_reward": env.cumulative_reward,
                            "net_profit": info.get('net_profit', None),        # Добавлено
                        })

                    
            current_cumulative_reward = env.cumulative_reward
            if terminal:      
                                
                if info.get("total_profit", 0) > 0:
                    dqn_solver.exploration_rate = max(
                        EXPLORATION_MIN,
                        dqn_solver.exploration_rate * EXPLORATION_DECAY
                    )       
                    successful_episodes += 1
                elif previous_cumulative_reward is not None and \
                    current_cumulative_reward < previous_cumulative_reward and \
                    abs(current_cumulative_reward - previous_cumulative_reward) > abs(previous_cumulative_reward) * 0.2:
                    dqn_solver.exploration_rate = max(
                        EXPLORATION_MIN,
                        dqn_solver.exploration_rate * EXPLORATION_DECAY
                    )     
                    successful_episodes += 1
                
                previous_cumulative_reward = current_cumulative_reward 
                                                                         
                print(f"Эпизод {episode+1}/{episodes} завершен. "
                    f"Общая прибыль: {info.get('total_profit', 0):.2f}, "
                    f"Баланс: {info.get('current_balance', env.initial_balance):.2f}, "
                    f"BTC: {info.get('crypto_held', 0):.4f}, "
                    f"Cumulative reward: {env.cumulative_reward:.2f}, "
                    f"Epsilon: {dqn_solver.exploration_rate:.4f}, "
                    f"Successful_episodes: {successful_episodes:.2f}")

                # Логирование в wandb
                wandb.log({
                    "step": global_step,
                    "episode": episode + 1,
                    "total_profit": info.get('total_profit', 0),
                    "final_balance": info.get('current_balance', env.initial_balance),
                    "final_crypto_held": info.get('crypto_held', 0),
                    "final_cumulative_reward": env.cumulative_reward,
                    "epsilon": dqn_solver.exploration_rate
                })
                break
            
        # Обновление целевой сети
        if (episode + 1) % target_update_frequency == 0:
            dqn_solver.update_target_model()
            print(f"Целевая сеть обновлена после эпизода {episode+1}")

        # Сохранение модели (частота сохранения)
        if (episode + 1) % 100 == 0: 
            dqn_solver.save()
            print(f"Модель и replay buffer сохранены после эпизода {episode+1}")

    # Сохранение финальной модели
    dqn_solver.save()
    print("Финальная модель сохранена.")
    return "Обучение завершено"


def trade_once(state: np.ndarray, observation_space_dim: int, model_path: str = "dqn_model.pth"): 
    """
    Принимает торговое решение на основе текущего состояния.

    Args:
        state (np.ndarray): Текущий вектор состояния.
        observation_space_dim (int): Размерность пространства наблюдений (длина вектора состояния).
        model_path (str): Путь к файлу сохраненной модели.
    Returns:
        str: Рекомендованное торговое действие ("BUY", "SELL", "HOLD").
    """
    
    if not os.path.exists(model_path):
        return "Модель не найдена, сначала обучите её!"

    # Мы знаем, что action_space_dim = 3 (HOLD, BUY, SELL) из вашего Gym окружения.
    # Или можно получить это из окружения, если оно было зарегистрировано:
    # action_space_dim = gym.make(dqncfg.ENV_NAME).action_space.n 
    action_space_dim = 3 # Поскольку действия фиксированы, можно захардкодить или передать из dqncfg

    model = DQNN(observation_space_dim, action_space_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Переводим модель в режим оценки для предсказания

    # Преобразование состояния в тензор и отправка на устройство
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): # Отключаем вычисление градиентов для инференса
        q_values = model(state_tensor)
    
    action = torch.argmax(q_values[0]).item() # Получаем индекс действия

    actions_map = {0: "HOLD", 1: "BUY", 2: "SELL"} 
    executed_action = actions_map[action]

    print(f"Торговое действие: {executed_action} с Q-values {q_values.cpu().numpy()}")
    return executed_action