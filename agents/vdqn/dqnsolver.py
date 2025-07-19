from agents.vdqn.dqnn import DQNN
from envs.dqn_model.gym.gutils import setup_wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import pickle
from pickle import HIGHEST_PROTOCOL
import math
from collections import deque

# Ваши импорты
from agents.vdqn.cfg.vconfig import vDqnConfig

cfg = vDqnConfig()

print(f"Используемое устройство для PyTorch: {cfg.device}")

# --- DQNSolver адаптированный под PyTorch ---
class DQNSolver:
    def __init__(self, observation_space, action_space, load=False):
        self.cfg          = cfg
        self.epsilon      = cfg.eps_start
        self.action_space = action_space
        self.memory = deque(maxlen=self.cfg.memory_size)       
        
        self._eps_decay_rate = math.exp(math.log(self.cfg.eps_final / self.cfg.eps_start) / self.cfg.eps_decay_steps)        # ≈ 0.99986 при 10k шагов

        # Инициализация модели PyTorch
        self.model = DQNN(
            obs_dim=observation_space,
            act_dim=action_space,
            hidden_sizes=self.cfg.hidden_sizes   # берём из dataclass‑конфига
            ).to(self.cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.MSELoss() # Для вычисления Q-значений
        
        if load:
            self.load_model()
            self.load_state()
        
        # Target Network (часто используется в DQN для стабильности)
        # Это копия основной модели, параметры которой обновляются реже
        self.target_model = DQNN(
            obs_dim=observation_space,
            act_dim=action_space,
            hidden_sizes=self.cfg.hidden_sizes
            ).to(self.cfg.device)
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
        if np.random.rand() < self.epsilon:
            return np.random.choice([0,1,2], p=[0.5,0.25,0.25])
        
        # Преобразование состояния в тензор и отправка на устройство
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
        
        # Отключаем вычисление градиентов для предсказания
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Получаем действие с максимальным Q-значением
        return torch.argmax(q_values[0]).item()
          
    def save(self):
        # Сохраняем модель
        torch.save(self.model.state_dict(), self.cfg.model_path)

        # Сохраняем старый буфер в .bak
        if os.path.exists(self.cfg.buffer_path):
            os.rename(self.cfg.buffer_path, self.cfg.buffer_path + ".bak")

        # Убедимся, что temp-папка есть
        tmp_path = f"./temp/{self.cfg.buffer_path}.tmp"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        agent_data = {
            "memory": self.memory,
            "epsilon": self.epsilon
        }

        with open(tmp_path, "wb") as f:
            pickle.dump(agent_data, f, protocol=HIGHEST_PROTOCOL)

        os.replace(tmp_path, self.cfg.buffer_path)

    def experience_replay(self):
        
        """
        ↩︎  did_step: bool          — был ли сделан grad шаг
            td_loss : float | None — Huber/MSE loss
            abs_q   : float | None — ⟨|Q(s,a)|⟩ по батчу
            q_gap   : float | None — ⟨|Q_online   Q_target|⟩
        """
        
        if len(self.memory) < self.cfg.batch_size:
            return False, None, None, None      # ← ничего не делали


        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).to(self.cfg.device)
        actions     = torch.stack(actions).to(self.cfg.device)
        rewards     = torch.stack(rewards).to(self.cfg.device)
        next_states = torch.stack(next_states).to(self.cfg.device)
        dones       = torch.stack(dones).to(self.cfg.device)

        # ---- Q(s,a) ----
        q_all      = self.model(states)                 # (B, n_actions)
        current_q  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- r + γ·max Q' ----
        with torch.no_grad():
            next_q  = self.target_model(next_states).max(1)[0]
        target_q = rewards + self.cfg.gamma * next_q * (~dones)

        # ---- loss ----
        loss = self.criterion(current_q, target_q)

        # ---- back‑prop ----
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---- доп.‑метрики ----
        abs_q  = q_all.abs().mean().item()                             # ⟨|Q|⟩
        with torch.no_grad():
            q_gap = (self.model(states) - self.target_model(states)).abs().mean().item()

        # сохраняем, чтобы при желании обратиться снаружи
        self.abs_q_mean = abs_q
        self.q_gap      = q_gap

        return True, loss.item(), abs_q, q_gap
          

    def load_model(self):
        if os.path.exists(self.cfg.model_path):
            self.model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.cfg.device))
            self.model.eval()
            print("✅ - Модель загружена из", self.cfg.model_path)
        else:
            print("⚠️ - Файл модели не найден. Создана новая модель.")

    def load_state(self):
        if os.path.exists(self.cfg.buffer_path):
            try:
                print("Загрузка replay buffer и epsilon...")
                with open(self.cfg.buffer_path, "rb") as f:
                    agent_data = pickle.load(f)

                if isinstance(agent_data, dict):
                    self.memory = agent_data.get("memory", deque(maxlen=self.cfg.memory_size))
                    self.epsilon = agent_data.get("epsilon", 1.0)
                else:
                    self.memory = agent_data
                    self.epsilon = self.cfg.eps_start    

                print(f"✅ - Replay buffer загружен из {self.cfg.buffer_path}, {len(self.memory)} записей.")
            except (EOFError, pickle.UnpicklingError):
                with open(self.cfg.buffer_path + ".bak", "rb") as f:
                    agent_data = pickle.load(f)       
                    if isinstance(agent_data, dict):
                        self.memory = agent_data.get("memory", deque(maxlen=self.cfg.memory_size))
                        self.epsilon = agent_data.get("epsilon", self.cfg.eps_start)
                    else:
                        self.memory = agent_data
                        self.epsilon = self.cfg.eps_start
                print(f"✅ - Replay buffer загружен из BAK {self.cfg.buffer_path}.bak, {len(self.memory)} записей.")

            except Exception as e:
                print("⚠️ - Ошибка при загрузке replay buffer:", e)
        else:
            print(f"⚠️ - Файл replay buffer не найден по пути {self.cfg.buffer_path}. Память не восстановлена.")
            
    def print_trade_stats(self, trades=None):
        profits = [t['roi'] for t in trades if t['roi'] > 0]
        losses = [t['roi'] for t in trades if t['roi'] <= 0]
        winrate = len(profits) / len(trades) if trades else 0
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_roi = np.mean([t['roi'] for t in trades]) if trades else 0
        bad_trades = [t for t in trades if t['roi'] < 0.001]
        pl_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float("inf")

        print(f"📊 - Trades: {len(trades)}, Winrate: {winrate:.2%}, Avg P: {avg_profit:.3f}, Avg L: {avg_loss:.3f}, P/L ratio: {pl_ratio:.2f}")
        print(f"❗ Bad trades (<0.1% ROI): {len(bad_trades)}")

        return {
            "trades_count": len(trades),
            "winrate": winrate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,   
            "avg_roi": avg_roi,   
            "pl_ratio": pl_ratio,
            "bad_trades_count": len(bad_trades)
        }
        
    @torch.no_grad()
    def soft_update(self, tau: float = 1e-3):
        """
        Polyak‑обновление: target ← τ·online + (1‑τ)·target
        Вызывается как self.soft_update(tau).
        """
        for tgt_param, src_param in zip(self.target_model.parameters(),
                                        self.model.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tau) * tgt_param.data)        




