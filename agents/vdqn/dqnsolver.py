from agents.vdqn.dqnn import DQNN
from envs.dqn_model.gym.gutils import check_nan, setup_wandb
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
        self.cfg      = cfg or vDqnConfig()
        self.n_step   = getattr(self.cfg, "n_step", 3)   # 3‑5 шагов
        self.n_queue  = deque(maxlen=self.n_step)                
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
        
        # Target Network (часто используется в DQN для стабильности)
        # Это копия основной модели, параметры которой обновляются реже
        self.target_model = DQNN(
            obs_dim=observation_space,
            act_dim=action_space,
            hidden_sizes=self.cfg.hidden_sizes
            ).to(self.cfg.device)
        
        print("[DQN] model device:", next(self.model.parameters()).device)  # <-- self.model        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.SmoothL1Loss() # Для вычисления Q-значений
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                             self.optimizer, gamma=0.999)
        
        if load:
            self.load_model()
            self.load_state()
                
        self.update_target_model() # Инициализируем целевую сеть
        self.target_model.eval() # Целевая сеть всегда в режиме оценки

    def update_target_model(self):
        """Копирует веса из основной модели в целевую модель."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, gamma_n=1.0):
        self.memory.append((
            torch.tensor(state,      dtype=torch.float32),  # s₀
            torch.tensor(action,     dtype=torch.long),     # a₀
            torch.tensor(reward,     dtype=torch.float32),  # Rₙ
            torch.tensor(next_state, dtype=torch.float32),  # sₙ
            torch.tensor(done,       dtype=torch.bool),     # doneₙ
            torch.tensor(gamma_n,    dtype=torch.float32)   # γⁿ
        ))
           
                
    def store_transition(self, s, a, r, s_next, done):
        # 1. кладём переход в очередь
        self.n_queue.append((s, a, r, s_next, done))

        # 2. если набралось n шагов ИЛИ эпизод завершён
        if len(self.n_queue) == self.n_step or done:
            R_n, gamma_pow = 0.0, 1.0
            for (_, _, r_i, _, _) in self.n_queue:
                R_n      += gamma_pow * r_i
                gamma_pow *= self.cfg.gamma    # γ, γ², γ³ …

            s0, a0, _, s_n, d_n = self.n_queue[0]

            # 3. сохраняем «укрупнённый» переход в буфер
            self.remember(s0, a0, R_n, s_n, d_n, gamma_pow)

            # 4. если эпизод закончился — очистить очередь
            if done:
                self.n_queue.clear()                
                
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
    
    def experience_replay(self, need_metrics: bool = False):
        """
        ↩︎ did_step: bool
        td_loss : torch.Tensor|None (detached)
        abs_q   : torch.Tensor|None (detached)
        q_gap   : torch.Tensor|None (detached)
        """
        if len(self.memory) < self.cfg.batch_size:
            return False, None, None, None

        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones, gammas = zip(*batch)

        device = self.cfg.device

        # ВАЖНО: явные типы
        states      = torch.stack(states,      dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
        actions     = torch.stack(actions,     dim=0).to(device=device, dtype=torch.long,     non_blocking=True)  # индексы для gather
        rewards     = torch.stack(rewards,     dim=0).to(device=device, dtype=torch.float32,  non_blocking=True)
        next_states = torch.stack(next_states, dim=0).to(device=device, dtype=torch.float32,  non_blocking=True)
        dones       = torch.stack(dones,       dim=0).to(device=device, dtype=torch.bool,     non_blocking=True)
        gammas   = torch.tensor(gammas, device=device, dtype=torch.float32) 

        # ---- Q(s,a) ----
        q_all     = self.model(states)                     # (B, A)
        current_q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ---- r + γ·Q̄(s', argmax_a Q_online(s',a)) ---- (Double DQN)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q  =    rewards + gammas * next_q * (~dones).float()   # !!! γⁿ здесь

        # ---- loss ----
        loss = self.criterion(current_q, target_q)

        # ---- back‑prop ----
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not check_nan("grad", *(p.grad for p in self.model.parameters() if p.grad is not None)):
            self.optimizer.zero_grad()
            return False, torch.tensor(float('nan')), None, None
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()     
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                torch.nan_to_num_(layer.weight, nan=0.0, posinf=1e3, neginf=-1e3)   
                
        self.scheduler.step()
        

        abs_q = q_gap = None
        if need_metrics:
            with torch.no_grad():
                abs_q = q_all.abs().mean().detach()
                q_gap = (q_all - self.target_model(states)).abs().mean().detach()

        return True, loss.detach(), abs_q, q_gap

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




