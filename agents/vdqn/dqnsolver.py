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
import heapq

# Ваши импорты
from agents.vdqn.cfg.vconfig import vDqnConfig

cfg = vDqnConfig()

print(f"Используемое устройство для PyTorch: {cfg.device}")

class PrioritizedReplayBuffer:
    """Приоритизированный буфер воспроизведения опыта с оптимизацией для GPU"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        
        # GPU оптимизации
        self.device = cfg.device
        
    def push(self, state, action, reward, next_state, done, gamma_n=1.0):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, gamma_n))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, gamma_n)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []
        
        # Выбираем индексы на основе приоритетов
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Вычисляем веса для importance sampling
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Увеличиваем beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, gamma_ns = zip(*batch)
        
        return states, actions, rewards, next_states, dones, gamma_ns, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

# --- DQNSolver адаптированный под PyTorch с GPU оптимизациями ---
class DQNSolver:
    def __init__(self, observation_space, action_space, load=False):
        self.cfg      = cfg or vDqnConfig()
        self.n_step   = getattr(self.cfg, "n_step", 3)   # 3‑5 шагов
        self.n_queue  = deque(maxlen=self.n_step)                
        self.epsilon      = cfg.eps_start
        self.action_space = action_space
        
        # Используем Prioritized Replay Buffer если включен
        if self.cfg.prioritized:
            self.memory = PrioritizedReplayBuffer(
                self.cfg.memory_size, 
                self.cfg.alpha, 
                self.cfg.beta, 
                self.cfg.beta_increment
            )
        else:
            self.memory = deque(maxlen=self.cfg.memory_size)
        
        self._eps_decay_rate = math.exp(math.log(self.cfg.eps_final / self.cfg.eps_start) / self.cfg.eps_decay_steps)

        # Создаем модели
        self.model = DQNN(
            observation_space, 
            action_space, 
            self.cfg.hidden_sizes,
            dropout_rate=self.cfg.dropout_rate,
            layer_norm=self.cfg.layer_norm,
            dueling=self.cfg.dueling_dqn
        ).to(self.cfg.device)
        
        self.target_model = DQNN(
            observation_space, 
            action_space, 
            self.cfg.hidden_sizes,
            dropout_rate=self.cfg.dropout_rate,
            layer_norm=self.cfg.layer_norm,
            dueling=self.cfg.dueling_dqn
        ).to(self.cfg.device)
        
        # Копируем веса
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Оптимизатор с увеличенным learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.lr,
            weight_decay=1e-4,
            eps=1e-7
        )
        
        # Scheduler для динамического learning rate
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=1000, 
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Счетчики
        self.step_count = 0
        self.update_count = 0
        
        # GPU оптимизации
        if self.cfg.device.type == 'cuda':
            # Включаем cudnn benchmark для ускорения
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Mixed precision training
            if self.cfg.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
        
        # Загружаем модель если нужно
        if load:
            self.load_model()
            
        # Переводим модели в режим оценки для ускорения
        self.model.eval()
        self.target_model.eval()

    def update_target_model(self):
        """Копирует веса из основной модели в целевую модель."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, gamma_n=1.0):
        if self.cfg.prioritized:
            self.memory.push(state, action, reward, next_state, done, gamma_n)
        else:
            self.memory.append((
                torch.tensor(state,      dtype=torch.float32),
                torch.tensor(action,     dtype=torch.long),     
                torch.tensor(reward,     dtype=torch.float32),  
                torch.tensor(next_state, dtype=torch.float32),  
                torch.tensor(done,       dtype=torch.bool),     
                torch.tensor(gamma_n,    dtype=torch.float32)   
            ))
           
                
    def store_transition(self, state, action, reward, next_state, done, gamma_n=1.0):
        """Сохраняет переход в replay buffer с n-step returns"""
        # Добавляем переход в n-step очередь
        self.n_queue.append((state, action, reward, next_state, done))
        
        # Если очередь заполнена, вычисляем n-step return
        if len(self.n_queue) == self.n_step:
            s0, a0, r0, _, _ = self.n_queue[0]
            
            # Вычисляем n-step return
            R_n = r0
            gamma_pow = 1.0
            for i in range(1, self.n_step):
                _, _, r, _, d = self.n_queue[i]
                if d:  # Если эпизод закончился, прерываем
                    break
                gamma_pow *= self.cfg.gamma
                R_n += gamma_pow * r
            
            # Получаем финальное состояние
            s_n = self.n_queue[-1][3]  # next_state из последнего перехода
            d_n = self.n_queue[-1][4]  # done из последнего перехода
            
            # Сохраняем "укрупнённый" переход в буфер
            if self.cfg.prioritized:
                self.memory.push(s0, a0, R_n, s_n, d_n, gamma_pow)
            else:
                self.memory.append((s0, a0, R_n, s_n, d_n, gamma_pow))
            
            # Если эпизод закончился — очищаем очередь
            if done:
                self.n_queue.clear()
                
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        
        # Проверяем на NaN и заменяем на нули
        if torch.isnan(torch.tensor(state)).any():
            state = np.nan_to_num(state, nan=0.0)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.cfg.device)
            q_values = self.model(state_tensor)
            
            # Проверяем на NaN в выходе
            if torch.isnan(q_values).any():
                print("Warning: NaN detected in Q-values, using random action")
                return random.randrange(self.action_space)
            
            return q_values.argmax().item()

    def experience_replay(self, need_metrics=False):
        if len(self.memory) < self.cfg.batch_size:
            return False, None, None, None
        
        # Получаем batch
        if self.cfg.prioritized:
            states, actions, rewards, next_states, dones, gamma_ns, weights, indices = \
                self.memory.sample(self.cfg.batch_size)
            
            # Конвертируем в тензоры с оптимизацией для GPU
            states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.cfg.device, non_blocking=True)
            actions = torch.LongTensor(actions).to(self.cfg.device, non_blocking=True)
            rewards = torch.FloatTensor(rewards).to(self.cfg.device, non_blocking=True)
            next_states = torch.stack([torch.FloatTensor(s) for s in next_states]).to(self.cfg.device, non_blocking=True)
            dones = torch.BoolTensor(dones).to(self.cfg.device, non_blocking=True)
            gamma_ns = torch.FloatTensor(gamma_ns).to(self.cfg.device, non_blocking=True)
            weights = weights.to(self.cfg.device, non_blocking=True)
        else:
            batch = random.sample(self.memory, self.cfg.batch_size)
            states, actions, rewards, next_states, dones, gamma_ns = zip(*batch)
            
            states = torch.stack(states).to(self.cfg.device, non_blocking=True)
            actions = actions.to(self.cfg.device, non_blocking=True)
            rewards = rewards.to(self.cfg.device, non_blocking=True)
            next_states = torch.stack(next_states).to(self.cfg.device, non_blocking=True)
            dones = dones.to(self.cfg.device, non_blocking=True)
            gamma_ns = gamma_ns.to(self.cfg.device, non_blocking=True)
            weights = torch.ones(self.cfg.batch_size).to(self.cfg.device, non_blocking=True)
        
        # Проверяем на NaN
        if torch.isnan(states).any() or torch.isnan(next_states).any():
            print("Warning: NaN detected in states, skipping batch")
            return False, None, None, None
        
        # Переводим модель в режим обучения
        self.model.train()
        
        # Double DQN: используем основную сеть для выбора действий, целевую для оценки
        if self.cfg.double_dqn:
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(dim=1)
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
        
        # Вычисляем target Q-values
        target_q_values = rewards + (gamma_ns * next_q_values * ~dones)
        
        # Текущие Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Вычисляем loss с importance sampling weights
        td_errors = target_q_values - current_q_values
        loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        
        # Mixed Precision Training для ускорения
        if self.scaler is not None:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Градиентный клиппинг
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Обычное обучение
            self.optimizer.zero_grad()
            loss.backward()
            
            # Градиентный клиппинг
            if self.cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            
            self.optimizer.step()
        
        # Обновляем приоритеты если используется PER
        if self.cfg.prioritized:
            priorities = (torch.abs(td_errors) + 1e-6).detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # Обновляем scheduler
        self.scheduler.step()
        
        # Переводим модель обратно в режим оценки для ускорения inference
        self.model.eval()
        
        # Метрики
        if need_metrics:
            abs_q = current_q_values.abs().mean().item()
            q_gap = (target_q_values - current_q_values).abs().mean().item()
            return True, loss.item(), abs_q, q_gap
        
        return True, None, None, None

    def print_trade_stats(self, trades):
        if not trades:
            return {"trades_count": 0, "winrate": 0.0, "avg_profit": 0.0, "avg_loss": 0.0}
        
        profits = [t["profit"] for t in trades if t["profit"] > 0]
        losses = [t["profit"] for t in trades if t["profit"] < 0]
        
        winrate = len(profits) / len(trades) if trades else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_roi = np.mean([t["profit"] for t in trades]) if trades else 0.0
        
        pl_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
        
        # Считаем плохие сделки
        bad_trades = [t for t in trades if abs(t["profit"]) < 0.001]  # <0.1%
        bad_trades_count = len(bad_trades)
        
        stats = {
            "trades_count": len(trades),
            "winrate": winrate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_roi": avg_roi,
            "pl_ratio": pl_ratio,
            "bad_trades_count": bad_trades_count
        }
        
        print(f"📊 - Trades: {stats['trades_count']}, Winrate: {stats['winrate']*100:.2f}%, "
              f"Avg P: {stats['avg_profit']:.3f}, Avg L: {stats['avg_loss']:.3f}, "
              f"P/L ratio: {stats['pl_ratio']:.2f}")
        print(f"❗ Bad trades (<0.1% ROI): {stats['bad_trades_count']}")
        
        return stats

    def save(self):
        """Сохраняет модель и replay buffer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'cfg': self.cfg
        }, self.cfg.model_path)
        
        # Сохраняем replay buffer
        with open(self.cfg.buffer_path, 'wb') as f:
            pickle.dump(self.memory, f, protocol=HIGHEST_PROTOCOL)
        
        print(f"Модель сохранена в {self.cfg.model_path}")
        print(f"Replay buffer сохранен в {self.cfg.buffer_path}")

    def load_model(self):
        """Загружает модель с проверкой совместимости архитектуры"""
        if os.path.exists(self.cfg.model_path):
            try:
                checkpoint = torch.load(self.cfg.model_path, map_location=self.cfg.device)
                
                # Проверяем совместимость архитектуры
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    
                    # Проверяем размеры первого слоя
                    if 'net.feature_layers.0.weight' in model_state:
                        saved_input_size = model_state['net.feature_layers.0.weight'].shape[1]
                        current_input_size = self.model.net.feature_layers[0].weight.shape[1]
                        
                        if saved_input_size != current_input_size:
                            print(f"⚠️ Архитектура несовместима: сохраненная {saved_input_size}, текущая {current_input_size}")
                            print("🔄 Создаем новую модель с текущей архитектурой")
                            return
                    
                    # Загружаем модель
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.target_model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Загружаем остальные параметры если они есть
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if 'epsilon' in checkpoint:
                        self.epsilon = checkpoint['epsilon']
                        
                    print(f"✅ Модель загружена из {self.cfg.model_path}")
                else:
                    print(f"⚠️ Неверный формат checkpoint в {self.cfg.model_path}")
                    
            except Exception as e:
                print(f"❌ Ошибка при загрузке модели: {e}")
                print("🔄 Создаем новую модель")
        else:
            print(f"📝 Файл модели {self.cfg.model_path} не найден, создаем новую модель")

    def load_state(self):
        """Загружает replay buffer"""
        if os.path.exists(self.cfg.buffer_path):
            with open(self.cfg.buffer_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Replay buffer загружен из {self.cfg.buffer_path}")
        else:
            print(f"Файл replay buffer {self.cfg.buffer_path} не найден")

    def soft_update(self, tau=0.01):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)        

    def save_model(self):
        """Сохраняет модель (аналог save)"""
        self.save()
        
    def update_target_model(self):
        """Hard update target network для ускорения"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def store_transition(self, state, action, reward, next_state, done, gamma_n=1.0):
        """Сохраняет переход в replay buffer с n-step returns"""
        # Добавляем переход в n-step очередь
        self.n_queue.append((state, action, reward, next_state, done))
        
        # Если очередь заполнена, вычисляем n-step return
        if len(self.n_queue) == self.n_step:
            s0, a0, r0, _, _ = self.n_queue[0]
            
            # Вычисляем n-step return
            R_n = r0
            gamma_pow = 1.0
            for i in range(1, self.n_step):
                _, _, r, _, d = self.n_queue[i]
                if d:  # Если эпизод закончился, прерываем
                    break
                gamma_pow *= self.cfg.gamma
                R_n += gamma_pow * r
            
            # Получаем финальное состояние
            s_n = self.n_queue[-1][3]  # next_state из последнего перехода
            d_n = self.n_queue[-1][4]  # done из последнего перехода
            
            # Сохраняем "укрупнённый" переход в буфер
            if self.cfg.prioritized:
                self.memory.push(s0, a0, R_n, s_n, d_n, gamma_pow)
            else:
                self.memory.append((s0, a0, R_n, s_n, d_n, gamma_pow))
            
            # Если эпизод закончился — очищаем очередь
            if done:
                self.n_queue.clear()
                
    def remember(self, state, action, reward, next_state, done, gamma_n=1.0):
        """Альтернативное название для store_transition"""
        self.store_transition(state, action, reward, next_state, done, gamma_n)
        




