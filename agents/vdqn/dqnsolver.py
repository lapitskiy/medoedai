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
from pathlib import Path

# Ваши импорты
from agents.vdqn.cfg.vconfig import vDqnConfig

cfg = vDqnConfig()

print(f"Используемое устройство для PyTorch: {cfg.device}")

class PrioritizedReplayBuffer:
    """Приоритизированный буфер воспроизведения опыта с GPU оптимизациями"""
    
    def __init__(self, capacity, state_size, alpha=0.6, beta=0.4, beta_increment=0.001, use_gpu_storage=True):
        self.capacity = capacity
        self.state_size = state_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = cfg.device
        # Включаем GPU storage только если и флаг включен, и CUDA доступна, и конфиг-устройство — CUDA
        self.use_gpu_storage = bool(use_gpu_storage) and torch.cuda.is_available() and getattr(self.device, 'type', 'cpu') == 'cuda'
        
        # Инициализируем буферы на GPU или CPU с pinned memory
        if self.use_gpu_storage:
            # Полное хранение на GPU
            self.states = torch.zeros((capacity, state_size), dtype=torch.float32, device=self.device)
            self.next_states = torch.zeros((capacity, state_size), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros(capacity, dtype=torch.long, device=self.device)
            self.rewards = torch.zeros(capacity, dtype=torch.float32, device=self.device)
            self.dones = torch.zeros(capacity, dtype=torch.bool, device=self.device)
            self.gamma_ns = torch.ones(capacity, dtype=torch.float32, device=self.device)
            self.priorities = torch.ones(capacity, dtype=torch.float32, device=self.device)
        else:
            # Pinned memory на CPU имеет смысл только если доступна CUDA
            pin_flag = torch.cuda.is_available()
            self.states = torch.zeros((capacity, state_size), dtype=torch.float32, pin_memory=pin_flag)
            self.next_states = torch.zeros((capacity, state_size), dtype=torch.float32, pin_memory=pin_flag)
            self.actions = torch.zeros(capacity, dtype=torch.long, pin_memory=pin_flag)
            self.rewards = torch.zeros(capacity, dtype=torch.float32, pin_memory=pin_flag)
            self.dones = torch.zeros(capacity, dtype=torch.bool, pin_memory=pin_flag)
            self.gamma_ns = torch.ones(capacity, dtype=torch.float32, pin_memory=pin_flag)
            self.priorities = torch.ones(capacity, dtype=torch.float32, pin_memory=pin_flag)
        
        self.position = 0
        self.size = 0
        
        print(f"🚀 Replay Buffer: {'GPU storage' if self.use_gpu_storage else 'Pinned memory'} на {self.device}")
        
    def push(self, state, action, reward, next_state, done, gamma_n=1.0):
        # Конвертируем в тензоры если нужно
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        if not isinstance(action, torch.Tensor):
            action = torch.LongTensor([action])
        if not isinstance(reward, torch.Tensor):
            reward = torch.FloatTensor([reward])
        if not isinstance(done, torch.Tensor):
            done = torch.BoolTensor([done])
        if not isinstance(gamma_n, torch.Tensor):
            gamma_n = torch.FloatTensor([gamma_n])
        
        # Перемещаем на нужное устройство (только если хранилище на GPU)
        if self.use_gpu_storage:
            state = state.to(self.device, non_blocking=True)
            next_state = next_state.to(self.device, non_blocking=True)
            action = action.to(self.device, non_blocking=True)
            reward = reward.to(self.device, non_blocking=True)
            done = done.to(self.device, non_blocking=True)
            gamma_n = gamma_n.to(self.device, non_blocking=True)
        else:
            # Гарантируем CPU тензоры для CPU-хранилища
            if state.is_cuda:
                state = state.cpu()
            if next_state.is_cuda:
                next_state = next_state.cpu()
            if action.is_cuda:
                action = action.cpu()
            if reward.is_cuda:
                reward = reward.cpu()
            if done.is_cuda:
                done = done.cpu()
            if gamma_n.is_cuda:
                gamma_n = gamma_n.cpu()
        
        # Сохраняем данные
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.gamma_ns[self.position] = gamma_n
        
        # Устанавливаем максимальный приоритет для нового опыта
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_n_step(self, n_step_transitions):
        """
        Добавляет n-step transitions в replay buffer
        
        Args:
            n_step_transitions: список n-step transitions от environment
        """
        for transition in n_step_transitions:
            # Дополнительная проверка на None
            if (transition['state'] is not None and 
                transition['action'] is not None and 
                transition['reward'] is not None and 
                transition['next_state'] is not None):
                
                self.push(
                    state=transition['state'],
                    action=transition['action'],
                    reward=transition['reward'],
                    next_state=transition['next_state'],
                    done=transition['done'],
                    gamma_n=1.0  # gamma уже применен в n-step return
                )
    
    def sample(self, batch_size):
        if self.size == 0:
            return None, None, None, None, None, None, None, None
        
        # Выбираем индексы на основе приоритетов
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Вычисляем веса для importance sampling
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Убеждаемся, что weights находятся на правильном устройстве
        if self.use_gpu_storage:
            weights = weights.to(self.device, non_blocking=True)
        
        # Увеличиваем beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Получаем batch (уже на правильном устройстве)
        states = self.states[indices]
        next_states = self.next_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        gamma_ns = self.gamma_ns[indices]
        
        return states, actions, rewards, next_states, dones, gamma_ns, weights, indices
    
    def update_priorities(self, indices, priorities):
        # Убеждаемся, что priorities - это тензор
        if not isinstance(priorities, torch.Tensor):
            priorities = torch.FloatTensor(priorities)
        
        # Убеждаемся, что indices - это тензор
        if not isinstance(indices, torch.Tensor):
            indices = torch.LongTensor(indices)
        
        # Приводим к одному устройству
        if self.use_gpu_storage:
            priorities = priorities.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)
        else:
            priorities = priorities.cpu()
            indices = indices.cpu()
        
        # Обновляем приоритеты
        self.priorities[indices] = priorities
    
    def __len__(self):
        return self.size

# --- DQNSolver адаптированный под PyTorch с GPU оптимизациями ---
class DQNSolver:
    def __init__(self, observation_space, action_space, load=False):
        self.cfg      = vDqnConfig()
        self.n_step   = getattr(self.cfg, "n_step", 3)   # 3‑5 шагов
        self.n_queue  = deque(maxlen=self.n_step)                
        self.epsilon      = self.cfg.eps_start
        self.action_space = action_space
        
        # Используем Prioritized Replay Buffer если включен
        if self.cfg.prioritized:
            # Получаем размер состояния из observation_space
            state_size = observation_space if isinstance(observation_space, int) else observation_space
            use_gpu_storage = getattr(self.cfg, 'use_gpu_storage', True)
            
            self.memory = PrioritizedReplayBuffer(
                self.cfg.memory_size, 
                state_size, 
                self.cfg.alpha, 
                self.cfg.beta, 
                self.cfg.beta_increment,
                use_gpu_storage
            )
        else:
            self.memory = deque(maxlen=self.cfg.memory_size)
        
        self._eps_decay_rate = math.exp(math.log(self.cfg.eps_final / self.cfg.eps_start) / self.cfg.eps_decay_steps)

        # Создаем модели с поддержкой Rainbow компонентов
        if getattr(self.cfg, 'use_noisy_networks', True):
            # Используем Noisy Dueling DQN
            from agents.vdqn.dqnn import NoisyDuelingDQN
            self.model = NoisyDuelingDQN(
                observation_space, 
                action_space, 
                self.cfg.hidden_sizes,
                dropout_rate=self.cfg.dropout_rate,
                layer_norm=self.cfg.layer_norm,
                activation=self.cfg.activation,
                use_residual=self.cfg.use_residual_blocks,
                use_swiglu=self.cfg.use_swiglu_gate
            ).to(self.cfg.device)
            
            self.target_model = NoisyDuelingDQN(
                observation_space, 
                action_space, 
                self.cfg.hidden_sizes,
                dropout_rate=self.cfg.dropout_rate,
                layer_norm=self.cfg.layer_norm,
                activation=self.cfg.activation,
                use_residual=self.cfg.use_residual_blocks,
                use_swiglu=self.cfg.use_swiglu_gate
            ).to(self.cfg.device)
            
            print("🔀 Используем Noisy Dueling DQN для лучшего exploration")
        else:
            # Используем обычный DQN
            self.model = DQNN(
                observation_space, 
                action_space, 
                self.cfg.hidden_sizes,
                dropout_rate=self.cfg.dropout_rate,
                layer_norm=self.cfg.layer_norm,
                activation=self.cfg.activation,
                use_residual=self.cfg.use_residual_blocks,
                use_swiglu=self.cfg.use_swiglu_gate,
                dueling=self.cfg.dueling_dqn
            ).to(self.cfg.device)
            
            self.target_model = DQNN(
                observation_space, 
                action_space, 
                self.cfg.hidden_sizes,
                dropout_rate=self.cfg.dropout_rate,
                layer_norm=self.cfg.layer_norm,
                activation=self.cfg.activation,
                use_residual=self.cfg.use_residual_blocks,
                use_swiglu=self.cfg.use_swiglu_gate,
                dueling=self.cfg.dueling_dqn
            ).to(self.cfg.device)
        
        # Принудительно переносим модели на устройство конфига
        device = torch.device(self.cfg.device)
        self.model = self.model.to(device)
        self.target_model = self.target_model.to(device)
        print(f"✅ Модель и target-модель переведены на {device}")

        # 🚀 PyTorch 2.x Compile для максимального ускорения!
        if (getattr(self.cfg, 'use_torch_compile', True) and 
            not getattr(self.cfg, 'torch_compile_force_disable', False) and 
            hasattr(torch, 'compile')):
            try:
                print("🚀 Компилирую модель с torch.compile для максимального ускорения...")
                
                # Проверяем CUDA capability для выбора совместимого режима
                if self.cfg.device.type == 'cuda':
                    device_capability = torch.cuda.get_device_capability()
                    device_name = torch.cuda.get_device_name()
                    print(f"🔍 CUDA Capability: {device_capability[0]}.{device_capability[1]}")
                    print(f"🎯 GPU: {device_name}")
                    
                    # Специальная проверка для Tesla P100
                    if "Tesla P100" in device_name:
                        print("⚠️ Обнаружен Tesla P100 - отключаем torch.compile")
                        raise RuntimeError("Tesla P100 не поддерживает torch.compile")
                    
                    if device_capability[0] >= 7:  # Volta+ (V100, A100, H100, etc.)
                        compile_mode = 'max-autotune'
                        print("✅ Используем режим 'max-autotune' для современного GPU")
                    elif device_capability[0] >= 6:  # Pascal (GTX 1080, etc., но не P100)
                        if getattr(self.cfg, 'torch_compile_fallback', True):
                            compile_mode = 'default'
                            print("⚠️ GPU Pascal (не P100), используем режим 'default'")
                        else:
                            raise RuntimeError("GPU Pascal не поддерживает torch.compile в режиме max-autotune")
                    else:  # Maxwell и старше
                        if getattr(self.cfg, 'torch_compile_fallback', True):
                            compile_mode = 'default'
                            print("⚠️ GPU Maxwell или старше, используем режим 'default'")
                        else:
                            raise RuntimeError("GPU слишком старый для torch.compile")
                else:
                    compile_mode = 'default'
                    print("ℹ️ CPU режим, используем 'default'")
                
                self.model = torch.compile(self.model, mode=compile_mode)
                self.target_model = torch.compile(self.target_model, mode=compile_mode)
                print(f"✅ Модели скомпилированы успешно с режимом '{compile_mode}'!")
                
            except Exception as e:
                print(f"⚠️ torch.compile не удалось применить: {e}")
                print("📝 Модель будет работать без компиляции")
                
                # Автоматически отключаем torch.compile для этого запуска
                self.cfg.use_torch_compile = False
                self.cfg.torch_compile_force_disable = True
                print("🔄 torch.compile автоматически отключен для этого запуска")
        else:
            if not hasattr(torch, 'compile'):
                print("📝 PyTorch < 2.0, torch.compile недоступен")
            else:
                print("📝 torch.compile отключен в конфигурации")
        # Повторно гарантируем устройство после возможной компиляции
        device_post = next(self.model.parameters()).device
        if device_post != device:
            self.model = self.model.to(device)
        target_post = next(self.target_model.parameters()).device
        if target_post != device:
            self.target_model = self.target_model.to(device)
        
        # Копируем веса
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Оптимизатор: отдельные группы параметров для энкодера и головы
        try:
            encoder_params = None
            head_params = None
            encoder_lr_scale = float(getattr(self.cfg, 'encoder_lr_scale', 0.1))

            # Вариант 1: классический DQNN без dueling — есть _feature_extractor и head
            if hasattr(self.model, '_feature_extractor') and getattr(self.model, '_feature_extractor', None) is not None \
               and hasattr(self.model, 'head') and getattr(self.model, 'head', None) is not None:
                encoder_params = list(self.model._feature_extractor.parameters())
                # Остальные параметры как "голова"
                head_params = [p for p in self.model.parameters() if p not in encoder_params]

            # Вариант 2: dueling/noisy-dueling — энкодер внутри self.model.net.feature_extractor
            elif hasattr(self.model, 'net') and getattr(self.model, 'net', None) is not None \
                 and hasattr(self.model.net, 'feature_extractor') and getattr(self.model.net, 'feature_extractor', None) is not None:
                encoder_params = list(self.model.net.feature_extractor.parameters())
                head_params = [p for p in self.model.parameters() if p not in encoder_params]

            if encoder_params is not None and head_params is not None and len(encoder_params) > 0 and len(head_params) > 0:
                self.optimizer = optim.AdamW(
                    [
                        {"params": encoder_params, "lr": max(1e-8, float(self.cfg.lr) * encoder_lr_scale)},
                        {"params": head_params,    "lr": float(self.cfg.lr)},
                    ],
                    weight_decay=1e-4,
                    eps=1e-7,
                )
            else:
                # Фоллбек на единый LR, если не удалось разделить параметры
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.cfg.lr,
                    weight_decay=1e-4,
                    eps=1e-7,
                )
        except Exception:
            # Любая ошибка — безопасный фоллбек на единый LR
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

            # Включаем TF32 только на Ampere+ (compute capability >= 8.0)
            try:
                major, minor = torch.cuda.get_device_capability()
                if major >= 8:
                    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

            # Mixed precision training
            if self.cfg.use_amp and torch.cuda.is_available():
                # Новый API PyTorch 2.x
                self.scaler = torch.amp.GradScaler('cuda')
            else:
                self.scaler = None
        else:
            # На CPU scaler не используется, но атрибут должен существовать
            self.scaler = None
        
        # Автозагрузка старых весов отключена (контролируется внешним тренером)
        if load:
            print("🛑 Автозагрузка старых весов отключена — игнорируем load=True")
            
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
            out = self.model(state_tensor)
            q_values = out[0] if isinstance(out, tuple) else out
            
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
            
            # Проверяем на None (пустой буфер)
            if states is None:
                return False, None, None, None
        else:
            batch = random.sample(self.memory, self.cfg.batch_size)
            states, actions, rewards, next_states, dones, gamma_ns = zip(*batch)
            
            # Конвертируем в тензоры (перенос на устройство ниже, единообразно)
            states = torch.stack(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.stack(next_states)
            dones = torch.BoolTensor(dones)
            gamma_ns = torch.FloatTensor(gamma_ns)
            weights = torch.ones(self.cfg.batch_size)

        # Универсальный перенос батча на устройство модели
        model_device = next(self.model.parameters()).device
        def _move(t):
            return t.to(model_device, non_blocking=True) if t is not None else None
        states, actions, rewards, next_states, dones, gamma_ns, weights = map(
            _move, [states, actions, rewards, next_states, dones, gamma_ns, weights]
        )
        
        # Проверяем на NaN
        if torch.isnan(states).any() or torch.isnan(next_states).any():
            print("Warning: NaN detected in states, skipping batch")
            return False, None, None, None
        
        # Переводим модель в режим обучения
        self.model.train()
        
        # Вспомогательная функция на случай, если модель возвращает кортеж (logits, state)
        def _model_logits(mdl, x):
            out = mdl(x)
            if isinstance(out, tuple):
                return out[0]
            return out

        # Double DQN: используем основную сеть для выбора действий, целевую для оценки
        if self.cfg.double_dqn:
            with torch.no_grad():
                next_actions = _model_logits(self.model, next_states).argmax(dim=1)
                next_q_values = _model_logits(self.target_model, next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            with torch.no_grad():
                next_q_values = _model_logits(self.target_model, next_states).max(1)[0]
        
        # Вычисляем target Q-values
        target_q_values = rewards + (gamma_ns * next_q_values * ~dones)
        
        # Текущие Q-values
        current_q_values = _model_logits(self.model, states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
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
            # Убеждаемся, что priorities и indices находятся на одном устройстве
            priorities = (torch.abs(td_errors) + 1e-6).detach()
            if self.memory.use_gpu_storage:
                # Если используем GPU storage, оставляем priorities на GPU
                priorities = priorities.to(self.memory.device)
                indices = indices.to(self.memory.device)
            else:
                # Если используем CPU storage, переводим priorities на CPU
                priorities = priorities.cpu()
                indices = indices.cpu()
            
            self.memory.update_priorities(indices, priorities)
        
        # Обновляем scheduler
        self.scheduler.step()
        
        # Сбрасываем шум в Noisy Networks после обновления
        if hasattr(self.model, 'reset_noise'):
            self.model.reset_noise()
        if hasattr(self.target_model, 'reset_noise'):
            self.target_model.reset_noise()
        
        # Переводим модель обратно в режим оценки для ускорения inference
        self.model.eval()
        
        # Метрики
        if need_metrics:
            abs_q = current_q_values.abs().mean().item()
            q_gap = (target_q_values - current_q_values).abs().mean().item()
            return True, loss.item(), abs_q, q_gap
        
        return True, None, None, None

    def print_trade_stats(self, trades, failed_attempts: int | None = None):
        if not trades:
            return {"trades_count": 0, "winrate": 0.0, "avg_profit": 0.0, "avg_loss": 0.0}
        
        # Используем безопасный доступ к ключам, поддерживая оба варианта: "profit" и "roi"
        def get_profit(trade):
            return trade.get("profit", trade.get("roi", 0))
        
        profits = [get_profit(t) for t in trades if get_profit(t) > 0]
        losses = [get_profit(t) for t in trades if get_profit(t) < 0]
        
        winrate = len(profits) / len(trades) if trades else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_roi = np.mean([get_profit(t) for t in trades]) if trades else 0.0
        
        pl_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
        
        # Считаем плохие сделки
        bad_trades = [t for t in trades if abs(get_profit(t)) < 0.001]  # <0.1%
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
        
        suffix = ''
        if failed_attempts is not None:
            suffix = f", Failed train: {failed_attempts}"
        print(f"📊 Trades: {stats['trades_count']}, Winrate: {stats['winrate']*100:.2f}%, "
              f"Avg P: {stats['avg_profit']:.3f}, Avg L: {stats['avg_loss']:.3f}, "
              f"P/L ratio: {stats['pl_ratio']:.2f}, Bad trades: {stats['bad_trades_count']}{suffix}")
        
        return stats

    def save(self, normalization_stats: dict | None = None):
        """Сохраняет модель и replay buffer (полное сохранение)
        Args:
            normalization_stats: статистики нормализации env (единый препроцессинг)
        """
        # Обрабатываем torch.compile префикс при сохранении
        model_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        
        # Убираем префикс _orig_mod если он есть
        cleaned_model_state = {}
        cleaned_target_state = {}
        
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                cleaned_model_state[new_key] = value
            else:
                cleaned_model_state[key] = value
                
        for key, value in target_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                cleaned_target_state[new_key] = value
            else:
                cleaned_target_state[key] = value
        
        payload = {
            'model_state_dict': cleaned_model_state,
            'target_model_state_dict': cleaned_target_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'cfg': self.cfg
        }
        if normalization_stats is not None:
            try:
                payload['normalization_stats'] = normalization_stats
            except Exception:
                pass
        torch.save(payload, self.cfg.model_path)
        # Сохраняем энкодер отдельно
        encoder_state = {}
        extractor = None
        if hasattr(self.model, 'get_feature_extractor'):
            extractor = self.model.get_feature_extractor()
        if extractor is not None:
            encoder_state['encoder'] = extractor.state_dict()
        if hasattr(self.target_model, 'get_feature_extractor'):
            target_extractor = self.target_model.get_feature_extractor()
            if target_extractor is not None:
                encoder_state['target_encoder'] = target_extractor.state_dict()
        if encoder_state and getattr(self.cfg, 'encoder_path', None):
            encoder_path = Path(self.cfg.encoder_path)
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(encoder_state, encoder_path)
        
        # Сохраняем replay buffer
        with open(self.cfg.buffer_path, 'wb') as f:
            pickle.dump(self.memory, f, protocol=HIGHEST_PROTOCOL)
        
        print(f"✅ Модель сохранена в {self.cfg.model_path}")
        print(f"✅ Replay buffer сохранен в {self.cfg.buffer_path}")

    def load_model(self):
        """Загружает модель с проверкой совместимости архитектуры"""
        if os.path.exists(self.cfg.encoder_path):
            try:
                encoder_state = torch.load(self.cfg.encoder_path, map_location=self.cfg.device)
                if 'encoder' in encoder_state and hasattr(self.model, 'get_feature_extractor'):
                    extractor = self.model.get_feature_extractor()
                    if extractor is not None:
                        extractor.load_state_dict(encoder_state['encoder'])
                if 'target_encoder' in encoder_state and hasattr(self.target_model, 'get_feature_extractor'):
                    target_extractor = self.target_model.get_feature_extractor()
                    if target_extractor is not None:
                        target_extractor.load_state_dict(encoder_state['target_encoder'])
            except Exception as exc:
                print(f"⚠️ Не удалось загрузить энкодер: {exc}")

        if os.path.exists(self.cfg.model_path):
            # Явно логируем путь к модели и буферу перед загрузкой
            try:
                print(f"🧾 Пытаюсь загрузить модель из: {self.cfg.model_path}")
                if hasattr(self.cfg, 'buffer_path'):
                    print(f"🧾 Путь к replay buffer: {self.cfg.buffer_path}")
            except Exception:
                pass
            try:
                checkpoint = torch.load(self.cfg.model_path, map_location=self.cfg.device)
                
                # Проверяем совместимость архитектуры
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    
                    # Проверяем размеры первого слоя (учитываем torch.compile префикс)
                    first_layer_key = None
                    for key in model_state.keys():
                        if 'feature_layers.0.weight' in key:
                            first_layer_key = key
                            break
                    
                    if first_layer_key:
                        saved_input_size = model_state[first_layer_key].shape[1]

                        # Надёжно определяем размерность входа первого линейного слоя текущей модели
                        def _infer_first_linear_input_size(model) -> int | None:
                            try:
                                # Попытка №1: у моделей вида *Dueling* есть feature_layers (ModuleList)
                                if hasattr(model, 'feature_layers') and model.feature_layers:
                                    for layer in model.feature_layers:
                                        if isinstance(layer, nn.Linear):
                                            return layer.weight.shape[1]
                                # Попытка №2: у классической DQNN хранится в model.net (Sequential)
                                if hasattr(model, 'net') and hasattr(model.net, 'modules'):
                                    for layer in model.net.modules():
                                        if isinstance(layer, nn.Linear):
                                            return layer.weight.shape[1]
                            except Exception:
                                return None
                            return None

                        current_input_size = _infer_first_linear_input_size(self.model)

                        if current_input_size is not None and saved_input_size != current_input_size:
                            print(f"⚠️ Архитектура несовместима: сохраненная {saved_input_size}, текущая {current_input_size}")
                            print("🔄 Создаем новую модель с текущей архитектурой")
                            return
                    
                    # Обрабатываем torch.compile префикс
                    try:
                        # Пробуем загрузить как есть
                        # Строгая загрузка по умолчанию
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.target_model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Модель загружена без обработки префикса")
                    except Exception as compile_error:
                        # Если не получилось, пробуем обработать префикс _orig_mod
                        if "_orig_mod" in str(compile_error):
                            print("🔄 Обрабатываем torch.compile префикс...")
                            
                            # Проверяем, есть ли в сохраненной модели префикс _orig_mod
                            has_orig_mod = any(key.startswith('_orig_mod.') for key in checkpoint['model_state_dict'].keys())
                            
                            print(f"🔍 Анализ ключей модели:")
                            print(f"   • Ключи с _orig_mod: {has_orig_mod}")
                            print(f"   • Примеры ключей: {list(checkpoint['model_state_dict'].keys())[:3]}")
                            
                            if has_orig_mod:
                                # Убираем префикс _orig_mod из сохраненной модели
                                print("📝 Убираем префикс _orig_mod из сохраненной модели...")
                                adjusted_state_dict = {}
                                for key, value in checkpoint['model_state_dict'].items():
                                    if key.startswith('_orig_mod.'):
                                        new_key = key.replace('_orig_mod.', '')
                                        adjusted_state_dict[new_key] = value
                                    else:
                                        adjusted_state_dict[key] = value
                                
                                self.model.load_state_dict(adjusted_state_dict)
                                self.target_model.load_state_dict(adjusted_state_dict)
                                print("✅ Модель загружена с обработкой префикса")
                            else:
                                # Добавляем префикс _orig_mod к сохраненной модели
                                print("📝 Добавляем префикс _orig_mod к сохраненной модели...")
                                adjusted_state_dict = {}
                                for key, value in checkpoint['model_state_dict'].items():
                                    new_key = f"_orig_mod.{key}"
                                    adjusted_state_dict[new_key] = value
                            
                                self.model.load_state_dict(adjusted_state_dict)
                                self.target_model.load_state_dict(adjusted_state_dict)
                                print("✅ Модель загружена с обработкой префикса")
                        else:
                            print(f"❌ Ошибка не связана с torch.compile префиксом при загрузке {self.cfg.model_path}: {compile_error}")
                            raise compile_error
                    
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
                print(f"❌ Ошибка при загрузке модели из {self.cfg.model_path}: {e}")
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
        """Сохраняет только модель (без replay buffer)"""
        # Обрабатываем torch.compile префикс при сохранении
        model_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        
        # Убираем префикс _orig_mod если он есть
        cleaned_model_state = {}
        cleaned_target_state = {}
        
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                cleaned_model_state[new_key] = value
            else:
                cleaned_model_state[key] = value
                
        for key, value in target_state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                cleaned_target_state[new_key] = value
            else:
                cleaned_target_state[key] = value
        
        base_payload = {
            'model_state_dict': cleaned_model_state,
            'target_model_state_dict': cleaned_target_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'cfg': self.cfg
        }

        torch.save(base_payload, self.cfg.model_path)

        encoder_state = {}
        extractor = None
        if hasattr(self.model, 'get_feature_extractor'):
            extractor = self.model.get_feature_extractor()
        if extractor is not None:
            encoder_state['encoder'] = extractor.state_dict()
        if hasattr(self.target_model, 'get_feature_extractor'):
            target_extractor = self.target_model.get_feature_extractor()
            if target_extractor is not None:
                encoder_state['target_encoder'] = target_extractor.state_dict()
        if encoder_state and getattr(self.cfg, 'encoder_path', None):
            torch.save(encoder_state, self.cfg.encoder_path)
        
        print(f"✅ Модель сохранена в {self.cfg.model_path}")
        
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
        




