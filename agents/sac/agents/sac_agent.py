"""Основной SAC агент."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch import nn, optim # type: ignore

from .config import SacConfig
from .networks import SacCategoricalActor, SacDoubleCritic
from .replay_buffer import SacReplayBuffer


@dataclass
class SacAgentState:
    actor: nn.Module
    critic: nn.Module
    target_critic: nn.Module
    log_alpha: torch.Tensor
    alpha_optimizer: optim.Optimizer


class SacAgent:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Optional[SacConfig] = None) -> None:
        self.cfg = cfg or SacConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        print(f"🔍 [SacAgent] Инициализация: observation_dim={observation_dim}, action_dim={action_dim}")

        # Очищаем кэш CUDA перед созданием буфера
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 [SacAgent] Очищен кэш CUDA перед созданием буфера")
        
        # Инициализация актора и критика
        self.actor = SacCategoricalActor(
            obs_dim=observation_dim,
            action_dim=action_dim,
            cfg=self.cfg,
        ).to(self.device)
        
        self.critic = SacDoubleCritic(
            obs_dim=observation_dim,
            action_dim=action_dim,
            cfg=self.cfg,
        ).to(self.device)
        
        self.target_critic = SacDoubleCritic(
            obs_dim=observation_dim,
            action_dim=action_dim,
            cfg=self.cfg,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_entropy = -torch.log(torch.tensor(1.0 / action_dim, device=self.device)) * self.cfg.target_entropy_scale

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.cfg.lr_critic)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=self.cfg.lr_alpha)

        # Инициализация GradScaler для AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        # Буфер воспроизведения создаём на CPU для экономии GPU памяти
        # Можно включить обратно на GPU, если памяти достаточно
        buffer_device = torch.device("cpu")
        print(f"🔧 Буфер воспроизведения создан на {buffer_device}")

        self.replay_buffer = SacReplayBuffer(
            capacity=self.cfg.memory_size,
            state_dim=observation_dim,
            device=buffer_device,
            use_gpu_storage=False,  # Отключаем GPU хранение для экономии памяти
        )

        self.total_steps = 0

    # === Action selection ===
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs = obs.to(self.device)
        if deterministic:
            with torch.no_grad():
                logits = self.actor.forward(obs)
                action = torch.argmax(logits, dim=-1, keepdim=True)
                return action.float()
        with torch.no_grad():
            action, _ = self.actor.sample(obs)
        return action

    # === Training ===
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Проверка входного батча на NaN и inf
        for k, v in batch.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                logging.warning(f"Обнаружены NaN или Inf в {k} батча. Пропускаем обновление.")
                # Очищаем буфер при обнаружении NaN в батче
                if hasattr(self, 'clear_buffer_on_nan') and self.clear_buffer_on_nan:
                    self.replay_buffer.clear()
                    logging.warning("🔄 Буфер очищен из-за NaN в батче")
                return {}

        # Ленивая инициализация счётчика шагов
        if not hasattr(self, "update_steps"):
            self.update_steps = 0

        obs = batch["obs"].float().to(self.device)
        actions = batch["actions"].long().to(self.device).view(-1, 1)
        rewards = batch["rewards"].float().to(self.device).view(-1, 1)
        next_obs = batch["next_obs"].float().to(self.device)
        dones = batch["dones"].float().to(self.device).view(-1, 1)

                # Агрессивная очистка нечисловых значений во входах
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e5, neginf=-1e5)
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=1e5, neginf=-1e5)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1e3, neginf=-1e3)

        # Дополнительная проверка на экстремальные значения
        if torch.abs(obs).max() > 1e6 or torch.abs(next_obs).max() > 1e6 or torch.abs(rewards).max() > 1e4:
            logging.warning("⚠️ [SacAgent] Обнаружены экстремальные значения во входах. Очищаем буфер.")
            self.replay_buffer.clear()
            return {}

        with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
            alpha = self.log_alpha.exp()

            # Проверки на NaN в alpha и промежуточных расчетах
            if not torch.isfinite(alpha):
                logging.warning("⚠️ [SacAgent] alpha не конечен (NaN/Inf). Пропускаем обновление.")
                return {}

            with torch.no_grad():
                next_dist = self.actor.forward(next_obs)
                next_probs = next_dist.probs
                next_log_probs = next_dist.logits.log_softmax(dim=-1)

                # Проверка на NaN в next_probs и next_log_probs
                if not torch.isfinite(next_probs).all() or not torch.isfinite(next_log_probs).all():
                    logging.warning("⚠️ [SacAgent] next_probs или next_log_probs не конечны. Пропускаем обновление.")
                    return {}

                target_q1_all, target_q2_all = self.target_critic(next_obs)
                min_next_q = torch.min(target_q1_all, target_q2_all)

                # Детальный дебаг target_q
                if not torch.isfinite(target_q1_all).all() or not torch.isfinite(target_q2_all).all():
                    logging.warning(f"⚠️ [SacAgent] target_q1_all содержит NaN: {torch.isnan(target_q1_all).sum().item()}/{target_q1_all.numel()}")
                    logging.warning(f"⚠️ [SacAgent] target_q2_all содержит NaN: {torch.isnan(target_q2_all).sum().item()}/{target_q2_all.numel()}")
                    # Очищаем буфер при проблемах с target_critic
                    self.replay_buffer.clear()
                    logging.warning("🔄 Буфер очищен из-за NaN в target_critic")
                    return {}

                target_q = (next_probs * (min_next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)

                # Детальный дебаг target_q
                if not torch.isfinite(target_q).all():
                    nan_count = torch.isnan(target_q).sum().item()
                    logging.warning(f"⚠️ [SacAgent] target_q содержит {nan_count}/{target_q.numel()} NaN")
                    if nan_count > 0:
                        # Попробуем понять откуда NaN
                        finite_mask = torch.isfinite(target_q).squeeze()
                        if finite_mask.any():
                            logging.warning(f"⚠️ [SacAgent] Среднее конечное target_q: {target_q[finite_mask].mean().item():.6f}")
                        # Очищаем буфер
                        self.replay_buffer.clear()
                        logging.warning("🔄 Буфер очищен из-за NaN в target_q")
                    return {}

                target_value = rewards + (1 - dones) * self.cfg.gamma * target_q

                # Проверка на NaN в target_value
                if not torch.isfinite(target_value).all():
                    logging.warning("⚠️ [SacAgent] target_value не конечен. Пропускаем обновление.")
                    return {}

            current_q1_all, current_q2_all = self.critic(obs)
            actions_indices = actions.long().squeeze(-1)
            current_q1 = current_q1_all.gather(1, actions_indices.unsqueeze(-1))
            current_q2 = current_q2_all.gather(1, actions_indices.unsqueeze(-1))

            # Проверка на NaN в current_q1 и current_q2
            if not torch.isfinite(current_q1).all() or not torch.isfinite(current_q2).all():
                logging.warning("⚠️ [SacAgent] current_q не конечны. Пропускаем обновление.")
                return {}

            critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        # Обновляем критика (проверки на NaN/Inf)
        self.critic_optimizer.zero_grad()
        if not torch.isfinite(critic_loss):
            logging.warning("⚠️ [SacAgent] critic_loss не конечен (NaN/Inf). Пропускаем шаг обновления критика.")
            return {}
        self.scaler.scale(critic_loss).backward()
        # Unscale перед проверками и клиппингом
        self.scaler.unscale_(self.critic_optimizer)
        # Проверка градиентов критика на NaN/Inf
        bad_grad_critic = any(
            (p.grad is not None) and (not torch.all(torch.isfinite(p.grad))) for p in self.critic.parameters()
        )
        if bad_grad_critic:
            self.critic_optimizer.zero_grad(set_to_none=True)
            # Важно: синхронизируем GradScaler, иначе следующий unscale_ вызовет ошибку
            self.scaler.update()
            logging.warning("⚠️ [SacAgent] NaN/Inf в градиентах критика. Шаг пропущен.")
            return {}
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        # Мониторинг градиентов критика (редко)
        try:
            max_critic_grad = max(torch.max(p.grad).item() for p in self.critic.parameters() if p.grad is not None)
        except ValueError:
            max_critic_grad = float("nan")
        if (self.update_steps % 200 == 0) or (not torch.isfinite(torch.tensor(max_critic_grad))):
            logging.warning(f"📉 [SacAgent] Максимальный градиент критика: {max_critic_grad:.4f}")

        for param in self.critic.parameters():
            param.requires_grad = False

        # Политика: ожидаем распределение, вычисляем энтропию и потери
        dist = self.actor.forward(obs)
        probs = dist.probs
        log_probs = dist.logits.log_softmax(dim=-1)

        # Проверка на NaN в probs и log_probs
        if not torch.isfinite(probs).all() or not torch.isfinite(log_probs).all():
            for param in self.critic.parameters():
                param.requires_grad = True
            logging.warning("⚠️ [SacAgent] probs или log_probs не конечны. Пропускаем обновление актора.")
            return {}

        q1_all, q2_all = self.critic(obs)
        min_q_all = torch.min(q1_all, q2_all)

        # Проверка на NaN в q1_all и q2_all
        if not torch.isfinite(q1_all).all() or not torch.isfinite(q2_all).all():
            for param in self.critic.parameters():
                param.requires_grad = True
            logging.warning("⚠️ [SacAgent] q_values не конечны. Пропускаем обновление актора.")
            return {}

        actor_loss = (alpha * (probs * log_probs).sum(dim=-1, keepdim=True) - (probs * min_q_all).sum(dim=-1, keepdim=True)).mean()

        # Проверка на NaN в actor_loss
        if not torch.isfinite(actor_loss):
            for param in self.critic.parameters():
                param.requires_grad = True
            logging.warning("⚠️ [SacAgent] actor_loss не конечен (NaN/Inf). Пропускаем шаг обновления актора.")
            return {}

        # Обновляем актора (проверки на NaN/Inf)
        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        bad_grad_actor = any(
            (p.grad is not None) and (not torch.all(torch.isfinite(p.grad))) for p in self.actor.parameters()
        )
        if bad_grad_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            for param in self.critic.parameters():
                param.requires_grad = True
            # Важно: синхронизируем GradScaler, иначе следующий unscale_ вызовет ошибку
            self.scaler.update()
            logging.warning("⚠️ [SacAgent] NaN/Inf в градиентах актора. Шаг пропущен.")
            return {}
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()

        # Мониторинг градиентов актора (редко)
        try:
            max_actor_grad = max(torch.max(p.grad).item() for p in self.actor.parameters() if p.grad is not None)
        except ValueError:
            max_actor_grad = float("nan")
        if (self.update_steps % 200 == 0) or (not torch.isfinite(torch.tensor(max_actor_grad))):
            logging.warning(f"📉 [SacAgent] Максимальный градиент актора: {max_actor_grad:.4f}")

        for param in self.critic.parameters():
            param.requires_grad = True

        # Обновление температуры (alpha)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        if torch.isfinite(alpha_loss):
            self.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()
        else:
            logging.warning("⚠️ [SacAgent] alpha_loss не конечен (NaN/Inf). Пропускаем шаг обновления alpha.")

        # Обновление целевых сетей
        self._soft_update(self.critic, self.target_critic)

        self.update_steps += 1

        return {
            "critic_loss": critic_loss.item() if critic_loss is not None else float('nan'),
            "actor_loss": actor_loss.item() if actor_loss is not None else float('nan'),
            "alpha_loss": alpha_loss.item() if torch.isfinite(alpha_loss) else float('nan'),
            "alpha": alpha.item() if alpha is not None else float('nan'),
            "max_actor_grad": max_actor_grad,
            "max_critic_grad": max_critic_grad,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    # === Replay buffer integration ===
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.replay_buffer.add_transition(state, action, reward, next_state, done)
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        return len(self.replay_buffer) >= self.cfg.start_learning_after

    def sample_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        batch = self.replay_buffer.sample_batch(self.cfg.batch_size)
        if batch[0] is None:
            return None
        states, actions, rewards, next_states, dones, _, weights, _ = batch

        # Переносим данные на GPU устройство, если буфер на CPU
        if self.device.type == "cuda":
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)

        return {
            "obs": states,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_states,
            "dones": dones.float(),
            "weights": weights,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha.detach(),
            },
            path,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_()


