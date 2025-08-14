import torch
import numpy as np
from agents.vdqn.dqnsolver import DQNSolver
from envs.dqn_model.gym.crypto_trading_env import CryptoTradingEnv
import wandb
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.gutils import get_nan_stats, log_csv, setup_logger, setup_wandb

cfg = vDqnConfig()

def train_model(dfs: dict, load_previous: bool = False, episodes: int = 200):
    """
    Обучает улучшенную модель DQN для торговли криптовалютой с GPU оптимизациями.

    Args:
        dfs (dict): Словарь с Pandas DataFrames для разных таймфреймов (df_5min, df_15min, df_1h).
        load_previous (bool): Загружать ли ранее сохраненную модель.
        episodes (int): Количество эпизодов для обучения.
    Returns:
        str: Сообщение о завершении обучения.
    """

    import time    

    all_trades = []
    best_winrate = 0.0
    patience_counter = 0
    patience_limit = 500  # Early stopping после 500 эпизодов без улучшений (увеличено для более длительного обучения)

    wandb_run = None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            
        
        if device.type == 'cuda':
            # GPU оптимизации
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Очищаем GPU кэш
            torch.cuda.empty_cache()
            
            # Проверяем доступную память
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Available Memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
            
        cfg.device = device 
        
        # Создаем окружение
        env = CryptoTradingEnv(dfs=dfs) 

        observation_space_dim = env.observation_space.shape[0]
        action_space = env.action_space.n

        logger = setup_logger("rl")
        if getattr(cfg, "use_wandb", False):
            wandb_run, _ = setup_wandb(cfg)
                        
        global_step = 0
        last_time = time.perf_counter()
        _next_tick = {}

        def tick(label: str):
            nonlocal last_time, global_step, _next_tick
            now = time.perf_counter()
            dt_ms = (now - last_time) * 1e3
            last_time = now

            if (dt_ms >= cfg.tick_slow_ms) or (global_step >= _next_tick.get(label, -1)):
                logger.info("[T] %s: %.1f ms", label, dt_ms)
                _next_tick[label] = global_step + cfg.tick_every

        dqn_solver = DQNSolver(observation_space_dim, action_space, load=load_previous)
                
        logger.info("Training started: torch=%s cuda=%s device=%s",
            torch.__version__, torch.version.cuda, device)

        successful_episodes = 0        
        episode_rewards = []
        episode_profits = []
        episode_winrates = []

        # Предварительная загрузка данных в GPU память
        if device.type == 'cuda':
            # Создаем dummy tensor для разогрева GPU
            dummy_tensor = torch.randn(1000, observation_space_dim).to(device)
            _ = dqn_solver.model(dummy_tensor)
            del dummy_tensor
            torch.cuda.empty_cache()

        for episode in range(episodes):
            # Переводим модель в режим обучения только когда нужно
            state = env.reset()
            
            # Проверяем состояние на NaN
            if np.isnan(state).any():
                state = np.nan_to_num(state, nan=0.0)
                logger.warning("NaN detected in initial state, replaced with zeros")
            
            grad_steps = 0
            episode_reward = 0
            tick(f"{episode} episode [{cfg.device}]")
            
            while True:                                                
                env.epsilon = dqn_solver.epsilon
                                  
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                
                # Проверяем next_state на NaN
                if np.isnan(state_next).any():
                    state_next = np.nan_to_num(state_next, nan=0.0)
                    logger.warning("NaN detected in next_state, replaced with zeros")
                
                # Сохраняем переход в replay buffer
                dqn_solver.store_transition(state, action, reward, state_next, terminal)
                
                # Обновляем состояние
                state = state_next
                episode_reward += reward
                global_step += 1
                
                # Обучаем модель чаще для ускорения
                if global_step % cfg.soft_update_every == 0 and len(dqn_solver.memory) >= cfg.batch_size:
                    success, loss, abs_q, q_gap = dqn_solver.experience_replay(need_metrics=True)
                    if success:
                        grad_steps += 1
                        
                        # Обновляем target network чаще
                        if global_step % cfg.target_update_freq == 0:
                            dqn_solver.update_target_model()
                
                if terminal:
                    break
            
            # Обновляем epsilon
            dqn_solver.epsilon = max(cfg.eps_final, dqn_solver.epsilon * dqn_solver._eps_decay_rate)
            
            # Собираем статистику эпизода
            if hasattr(env, 'trades') and env.trades:
                all_trades.extend(env.trades)
                
                # Вычисляем winrate для эпизода
                profitable_trades = [t for t in env.trades if t.get('profit', 0) > 0]
                episode_winrate = len(profitable_trades) / len(env.trades) if env.trades else 0
                episode_winrates.append(episode_winrate)
                
                # Проверяем на улучшение
                if episode_winrate > best_winrate:
                    best_winrate = episode_winrate
                    patience_counter = 0
                    
                    # Сохраняем лучшую модель
                    dqn_solver.save_model()
                    logger.info("[INFO] New best winrate: %.3f, saving model", best_winrate)
                else:
                    patience_counter += 1
            else:
                # Если нет сделок, добавляем 0 winrate
                episode_winrates.append(0.0)
                patience_counter += 1
            
            # Логируем прогресс
            if episode % 10 == 0:
                avg_winrate = np.mean(episode_winrates[-10:]) if episode_winrates else 0
                logger.info(f"[INFO] Episode {episode}/{episodes}, Avg Winrate: {avg_winrate:.3f}, Epsilon: {dqn_solver.epsilon:.4f}")
                
                # Очищаем GPU память каждые 10 эпизодов
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Early stopping
            if patience_counter >= patience_limit:
                logger.info(f"[INFO] Early stopping triggered after {episode} episodes")
                break

        # Финальная статистика
        stats_all = dqn_solver.print_trade_stats(all_trades)
        log_csv(cfg.csv_metrics_path, {"scope":"cumulative", "episode": episodes, **stats_all})
        
        if cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        dqn_solver.save()
        print(stats_all)
        print("Финальная модель сохранена.")
        
        # Анализ трендов
        if len(episode_winrates) > 10:
            recent_winrate = np.mean(episode_winrates[-10:])
            overall_winrate = np.mean(episode_winrates)
            print(f"📈 Winrate тренд: последние 10 эпизодов: {recent_winrate:.3f}, общий: {overall_winrate:.3f}")
            
            if recent_winrate > overall_winrate:
                print("✅ Модель улучшается!")
            else:
                print("⚠️ Модель может переобучаться")
        
        return "Обучение завершено"    
    finally:
        # Закрываем wandb
        if wandb_run is not None:
            wandb_run.finish()