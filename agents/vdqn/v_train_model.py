import torch
from agents.vdqn.dqnsolver import DQNSolver
from envs.dqn_model.gym.crypto_trading_env import CryptoTradingEnv
import wandb
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.gutils import setup_wandb

cfg = vDqnConfig()

def train_model(dfs: dict, load_previous: bool = False, episodes: int = 10000):
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

    import time    

    all_trades = []

    wandb_run = None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')            
        
        if device.type == 'cuda':
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True  # <--- ответ на твой п.4 (включать здесь)
            
        cfg.device = device 
        
        # Теперь CryptoTradingEnv принимает словарь с DataFrame'ами
        env = CryptoTradingEnv(dfs=dfs) 

        observation_space_dim = env.observation_space.shape[0]
        action_space = env.action_space.n

        wandb_run, logger = setup_wandb(cfg)
        
        last = time.perf_counter()    
        def tick(label: str):
            nonlocal last, global_step
            now = time.perf_counter()
            # печатаем редко, чтобы не заспамить логи
            if global_step % 200 == 0:
                logger.info(f"[T] {label}: {(now - last)*1e3:.1f} ms")   
            last = now

        dqn_solver = DQNSolver(observation_space_dim, action_space, load=load_previous)
                
        logger.info("Training started: torch=%s cuda=%s device=%s",
            torch.__version__, torch.version.cuda, device)

        global_step = 0
        successful_episodes = 0        

        for episode in range(episodes):
            # Переводим модель в режим обучения
            dqn_solver.model.train() 
            state = env.reset() # env.reset() теперь возвращает начальное состояние    

            grad_steps   = 0

            while True:                
                
                env.epsilon = dqn_solver.epsilon
                                  
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                tick("env.step")
                
                dqn_solver.remember(state, action, reward, state_next, terminal)
                state = state_next       
                
                # сколько градиентных шагов на один env.step
                train_repeats = getattr(dqn_solver.cfg, "train_repeats", 2)   # начни с 2, потом 4
                need_metrics = (global_step % 100 == 0)  # метрики считаем не на каждом шаге

                did_step = False
                td_loss = abs_q = q_gap = None

                for i in range(train_repeats):
                    _did, _loss, _absq, _qgap = dqn_solver.experience_replay(
                        need_metrics = need_metrics and (i == train_repeats - 1)  # метрики один раз
                    )  
                    if _did:
                        did_step = True
                        td_loss, abs_q, q_gap = _loss, _absq, _qgap                                 
                   
                tick(f"replay x{train_repeats}")   
                
                if global_step % 50 == 0:                    
                    wandb.log({
                        "step":          global_step,
                        "episode":       episode + 1,
                        "reward":        reward,
                        "cumulative_reward": env.cumulative_reward,
                        "buy_attempts":  info.get('buy_attempts'),
                        "total_profit":  info.get('total_profit'),
                        "roi_block":     info.get('roi_block'),
                        "penalty":       info.get('penalty'),
                        "vol_block":     info.get('vol_block'),
                        "volatility":            info.get('volatility'),
                        "volatility_threshold":  info.get('volatility_threshold'),
                    })                    
                    tick("wandb.log 50")
                
                # --------------- лог каждые 100 grad‑шагов (только если был grad) ---
                if did_step and (global_step % 100 == 0):
                    wandb.log({
                        "step":     global_step,
                        "episode":  episode + 1,
                        "td_loss":  float(td_loss),   # или td_loss.item()
                        "abs_Q":    float(abs_q),     # или abs_q.item()
                        "q_gap":    float(q_gap),     # или q_gap.item()
                    })
                    tick("wandb.log 100")
                                      

                if did_step:
                    
                    dqn_solver.epsilon = max(dqn_solver.cfg.eps_final, dqn_solver.epsilon * dqn_solver._eps_decay_rate)
                    
                    grad_steps += 1
                    if grad_steps % cfg.soft_update_every  == 0:
                        dqn_solver.soft_update(tau=cfg.soft_tau)      
                        
                    if global_step % cfg.target_update_freq == 0:     # hard‑update
                        dqn_solver.update_target_model()                                                                                                  

                if global_step % 200 == 0:   
                    logger.info("[DBG] step=%d | replay_mem=%d | did_step=%s", global_step, len(dqn_solver.memory), did_step)                                                

                global_step += 1        

                if terminal:                      
                                                    
                    if info.get("total_profit", 0) > 0:
                        successful_episodes += 1
                                    
                    if env.can_log:                                           
                        print(f"{episode+1}/{episodes} завершен. "
                            f"Прибыль: {info.get('total_profit', 0):.2f}, "
                            f"Баланс: {info.get('current_balance', env.balance):.2f}, "
                            f"BTC: {info.get('crypto_held', 0):.4f}, "
                            f"sum reward: {env.cumulative_reward:.2f}, "
                            f"ε: {dqn_solver.epsilon:.4f}, "
                            f"Succ_epsоd: {successful_episodes:.2f}, "
                            f"BUY attempts={env.buy_attempts}, "
                            f"reject VOL={env.buy_rejected_vol}, "
                            f"reject ROI={env.buy_rejected_roi}")
                    
                        stats = dqn_solver.print_trade_stats(env.trades)
                        
                        
                    durations = [t["duration"] for t in env.trades]
                    if durations:
                        wandb.log({
                            "hold_time_hist": wandb.Histogram(durations),
                            "episode": episode + 1
                        })                            
                        
                    all_trades.extend(env.trades)                     
                            
                    break
                                        
            # --- УБРАТЬ ЖЁСТКИЙ СБРОС ---
            # if (episode + 1) % target_update_frequency == 0:
            #     dqn_solver.update_target_model()
            #     print(f"Целевая сеть обновлена после эпизода {episode+1}")

            # Сохранение модели (частота сохранения)
            if (episode + 1) % 100 == 0: 
                dqn_solver.save()
                print(f"Модель и replay buffer сохранены после эпизода {episode+1}")

        stats_all = dqn_solver.print_trade_stats(all_trades)
        wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        dqn_solver.save()
        print("Финальная модель сохранена.")
        return "Обучение завершено"    
    finally:
        # ---------- гарантированно закрываем run -----
        if wandb_run is not None:
            wandb_run.finish()