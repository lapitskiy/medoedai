import os
import sys
import logging
import numpy as np
import torch
import wandb
import time
from typing import Dict, List, Optional
import pickle
from pickle import HIGHEST_PROTOCOL

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.vdqn.dqnsolver import DQNSolver
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from envs.dqn_model.gym.gutils import log_csv

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_training(dfs: Dict) -> Dict:
    """
    Подготавливает данные для тренировки, конвертируя DataFrame в numpy массивы
    
    Args:
        dfs: словарь с DataFrame для разных таймфреймов
        
    Returns:
        Dict: словарь с numpy массивами для разных таймфреймов
    """
    print(f"📊 Подготавливаю данные для тренировки")
    
    # Проверяем наличие необходимых данных
    required_keys = ['df_5min', 'df_15min', 'df_1h']
    for key in required_keys:
        if key not in dfs:
            raise ValueError(f"Отсутствует {key} в переданных данных")
        if dfs[key] is None or dfs[key].empty:
            raise ValueError(f"{key} пустой или None")
    
    print(f"✅ Данные готовы: 5min={len(dfs['df_5min'])}, 15min={len(dfs['df_15min'])}, 1h={len(dfs['df_1h'])}")
    
    return dfs

def train_model_optimized(
    dfs: Dict,
    cfg: Optional[vDqnConfig] = None,
    episodes: int = 10,
    patience_limit: int = 50,
    use_wandb: bool = False
) -> str:
    """
    Оптимизированная функция тренировки модели без pandas в hot-path
    
    Args:
        dfs: словарь с DataFrame для разных таймфреймов (df_5min, df_15min, df_1h)
        cfg: конфигурация модели
        episodes: количество эпизодов для тренировки
        patience_limit: лимит терпения для early stopping
        use_wandb: использовать ли Weights & Biases
        
    Returns:
        str: сообщение о завершении тренировки
    """
    
    # Инициализация wandb
    wandb_run = None
    if use_wandb:
        try:
            run_name = getattr(cfg, 'run_name', 'default') if cfg else 'default'
            config_dict = cfg.__dict__ if cfg else {}
            
            wandb_run = wandb.init(
                project="medoedai-optimized",
                name=f"vDQN-optimized-{run_name}",
                config=config_dict
            )
        except Exception as e:
            logger.warning(f"Не удалось инициализировать wandb: {e}")
            use_wandb = False
    
    try:
        # Проверяем и создаем конфигурацию по умолчанию
        if cfg is None:
            cfg = vDqnConfig()
            print("⚠️ Конфигурация не передана, использую конфигурацию по умолчанию")
        
        # Подготавливаем данные
        dfs = prepare_data_for_training(dfs)
        
        # Создаем окружение
        env = CryptoTradingEnvOptimized(
            dfs=dfs,
            cfg=cfg,
            lookback_window=20,
            indicators_config=None  # Используем дефолтную конфигурацию
        )
        
        # Начинаем отсчет времени тренировки
        training_start_time = time.time()
        
        # Проверяем, что окружение правильно инициализировано
        if not hasattr(env, 'observation_space_shape'):
            raise ValueError("Окружение не имеет observation_space_shape")
        
        print(f"✅ Окружение создано, размер состояния: {env.observation_space_shape}")
        
        # Создаем DQN solver
        print(f"🚀 Создаю DQN solver")
        
        dqn_solver = DQNSolver(
            observation_space=env.observation_space_shape,
            action_space=env.action_space.n
        )
        
        # 🚀 Дополнительная оптимизация PyTorch 2.x
        if torch.cuda.is_available():
            # Включаем cudnn benchmark для максимального ускорения
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Включаем TF32 для ускорения на Ampere+ GPU
            if hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                
            print("🚀 CUDA оптимизации включены: cudnn.benchmark, TF32")
        
        # Загружаем модель если есть
        dqn_solver.load_model()
        
        # Переменные для отслеживания прогресса
        all_trades = []
        episode_winrates = []
        best_winrate = 0.0
        patience_counter = 0
        global_step = 0
        grad_steps = 0
        
        print(f"🎯 Начинаю тренировку на {episodes} эпизодов")
        print(f"📈 Размер состояния: {env.observation_space_shape}")
        print(f"🎮 Размер действий: {env.action_space.n}")
        
        # Основной цикл тренировки
        for episode in range(episodes):         
            state = env.reset()            
            # Убеждаемся, что state является numpy массивом
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Проверяем размер состояния
            if episode == 0:
                print(f"🔍 Первое состояние: тип={type(state)}, размер={state.shape if hasattr(state, 'shape') else len(state)}")
            
            episode_reward = 0
            print(f"  🎯 Эпизод {episode} начат, reward={episode_reward}")
            
            # Эпизод
            step_count = 0
            while True:
                step_count += 1
                # Показываем прогресс каждые 100 шагов для ускорения
                if step_count % 10000 == 0:
                    print(f"    🔄 Step {step_count} в эпизоде {episode}")
                
                env.epsilon = dqn_solver.epsilon
                
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                
                # Проверяем next_state на NaN
                if isinstance(state_next, (list, tuple)):
                    state_next = np.array(state_next, dtype=np.float32)
                elif not isinstance(state_next, np.ndarray):
                    state_next = np.array(state_next, dtype=np.float32)
                
                # Безопасная проверка на NaN
                try:
                    if np.isnan(state_next).any():
                        state_next = np.nan_to_num(state_next, nan=0.0)
                except (TypeError, ValueError):
                    # Если не можем проверить на NaN, преобразуем в numpy и попробуем снова (без спама в лог)
                    state_next = np.array(state_next, dtype=np.float32)
                    if np.isnan(state_next).any():
                        state_next = np.nan_to_num(state_next, nan=0.0)
                
                # Сохраняем переход в replay buffer
                dqn_solver.store_transition(state, action, reward, state_next, terminal)
                
                # Обновляем состояние
                state = state_next
                
                # Убеждаемся, что обновленный state является numpy массивом
                if isinstance(state, (list, tuple)):
                    state = np.array(state, dtype=np.float32)
                elif not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                
                episode_reward += reward
                global_step += 1
                
                # Обучаем модель реже для ускорения (как было раньше)
                soft_update_every = getattr(cfg, 'soft_update_every', 100)  # По умолчанию каждые 100 шагов
                batch_size = getattr(cfg, 'batch_size', 64)  # По умолчанию размер батча 64
                target_update_freq = getattr(cfg, 'target_update_freq', 1000)  # По умолчанию каждые 1000 шагов
                
                if global_step % soft_update_every == 0 and len(dqn_solver.memory) >= batch_size:                    
                    success, loss, abs_q, q_gap = dqn_solver.experience_replay(need_metrics=True)
                    if success:
                        grad_steps += 1                        
                        
                        # Обновляем target network чаще
                        if global_step % target_update_freq == 0:
                            dqn_solver.update_target_model()
                    else:
                        print(f"      ⚠️ Обучение не удалось")

                if terminal:
                    break
            
            print(f"  🏁 Эпизод {episode} завершен, reward={episode_reward:.4f}")
            
            # Статистика действий (если доступна)
            if hasattr(env, 'action_counts'):
                print(f"  🎮 Действия: HOLD={env.action_counts.get(0, 0)}, BUY={env.action_counts.get(1, 0)}, SELL={env.action_counts.get(2, 0)}")
            
            # Обновляем epsilon
            eps_final = getattr(cfg, 'eps_final', 0.01)  # По умолчанию минимальный epsilon 0.01
            dqn_solver.epsilon = max(eps_final, dqn_solver.epsilon * dqn_solver._eps_decay_rate)
            
            # Собираем статистику эпизода
            if hasattr(env, 'trades') and env.trades:
                all_trades.extend(env.trades)
                
                # Вычисляем winrate для эпизода
                profitable_trades = [t for t in env.trades if t.get('roi', 0) > 0]
                episode_winrate = len(profitable_trades) / len(env.trades) if env.trades else 0
                episode_winrates.append(episode_winrate)
                
                # Детальная статистика эпизода (как в оригинале)
                episode_stats = dqn_solver.print_trade_stats(env.trades)
                print(f"  📈 Статистика эпизода: сделок={len(env.trades)}, winrate={episode_winrate:.3f}")
                print(f"  💰 Прибыль: {episode_stats['avg_profit']:.4f}, Убыток: {episode_stats['avg_loss']:.4f}")
                print(f"  📊 P/L ratio: {episode_stats['pl_ratio']:.2f}, Bad trades: {episode_stats['bad_trades_count']}")
                
                # Проверяем на улучшение
                if episode_winrate > best_winrate:
                    best_winrate = episode_winrate
                    patience_counter = 0
                    
                    # Сохраняем лучшую модель
                    dqn_solver.save_model()
                    logger.info("[INFO] New best winrate: %.3f, saving model", best_winrate)
                    print(f"  🎉 Новый лучший winrate: {best_winrate:.3f}!")
                else:
                    patience_counter += 1
            else:
                # Если нет сделок, добавляем 0 winrate
                episode_winrates.append(0.0)
                patience_counter += 1
                print(f"  ⚠️ Эпизод {episode}: нет сделок")
                
                # Показываем статистику фильтров
                if hasattr(env, 'buy_attempts') and env.buy_attempts > 0:
                    vol_rejected = getattr(env, 'buy_rejected_vol', 0)
                    roi_rejected = getattr(env, 'buy_rejected_roi', 0)
                    print(f"  🔍 Попытки покупки: {env.buy_attempts}, отклонено по объему: {vol_rejected}, отклонено по ROI: {roi_rejected}")
            
            # Логируем прогресс
            if episode % 10 == 0:
                avg_winrate = np.mean(episode_winrates[-10:]) if episode_winrates else 0
                logger.info(f"[INFO] Episode {episode}/{episodes}, Avg Winrate: {avg_winrate:.3f}, Epsilon: {dqn_solver.epsilon:.4f}")
                
                # Очищаем GPU память каждые 10 эпизодов
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Early stopping
            if patience_counter >= patience_limit:
                logger.info(f"[INFO] Early stopping triggered after {episode} episodes")
                break

        # Финальная статистика
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print("\n" + "="*60)
        print("📊 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ")
        print("="*60)
        
        print(f"⏱️ ВРЕМЯ ОБУЧЕНИЯ:")
        print(f"  • Общее время: {total_training_time:.2f} секунд ({total_training_time/60:.1f} минут)")
        print(f"  • Время на эпизод: {total_training_time/episodes:.2f} секунд")
        print(f"  • Эпизодов в минуту: {episodes/(total_training_time/60):.1f}")
        
        stats_all = dqn_solver.print_trade_stats(all_trades)
        
        # Дополнительная статистика
        if all_trades:
            total_profit = sum([t.get('roi', 0) for t in all_trades if t.get('roi', 0) > 0])
            total_loss = abs(sum([t.get('roi', 0) for t in all_trades if t.get('roi', 0) < 0]))
            avg_duration = np.mean([t.get('duration', 0) for t in all_trades])
            
            print(f"\n💰 Общая статистика:")
            print(f"  • Общая прибыль: {total_profit:.4f}")
            print(f"  • Общий убыток: {total_loss:.4f}")
            print(f"  • Средняя длительность сделки: {avg_duration:.1f} минут")
            print(f"  • Всего эпизодов: {episodes}")
            print(f"  • Средний winrate: {np.mean(episode_winrates):.3f}")
        else:
            print(f"\n⚠️ Нет сделок за все {episodes} эпизодов!")
        
        # Проверяем, что у cfg есть необходимые атрибуты
        if hasattr(cfg, 'csv_metrics_path'):
            log_csv(cfg.csv_metrics_path, {"scope":"cumulative", "episode": episodes, **stats_all})
        else:
            print("⚠️ csv_metrics_path не найден в конфигурации, пропускаю логирование в CSV")
        
        if hasattr(cfg, 'use_wandb') and cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        dqn_solver.save()
        print("\n✅ Модель сохранена")
        
        # Сохраняем детальные результаты обучения
        training_results = {
            'episodes': episodes,
            'total_training_time': total_training_time,
            'episode_winrates': episode_winrates,
            'all_trades': all_trades,
            'best_winrate': best_winrate,
            'final_stats': stats_all,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': 'dqn_model.pth'
        }
        
        # Сохраняем результаты в файл
        results_file = f'training_results_{int(time.time())}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(training_results, f, protocol=HIGHEST_PROTOCOL)
        
        print(f"📊 Детальные результаты сохранены в: {results_file}")
        
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
