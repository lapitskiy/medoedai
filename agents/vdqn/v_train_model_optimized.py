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
    patience_limit: int = 3000,  # Увеличено с 2000 до 3000 для более длительного обучения
    use_wandb: bool = False,
    load_model_path: Optional[str] = None,
    load_buffer_path: Optional[str] = None
) -> str:
    """
    Оптимизированная функция тренировки модели без pandas в hot-path
    
    Args:
        dfs: словарь с DataFrame для разных таймфреймов (df_5min, df_15min, df_1h)
        cfg: конфигурация модели
        episodes: количество эпизодов для тренировки
        patience_limit: лимит терпения для early stopping (по умолчанию 2000 эпизодов)
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
        
        # Проверяем тип данных: мультивалютные или одиночные
        is_multi_crypto = False
        if dfs and isinstance(dfs, dict):
            # Проверяем, есть ли ключи с названиями криптовалют
            first_key = list(dfs.keys())[0]
            if isinstance(first_key, str) and first_key.endswith('USDT'):
                # Это мультивалютные данные
                is_multi_crypto = True
                print(f"🌍 Обнаружены мультивалютные данные для {len(dfs)} криптовалют")
                for symbol, data in dfs.items():
                    print(f"  • {symbol}: {data.get('candle_count', 'N/A')} свечей")
        
        if is_multi_crypto:
            # Используем мультивалютное окружение
            from envs.dqn_model.gym.crypto_trading_env_multi import MultiCryptoTradingEnv
            env = MultiCryptoTradingEnv(dfs=dfs, cfg=cfg)
            print(f"✅ Создано мультивалютное окружение для {len(dfs)} криптовалют")
        else:
            # Используем обычное окружение для одной криптовалюты
            dfs = prepare_data_for_training(dfs)
            env = CryptoTradingEnvOptimized(
                dfs=dfs,
                cfg=cfg,
                lookback_window=20,
                indicators_config=None  # Используем дефолтную конфигурацию
            )
            print(f"✅ Создано обычное окружение для одной криптовалюты")
        
        # Начинаем отсчет времени тренировки
        training_start_time = time.time()
        
        # Проверяем, что окружение правильно инициализировано
        if not hasattr(env, 'observation_space_shape'):
            # Попробуем вычислить размер состояния из observation_space
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'shape'):
                env.observation_space_shape = env.observation_space.shape[0]
                print(f"⚠️ Вычислен размер состояния из observation_space: {env.observation_space_shape}")
            else:
                raise ValueError("Окружение не имеет observation_space_shape и не может быть вычислен")
        
        # Получаем символ криптовалюты для логирования
        if is_multi_crypto:
            crypto_symbol = "МУЛЬТИВАЛЮТА"  # Для мультивалютного окружения
            print(f"✅ Мультивалютное окружение создано, размер состояния: {env.observation_space_shape}")
        else:
            crypto_symbol = getattr(env, 'symbol', 'UNKNOWN')
            print(f"✅ Окружение создано для {crypto_symbol}, размер состояния: {env.observation_space_shape}")

        # Настраиваем директорию вывода и имена файлов под символ
        def _symbol_code(sym: str) -> str:
            if not isinstance(sym, str) or not sym:
                return "model"
            s = sym.upper().replace('/', '')
            for suffix in ["USDT", "USD", "USDC", "BUSD", "USDP"]:
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            s = s.lower() if s else "model"
            if s in ("мультивалюта", "multi", "multicrypto"):
                s = "multi"
            return s

        result_dir = os.path.join("result")
        os.makedirs(result_dir, exist_ok=True)
        symbol_code = _symbol_code(crypto_symbol)
        # Короткий UUID для версионирования
        import uuid
        short_id = str(uuid.uuid4())[:4].lower()
        # Подготавливаем НОВЫЕ пути сохранения (после загрузки чекпойнта)
        new_model_path = os.path.join(result_dir, f"dqn_model_{symbol_code}_{short_id}.pth")
        new_buffer_path = os.path.join(result_dir, f"replay_buffer_{symbol_code}_{short_id}.pkl")

        # Создаем DQN solver
        print(f"🚀 Создаю DQN solver")
        
        dqn_solver = DQNSolver(
            observation_space=env.observation_space_shape,
            action_space=env.action_space.n
        )
        # Если указан путь загрузки существующей модели/буфера — загружаем сначала
        if load_model_path and isinstance(load_model_path, str):
            try:
                dqn_solver.cfg.model_path = load_model_path
            except Exception:
                pass
        if load_buffer_path and isinstance(load_buffer_path, str):
            try:
                dqn_solver.cfg.buffer_path = load_buffer_path
            except Exception:
                pass
        
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
        
        # Загружаем МОДЕЛЬ если есть (либо из указанных путей, либо по дефолту)
        dqn_solver.load_model()

        # После загрузки переназначаем пути сохранения на НОВЫЕ в result/<symbol>_<id>
        try:
            cfg.model_path = new_model_path
            cfg.buffer_path = new_buffer_path
            dqn_solver.cfg.model_path = cfg.model_path
            dqn_solver.cfg.buffer_path = cfg.buffer_path
        except Exception:
            pass
        
        # Переменные для отслеживания прогресса
        all_trades = []
        episode_winrates = []
        best_winrate = 0.0
        patience_counter = 0
        global_step = 0
        grad_steps = 0
        actual_episodes = episodes  # ИСПРАВЛЕНИЕ: Переменная для отслеживания реального количества эпизодов
        
        # Принудительное exploration в начале для Noisy Networks
        if getattr(cfg, 'use_noisy_networks', True):
            dqn_solver.epsilon = 0.3  # Начинаем с 30% exploration
            #print(f"🔀 Noisy Networks: принудительное exploration с epsilon={dqn_solver.epsilon}")
        
        # Улучшенные переменные для early stopping (УЛУЧШЕНО)
        min_episodes_before_stopping = getattr(cfg, 'min_episodes_before_stopping', max(4000, episodes // 3))  # Увеличил с 3000 до 4000 и с 1/4 до 1/3
        winrate_history = []  # История winrate для анализа трендов
        recent_improvement_threshold = 0.002  # Увеличил с 0.001 до 0.002 для более стабильного обучения
        
        # Адаптивный patience_limit в зависимости от количества эпизодов
        if episodes >= 10000:
            patience_limit = max(patience_limit, episodes // 3)  # Для очень длинных тренировок - минимум 1/3 (было 1/2)
        elif episodes >= 5000:
            patience_limit = max(patience_limit, episodes // 4)  # Для длинных тренировок - минимум 1/4 (было 1/3)
        elif episodes >= 2000:
            patience_limit = max(patience_limit, episodes // 3)  # Для средних тренировок - минимум 1/3 (было 1/2)
        
        # Увеличиваем patience для длинных тренировок
        patience_limit = max(patience_limit, 8000)  # Минимум 8000 эпизодов (было 5000)
        
        long_term_patience = int(patience_limit * getattr(cfg, 'long_term_patience_multiplier', 2.5))
        trend_threshold = getattr(cfg, 'early_stopping_trend_threshold', 0.05)  # Увеличиваем порог тренда с 0.03 до 0.05
        
        # Определяем название для логирования
        training_name = "МУЛЬТИВАЛЮТА" if is_multi_crypto else crypto_symbol
        print(f"🎯 Начинаю тренировку на {episodes} эпизодов для {training_name}")
        print(f"📊 Параметры Early Stopping:")
        print(f"  • min_episodes_before_stopping: {min_episodes_before_stopping}")
        print(f"  • patience_limit: {patience_limit}")
        print(f"  • long_term_patience: {long_term_patience}")
        print(f"  • trend_threshold: {trend_threshold}")
        print(f"  • Самый ранний stopping: {min_episodes_before_stopping + patience_limit} эпизодов")            
        # Информация о настройках сохранения
        save_frequency = getattr(cfg, 'save_frequency', 50)
        save_only_on_improvement = getattr(cfg, 'save_only_on_improvement', False)

        # Основной цикл тренировки
        for episode in range(episodes):         
            state = env.reset()            
            # Убеждаемся, что state является numpy массивом
            if isinstance(state, (list, tuple)):
                state = np.array(state, dtype=np.float32)
            elif not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            episode_reward = 0
            
            # Получаем текущую криптовалюту для мультивалютного окружения
            current_crypto = crypto_symbol
            if is_multi_crypto and hasattr(env, 'current_symbol'):
                current_crypto = env.current_symbol
            
            print(f"  🎯 Эпизод {episode} для {current_crypto} начат, reward={episode_reward}")
            
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
                
                # Получаем n-step transitions и добавляем их в replay buffer
                # Только если эпизод не завершен (не terminal)
                if not terminal:
                    n_step_transitions = env.get_n_step_return()
                    if n_step_transitions:
                        dqn_solver.memory.push_n_step(n_step_transitions)
                

                # Обновляем состояние
                state = state_next
                
                # Убеждаемся, что обновленный state является numpy массивом
                if isinstance(state, (list, tuple)):
                    state = np.array(state, dtype=np.float32)
                elif not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                
                episode_reward += reward
                global_step += 1
                
                # Обучаем модель чаще для лучшего обучения (УЛУЧШЕНО)
                soft_update_every = getattr(cfg, 'soft_update_every', 50)   # Уменьшил с 100 до 50 для более частого обучения
                batch_size = getattr(cfg, 'batch_size', 128)               # Увеличил с 64 до 128 для лучшей стабильности
                target_update_freq = getattr(cfg, 'target_update_freq', 500)  # Уменьшил с 1000 до 500 для более частого обновления target
                
                if global_step % soft_update_every == 0 and len(dqn_solver.memory) >= batch_size:                    
                    success, loss, abs_q, q_gap = dqn_solver.experience_replay(need_metrics=True)
                    if success:
                        grad_steps += 1                        
                        
                        # Обновляем target network чаще
                        if global_step % target_update_freq == 0:
                            dqn_solver.update_target_model()
                    else:
                        print(f"      ⚠️   Обучение не удалось")

                if terminal:
                    break
            
            # Обновляем epsilon (только если не используем Noisy Networks)
            if not getattr(cfg, 'use_noisy_networks', True):
                eps_final = getattr(cfg, 'eps_final', 0.01)  # По умолчанию минимальный epsilon 0.01
                dqn_solver.epsilon = max(eps_final, dqn_solver.epsilon * dqn_solver._eps_decay_rate)
            else:
                # При использовании Noisy Networks оставляем небольшой epsilon для стабильности
                dqn_solver.epsilon = max(0.05, dqn_solver.epsilon * 0.999)  # Минимум 5%
            
            # Собираем статистику эпизода
            # РАДИКАЛЬНОЕ ИСПРАВЛЕНИЕ: Используем env.all_trades для расчета winrate
            trades_before = len(all_trades)
            
            # ИСПРАВЛЕНИЕ: Получаем сделки из env.all_trades вместо env.trades
            if hasattr(env, 'all_trades') and env.all_trades:
                episode_trades = env.all_trades
            else:
                # Fallback: используем env.trades
                episode_trades = env.trades if hasattr(env, 'trades') and env.trades else []
            
            # ИСПРАВЛЕНИЕ: Инициализируем episode_winrate по умолчанию
            episode_winrate = 0.0
            
            if hasattr(env, 'all_trades') and env.all_trades:
                # Используем все сделки из окружения для расчета winrate
                all_profitable = [t for t in env.all_trades if t.get('roi', 0) > 0]
                episode_winrate = len(all_profitable) / len(env.all_trades) if env.all_trades else 0
                episode_winrates.append(episode_winrate)
                
                # Детальная статистика эпизода
                episode_stats = dqn_solver.print_trade_stats(env.all_trades)
                
                # Добавляем сделки в общий список если их там нет
                if len(all_trades) < len(env.all_trades):
                    all_trades.extend(env.all_trades[len(all_trades):])
                    
            elif episode_trades:
                # Fallback: используем env.trades
                all_trades.extend(episode_trades)
                
                # Вычисляем winrate для эпизода
                profitable_trades = [t for t in episode_trades if t.get('roi', 0) > 0]
                episode_winrate = len(profitable_trades) / len(episode_trades) if episode_trades else 0
                episode_winrates.append(episode_winrate)
                
                # Детальная статистика эпизода
                episode_stats = dqn_solver.print_trade_stats(episode_trades)
            else:
                # Если нет сделок вообще, используем последние сделки из all_trades
                if len(all_trades) > 0:
                    # Берем последние сделки для расчета winrate
                    recent_trades = all_trades[-min(10, len(all_trades)):]  # Последние 10 сделок
                    profitable_trades = [t for t in recent_trades if t.get('roi', 0) > 0]
                    episode_winrate = len(profitable_trades) / len(recent_trades) if recent_trades else 0
                    episode_winrates.append(episode_winrate)
                    episode_stats = dqn_solver.print_trade_stats(recent_trades)
                else:
                    # Только если действительно нет сделок
                    episode_winrate = 0.0  # ИСПРАВЛЕНИЕ: Определяем episode_winrate
                    episode_winrates.append(episode_winrate)
                    episode_stats = "Нет сделок"
                
                # Объединяем всю статистику эпизода в одну строку
                action_stats = ""
                if hasattr(env, 'action_counts'):
                    action_stats = f" | HOLD={env.action_counts.get(0, 0)}, BUY={env.action_counts.get(1, 0)}, SELL={env.action_counts.get(2, 0)}"
                
                # Добавляем информацию о времени выполнения
                time_stats = ""
                if hasattr(env, 'episode_start_time') and env.episode_start_time is not None:
                    episode_duration = time.time() - env.episode_start_time
                    steps_per_second = env.episode_step_count / episode_duration if episode_duration > 0 else 0
                    time_stats = f" | {episode_duration:.2f}с, {env.episode_step_count} шагов, {steps_per_second:.1f} шаг/с"
                
                print(f"  🏁 Эпизод {episode} для {current_crypto} завершен | reward={episode_reward:.4f}{action_stats}{time_stats} | {episode_stats}")
                
                # Проверяем на улучшение с более умной логикой
                if episode_winrate > best_winrate:
                    best_winrate = episode_winrate
                    patience_counter = 0
                    
                    # Сохраняем лучшую модель только при улучшении
                    dqn_solver.save_model()
                    logger.info("[INFO] New best winrate: %.3f, saving model", best_winrate)
                else:
                    # Мягкая логика patience - увеличиваем только при явном ухудшении
                    if episode >= min_episodes_before_stopping:
                        # Анализируем тренд winrate
                        if len(episode_winrates) >= 30:  # Увеличено с 20 до 30 для более стабильного анализа
                            recent_avg = np.mean(episode_winrates[-30:])  # Увеличено окно анализа
                            older_avg = np.mean(episode_winrates[-60:-30]) if len(episode_winrates) >= 60 else recent_avg  # Увеличено окно
                            
                            # Если есть стабильный тренд улучшения, сбрасываем patience
                            if recent_avg > older_avg + recent_improvement_threshold:
                                patience_counter = max(0, patience_counter - 5)  # Уменьшаем patience сильнее (было -3)
                            elif recent_avg > older_avg:
                                patience_counter = max(0, patience_counter - 2)  # Небольшое улучшение (было -1)
                            elif recent_avg < older_avg - 0.05:  # Увеличиваем порог ухудшения с 0.03 до 0.05
                                patience_counter += 1
                            # Если изменения небольшие, не меняем patience
                        else:
                            patience_counter += 0  # Не увеличиваем patience в начале
                    else:
                        # В начале обучения не увеличиваем patience
                        patience_counter = 0

            
            # Логируем прогресс и периодически сохраняем модель
            if episode % 10 == 0:
                avg_winrate = np.mean(episode_winrates[-10:]) if episode_winrates else 0
                
                # Получаем текущую криптовалюту для логирования
                log_crypto = current_crypto
                
                logger.info(f"[INFO] Episode {episode}/{episodes} для {log_crypto}, Avg Winrate: {avg_winrate:.3f}, Epsilon: {dqn_solver.epsilon:.4f}")
                
                # Показываем информацию о early stopping
                if episode >= min_episodes_before_stopping:
                    remaining_patience = patience_limit - patience_counter
                    print(f"  📊 Early stopping для {log_crypto}: patience {patience_counter}/{patience_limit} (осталось {remaining_patience})")
                    if patience_counter > patience_limit * 0.8:  # Показываем предупреждение при приближении к лимиту
                        print(f"  ⚠️ ВНИМАНИЕ: patience_counter приближается к лимиту!")                    
                
                # Очищаем GPU память каждые 10 эпизодов
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Периодическое сохранение модели
            save_frequency = getattr(cfg, 'save_frequency', 50)  # По умолчанию каждые 50 эпизодов
            save_only_on_improvement = getattr(cfg, 'save_only_on_improvement', False)
            
            if not save_only_on_improvement and episode > 0 and episode % save_frequency == 0:
                dqn_solver.save_model()
            
            # Улучшенный Early stopping с множественными критериями
            if episode >= min_episodes_before_stopping:
                # Дополнительная защита от слишком раннего stopping
                if episode < episodes // 2:  # Не останавливаемся в первой половине обучения (было 1/3)
                    patience_counter = min(patience_counter, patience_limit // 4)  # Ограничиваем patience сильнее (было 1/3)
                elif episode < episodes * 3 // 4:  # Дополнительная защита до 3/4 (было 1/2)
                    patience_counter = min(patience_counter, patience_limit // 2)
                
                # Основной критерий - patience
                if patience_counter >= patience_limit:
                    logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (patience limit reached)")
                    print(f"  ⚠️ Early stopping: достигнут лимит patience ({patience_limit})")
                    print(f"  🔍 Отладка: patience_counter={patience_counter}, patience_limit={patience_limit}")
                    # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                    actual_episodes = episode
                    break
                
                # Дополнительный критерий - анализ трендов (УЛУЧШЕНО)
                if len(episode_winrates) >= 400 and episode >= episodes * 4 // 5:  # Увеличил требования: 400 эпизодов и последняя 1/5
                    recent_winrate = np.mean(episode_winrates[-80:])   # Увеличил окно анализа с 50 до 80
                    mid_winrate = np.mean(episode_winrates[-160:-80])  # Увеличил окно с 100:-50 до 160:-80
                    early_winrate = np.mean(episode_winrates[-240:-160])  # Увеличил окно с 150:-100 до 240:-160
                    
                    # Если winrate стабильно падает на протяжении 240 эпизодов (более строгое условие)
                    if (recent_winrate < mid_winrate < early_winrate and 
                        mid_winrate - recent_winrate > trend_threshold * 2.5 and  # Увеличил порог с 2.0 до 2.5
                        early_winrate - mid_winrate > trend_threshold * 2.5):
                        
                        logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (declining trend)")
                        # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                        actual_episodes = episode
                        break
                
                                # Долгосрочный критерий - если модель стабильна, даем больше времени
                if patience_counter >= long_term_patience:
                    logger.info(f"[INFO] Early stopping triggered for {training_name} after {episode} episodes (long-term patience)")
                    # ИСПРАВЛЕНИЕ: Обновляем actual_episodes при early stopping
                    actual_episodes = episode
                    break

        # Финальная статистика
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        print("\n" + "="*60)
        print(f"📊 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ для {training_name}")
        print("="*60)
        
        print(f"⏱️ ВРЕМЯ ОБУЧЕНИЯ:")
        print(f"  • Общее время: {total_training_time:.2f} секунд ({total_training_time/60:.1f} минут)")
        print(f"  • Время на эпизод: {total_training_time/episode:.2f} секунд")
        print(f"  • Эпизодов в минуту: {episode/(total_training_time/60):.1f}")
        
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
            print(f"  • Планируемые эпизоды: {episodes}")
            print(f"  • Реальные эпизоды: {episode}")
            if episode < episodes:
                print(f"  • Early Stopping: Сработал на {episode} эпизоде")
            else:
                print(f"  • Early Stopping: Не сработал")
            print(f"  • Средний winrate: {np.mean(episode_winrates):.3f}")
        else:
            print(f"\n⚠️ Нет сделок за все {episodes} эпизодов!")
        
        if hasattr(cfg, 'use_wandb') and cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        # Финальное сохранение модели и replay buffer
        print("\n💾 Финальное сохранение модели и replay buffer")
        dqn_solver.save()
        
        # Сохраняем детальные результаты обучения
        # Определяем список плохих сделок (убыточные сделки)
        bad_trades_list = []
        try:
            if all_trades:
                bad_trades_list = [t for t in all_trades if t.get('roi', 0) < 0]
        except Exception:
            bad_trades_list = []

        bad_trades_count = len(bad_trades_list)
        total_trades_count = len(all_trades) if all_trades else 0
        bad_trades_percentage = (bad_trades_count / total_trades_count * 100.0) if total_trades_count > 0 else 0.0

        training_results = {
            'episodes': episodes,  # Планируемое количество эпизодов
            'actual_episodes': episode,  # Реальное количество завершенных эпизодов (текущий эпизод)
            'total_training_time': total_training_time,
            'episode_winrates': episode_winrates,
            'all_trades': all_trades,
            'bad_trades': bad_trades_list,
            'bad_trades_count': bad_trades_count,
            'bad_trades_percentage': bad_trades_percentage,
            'best_winrate': best_winrate,
            'final_stats': stats_all,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': cfg.model_path,
            'buffer_path': cfg.buffer_path,
            'symbol': training_name,
            'model_id': short_id,
            'early_stopping_triggered': episode < episodes  # True если early stopping сработал
        }
        
        # Создаем папку если не существует (используем result/)
        results_dir = os.path.join("result")
        os.makedirs(results_dir, exist_ok=True)
        
        # Сохраняем результаты в файле c символом и id
        results_file = os.path.join(results_dir, f'train_result_{symbol_code}_{short_id}.pkl')
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
