import torch
import numpy as np
import pandas as pd
from agents.vdqn.dqnsolver import DQNSolver
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized as CryptoTradingEnv
from envs.dqn_model.gym.crypto_trading_env_multi import MultiCryptoTradingEnv
import wandb
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.gutils import get_nan_stats, log_csv, setup_logger, setup_wandb

cfg = vDqnConfig()



def prepare_multi_crypto_data(dfs: dict) -> dict:
    """
    Подготавливает и нормализует данные для мультивалютного обучения.
    
    Args:
        dfs (dict): Словарь с данными криптовалют
        
    Returns:
        dict: Нормализованные данные для обучения
    """
    import pandas as pd
    import numpy as np
    from utils.db_utils import db_get_or_fetch_ohlcv
    
    print("📊 Подготовка мультивалютных данных...")
    
    # Список криптовалют для обучения
    crypto_symbols = [
        'BTCUSDT',  # Биткоин
        'TONUSDT',  # TON
        'ETHUSDT',  # Эфириум
        'SOLUSDT',  # Solana
        'ADAUSDT',  # Cardano
        'BNBUSDT'   # Binance Coin
    ]
    
    normalized_dfs = {}
    
    for symbol in crypto_symbols:
        try:
            print(f"  📥 Загружаю {symbol}...")
            
            # Загружаем данные из базы
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=symbol, 
                timeframe='5m', 
                limit_candles=100000
            )
            
            if df_5min is not None and len(df_5min) > 0:
                print(f"    ✅ {symbol}: {len(df_5min)} свечей загружено")
                
                # Нормализуем данные
                df_normalized = normalize_crypto_data(df_5min, symbol)
                
                # Создаем 15-минутные и 1-часовые данные
                df_15min = df_5min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna().reset_index()
                
                df_1h = df_5min.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna().reset_index()
                
                # Добавляем в словарь
                normalized_dfs[symbol] = {
                    'df_5min': df_normalized,
                    'df_15min': df_15min,
                    'df_1h': df_1h,
                    'symbol': symbol,
                    'candle_count': len(df_normalized)
                }
                
            else:
                print(f"    ⚠️ {symbol}: данные не найдены, пропускаем")
                
        except Exception as e:
            print(f"    ❌ {symbol}: ошибка загрузки - {e}")
            continue
    
    print(f"📈 Всего подготовлено: {len(normalized_dfs)} криптовалют")
    
    # Проверяем минимальное количество свечей
    min_candles = min([data['candle_count'] for data in normalized_dfs.values()]) if normalized_dfs else 0
    print(f"📊 Минимальное количество свечей: {min_candles}")
    
    if min_candles < 10000:
        print("⚠️ Внимание: некоторые криптовалюты имеют мало данных")
    
    return normalized_dfs

def normalize_crypto_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Нормализует данные криптовалюты для обучения.
    
    Args:
        df (pd.DataFrame): Исходные данные
        symbol (str): Символ криптовалюты
        
    Returns:
        pd.DataFrame: Нормализованные данные
    """
    try:
        # Копируем данные
        df_norm = df.copy()
        
        # 1. Нормализация цен (логарифмическая)
        if 'close' in df_norm.columns:
            df_norm['close_norm'] = np.log(df_norm['close'] / df_norm['close'].iloc[0])
        
        # 2. Нормализация объемов
        if 'volume' in df_norm.columns:
            volume_mean = df_norm['volume'].mean()
            volume_std = df_norm['volume'].std()
            if volume_std > 0:
                df_norm['volume_norm'] = (df_norm['volume'] - volume_mean) / volume_std
            else:
                df_norm['volume_norm'] = 0
        
        # 3. Нормализация волатильности (High-Low)
        if 'high' in df_norm.columns and 'low' in df_norm.columns:
            df_norm['volatility'] = (df_norm['high'] - df_norm['low']) / df_norm['close']
            vol_mean = df_norm['volatility'].mean()
            vol_std = df_norm['volatility'].std()
            if vol_std > 0:
                df_norm['volatility_norm'] = (df_norm['volatility'] - vol_mean) / vol_std
            else:
                df_norm['volatility_norm'] = 0
        
        # 4. Технические индикаторы
        # RSI
        if 'close' in df_norm.columns:
            delta = df_norm['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_norm['rsi'] = 100 - (100 / (1 + rs))
            df_norm['rsi_norm'] = df_norm['rsi'] / 100  # Нормализация 0-1
        
        # MACD
        if 'close' in df_norm.columns:
            exp1 = df_norm['close'].ewm(span=12).mean()
            exp2 = df_norm['close'].ewm(span=26).mean()
            df_norm['macd'] = exp1 - exp2
            df_norm['macd_signal'] = df_norm['macd'].ewm(span=9).mean()
            
            # Нормализация MACD
            macd_mean = df_norm['macd'].mean()
            macd_std = df_norm['macd'].std()
            if macd_std > 0:
                df_norm['macd_norm'] = (df_norm['macd'] - macd_mean) / macd_std
            else:
                df_norm['macd_norm'] = 0
        
        # 5. Временные признаки
        df_norm['hour'] = pd.to_datetime(df_norm.index).hour
        df_norm['day_of_week'] = pd.to_datetime(df_norm.index).dayofweek
        
        # Нормализация времени
        df_norm['hour_norm'] = df_norm['hour'] / 24
        df_norm['day_norm'] = df_norm['day_of_week'] / 7
        
        # 6. Заполняем NaN значения
        df_norm = df_norm.ffill().fillna(0)
        
        print(f"    🔧 {symbol}: нормализация завершена")
        return df_norm
        
    except Exception as e:
        print(f"    ❌ {symbol}: ошибка нормализации - {e}")
        return df

def train_model(dfs: dict, load_previous: bool = False, episodes: int = 200, multi_crypto: bool = False):
    """
    Обучает улучшенную модель DQN для торговли криптовалютой с GPU оптимизациями.

    Args:
        dfs (dict): Словарь с Pandas DataFrames для разных таймфреймов (df_5min, df_15min, df_1h).
        load_previous (bool): Загружать ли ранее сохраненную модель.
        episodes (int): Количество эпизодов для обучения.
        multi_crypto (bool): Использовать ли мультивалютное обучение.
    Returns:
        str: Сообщение о завершении обучения.
    """

    import time    
    training_start_time = time.time()  # Засекаем время начала обучения

    all_trades = []
    best_winrate = 0.0
    patience_counter = 0
    patience_limit = 500  # Early stopping после 500 эпизодов без улучшений (увеличено для более длительного обучения)

    wandb_run = None
    
    # Подготовка мультивалютных данных (если не переданы)
    if multi_crypto and not dfs:
        print("🚀 Подготовка мультивалютных данных...")
        dfs = prepare_multi_crypto_data(dfs)
        print(f"✅ Подготовлено {len(dfs)} криптовалют для обучения")
    elif multi_crypto:
        print(f"✅ Используем переданные данные для {len(dfs)} криптовалют")

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
        if multi_crypto:
            # Для мультивалютного обучения создаем окружение с поддержкой переключения
            env = MultiCryptoTradingEnv(dfs=dfs, cfg=cfg)
            print(f"🌍 Создано мультивалютное окружение с {len(dfs)} криптовалютами")
        else:
            # Обычное окружение для одной криптовалюты
            env = CryptoTradingEnv(dfs=dfs)
            print("🌍 Создано окружение для одной криптовалюты") 

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
            
            # Логируем текущую криптовалюту для мультивалютного обучения
            if multi_crypto and hasattr(env, 'get_current_symbol'):
                current_crypto = env.get_current_symbol()
                logger.info(f"[INFO] Эпизод {episode + 1}/{episodes}: обучение на {current_crypto}")
            
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
            # ИСПРАВЛЕНИЕ: Используем env.all_trades для правильного расчета winrate
            if hasattr(env, 'all_trades') and env.all_trades:
                # Используем все сделки из окружения для расчета winrate
                all_profitable = [t for t in env.all_trades if t.get('roi', 0) > 0]
                episode_winrate = len(all_profitable) / len(env.all_trades) if env.all_trades else 0
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
                    
                # Синхронизируем сделки
                if hasattr(env, 'trades') and env.trades:
                    all_trades.extend(env.trades)
            else:
                # Fallback: используем env.trades если all_trades нет
                if hasattr(env, 'trades') and env.trades:
                    all_trades.extend(env.trades)
                    
                    # Вычисляем winrate для эпизода
                    profitable_trades = [t for t in env.trades if t.get('roi', 0) > 0]
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
                    # Если нет сделок вообще, используем последние сделки из all_trades
                    if all_trades:
                        # Берем последние сделки для расчета winrate
                        recent_trades = all_trades[-min(10, len(all_trades)):]  # Последние 10 сделок
                        profitable_trades = [t for t in recent_trades if t.get('roi', 0) > 0]
                        episode_winrate = len(profitable_trades) / len(recent_trades) if recent_trades else 0
                        episode_winrates.append(episode_winrate)
                        print(f"    💰 Эпизод {episode}: используем последние {len(recent_trades)} сделок, winrate={episode_winrate:.3f}")
                    else:
                        # Только если действительно нет сделок
                        episode_winrates.append(0.0)
                        print(f"    ⚠️ Эпизод {episode}: НЕТ сделок вообще!")
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
        
        # Сохраняем результаты в файл
        import pickle
        import os
        from datetime import datetime
        
        # Создаем папку если не существует
        os.makedirs('temp/train_results', exist_ok=True)
        
        # Имя файла с timestamp
        timestamp = int(datetime.now().timestamp())
        filename = f'temp/train_results/training_results_{timestamp}.pkl'
        
        # Добавляем недостающие поля для совместимости с анализатором
        stats_all['training_date'] = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        stats_all['episodes'] = episodes
        stats_all['actual_episodes'] = episodes
        stats_all['early_stopping_triggered'] = False  # Для мультивалютного обучения всегда False
        stats_all['episode_winrates'] = episode_winrates  # Список winrate по эпизодам
        stats_all['real_episodes'] = episodes  # Реальное количество завершенных эпизодов
        
        # Добавляем поля времени обучения
        training_end_time = time.time()
        stats_all['total_training_time'] = training_end_time - training_start_time
        
        # Добавляем best_winrate
        if episode_winrates:
            stats_all['best_winrate'] = max(episode_winrates)
        else:
            stats_all['best_winrate'] = 0.0
        
        # Добавляем all_trades (если есть)
        if 'all_trades' not in stats_all:
            stats_all['all_trades'] = all_trades
        
        # Добавляем final_stats
        stats_all['final_stats'] = stats_all.copy()
        
        # Добавляем статистику по криптовалютам для мультивалютного обучения
        if multi_crypto and hasattr(env, 'get_episode_stats'):
            crypto_stats = env.get_episode_stats()
            stats_all['crypto_stats'] = crypto_stats
            stats_all['timestamp'] = timestamp
            stats_all['episode'] = episodes
            print(f"📊 Статистика по криптовалютам добавлена в результаты")
        
        # Сохраняем результаты
        with open(filename, 'wb') as f:
            pickle.dump(stats_all, f)
        
        print(f"💾 Результаты сохранены в файл: {filename}")
        
        # Логируем в CSV
        log_csv(cfg.csv_metrics_path, {"scope":"cumulative", "episode": episodes, **stats_all})
        
        if cfg.use_wandb:
            wandb.log({**stats_all, "scope": "cumulative", "episode": episodes})
        
        dqn_solver.save()
        
        # Выводим только ключевую статистику, а не весь словарь
        print("\n" + "="*60)
        print("📊 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ")
        print("="*60)
        print(f"🎯 Эпизоды: {episodes}")
        print(f"💰 Сделок: {stats_all.get('trades_count', 'N/A')}")
        print(f"📈 Winrate: {stats_all.get('winrate', 'N/A'):.3f}")
        print(f"💵 P/L ratio: {stats_all.get('pl_ratio', 'N/A'):.3f}")
        print(f"⏱️ Время обучения: {stats_all.get('total_training_time', 'N/A'):.1f} сек")
        
        if multi_crypto and hasattr(env, 'get_episode_stats'):
            crypto_stats = env.get_episode_stats()
            print(f"\n🌍 СТАТИСТИКА ПО КРИПТОВАЛЮТАМ:")
            for symbol, stats in crypto_stats.items():
                episodes_count = stats.get('episodes', 0)
                percentage = stats.get('percentage', 0)
                print(f"  • {symbol}: {episodes_count} эпизодов ({percentage:.1f}%)")
        
        print("="*60)
        print("✅ Финальная модель сохранена.")
        
        # Анализ трендов
        if len(episode_winrates) > 10:
            recent_winrate = np.mean(episode_winrates[-10:])
            overall_winrate = np.mean(episode_winrates)
            print(f"📈 Winrate тренд: последние 10 эпизодов: {recent_winrate:.3f}, общий: {overall_winrate:.3f}")
            
            if recent_winrate > overall_winrate:
                print("✅ Модель улучшается!")
            else:
                print("⚠️ Модель может переобучаться")
        
        # Статистика по криптовалютам для мультивалютного обучения
        if multi_crypto and hasattr(env, 'print_episode_stats'):
            print("\n" + "="*60)
            print("🌍 СТАТИСТИКА ПО КРИПТОВАЛЮТАМ")
            print("="*60)
            env.print_episode_stats()
            print("="*60)
        
        return "Обучение завершено"    
    finally:
        # Закрываем wandb
        if wandb_run is not None:
            wandb_run.finish()