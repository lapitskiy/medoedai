import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import torch

def analyze_training_performance(metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Анализирует производительность обучения DQN агента
    
    Args:
        metrics_data: Список словарей с метриками эпизодов
        
    Returns:
        Словарь с анализом производительности
    """
    
    if not metrics_data:
        return {}
    
    df = pd.DataFrame(metrics_data)
    
    analysis = {
        'total_episodes': len(df),
        'avg_reward': df['reward'].mean() if 'reward' in df.columns else 0.0,
        'avg_profit': df['total_profit'].mean() if 'total_profit' in df.columns else 0.0,
        'success_rate': (df['total_profit'] > 0).mean() if 'total_profit' in df.columns else 0.0,
        'epsilon_trend': df['epsilon'].iloc[-10:].mean() if 'epsilon' in df.columns else 0.0,
    }
    
    # Анализ трендов
    if len(df) > 10:
        recent_df = df.tail(10)
        overall_df = df
        
        analysis['recent_vs_overall'] = {
            'reward': {
                'recent': recent_df['reward'].mean() if 'reward' in recent_df.columns else 0.0,
                'overall': overall_df['reward'].mean() if 'reward' in overall_df.columns else 0.0
            },
            'profit': {
                'recent': recent_df['total_profit'].mean() if 'total_profit' in recent_df.columns else 0.0,
                'overall': overall_df['total_profit'].mean() if 'total_profit' in overall_df.columns else 0.0
            }
        }
        
        # Определяем тренд
        if analysis['recent_vs_overall']['profit']['recent'] > analysis['recent_vs_overall']['profit']['overall']:
            analysis['trend'] = 'improving'
        elif analysis['recent_vs_overall']['profit']['recent'] < analysis['recent_vs_overall']['profit']['overall']:
            analysis['trend'] = 'declining'
        else:
            analysis['trend'] = 'stable'
    
    return analysis

def plot_training_metrics(metrics_data: List[Dict[str, Any]], save_path: str = None):
    """
    Создает графики для анализа обучения
    
    Args:
        metrics_data: Список словарей с метриками
        save_path: Путь для сохранения графиков
    """
    
    if not metrics_data:
        print("Нет данных для построения графиков")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Metrics Analysis', fontsize=16)
    
    # График 1: Награды по эпизодам
    if 'reward' in df.columns:
        axes[0, 0].plot(df.index, df['reward'], alpha=0.6, color='blue')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Добавляем скользящее среднее
        if len(df) > 10:
            window = min(20, len(df) // 5)
            rolling_mean = df['reward'].rolling(window=window).mean()
            axes[0, 0].plot(df.index, rolling_mean, color='red', linewidth=2, label=f'Rolling Mean (w={window})')
            axes[0, 0].legend()
    
    # График 2: Прибыль по эпизодам
    if 'total_profit' in df.columns:
        axes[0, 1].plot(df.index, df['total_profit'], alpha=0.6, color='green')
        axes[0, 1].set_title('Episode Profits')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Profit')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Добавляем линию нуля
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        if len(df) > 10:
            window = min(20, len(df) // 5)
            rolling_mean = df['total_profit'].rolling(window=window).mean()
            axes[0, 1].plot(df.index, rolling_mean, color='orange', linewidth=2, label=f'Rolling Mean (w={window})')
            axes[0, 1].legend()
    
    # График 3: Epsilon по эпизодам
    if 'epsilon' in df.columns:
        axes[1, 0].plot(df.index, df['epsilon'], alpha=0.8, color='purple')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)
    
    # График 4: Статистика по эпизодам
    if 'buy_attempts' in df.columns:
        axes[1, 1].scatter(df.index, df['buy_attempts'], alpha=0.6, color='orange')
        axes[1, 1].set_title('Buy Attempts per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Buy Attempts')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Графики сохранены в {save_path}")
    
    plt.show()

def analyze_trade_patterns(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Анализирует паттерны торговли
    
    Args:
        trades: Список сделок
        
    Returns:
        Словарь с анализом паттернов
    """
    
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    analysis = {
        'total_trades': len(trades),
        'winning_trades': len(df[df['profit'] > 0]) if 'profit' in df.columns else 0,
        'losing_trades': len(df[df['profit'] < 0]) if 'profit' in df.columns else 0,
        'winrate': len(df[df['profit'] > 0]) / len(trades) if 'profit' in df.columns else 0.0,
    }
    
    if 'profit' in df.columns:
        profits = df[df['profit'] > 0]['profit']
        losses = df[df['profit'] < 0]['profit']
        
        analysis.update({
            'avg_profit': profits.mean() if len(profits) > 0 else 0.0,
            'avg_loss': abs(losses.mean()) if len(losses) > 0 else 0.0,
            'max_profit': profits.max() if len(profits) > 0 else 0.0,
            'max_loss': abs(losses.min()) if len(losses) > 0 else 0.0,
            'profit_std': profits.std() if len(profits) > 0 else 0.0,
            'loss_std': losses.std() if len(losses) > 0 else 0.0,
        })
        
        # P/L ratio
        if analysis['avg_loss'] > 0:
            analysis['pl_ratio'] = analysis['avg_profit'] / analysis['avg_loss']
        else:
            analysis['pl_ratio'] = float('inf')
    
    # Анализ времени удержания позиций
    if 'duration' in df.columns:
        analysis['avg_hold_time'] = df['duration'].mean()
        analysis['hold_time_std'] = df['duration'].std()
        
        # Группировка по времени удержания
        df['hold_time_group'] = pd.cut(df['duration'], 
                                     bins=[0, 5, 15, 30, 60, float('inf')], 
                                     labels=['0-5min', '5-15min', '15-30min', '30-60min', '60min+'])
        
        hold_time_stats = df.groupby('hold_time_group')['profit'].agg(['count', 'mean']).to_dict()
        analysis['hold_time_analysis'] = hold_time_stats
    
    # Анализ по времени дня (если есть timestamp)
    if 'timestamp' in df.columns:
        try:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_stats = df.groupby('hour')['profit'].agg(['count', 'mean']).to_dict()
            analysis['hourly_analysis'] = hourly_stats
        except:
            pass
    
    return analysis

def generate_training_report(metrics_data: List[Dict[str, Any]], trades: List[Dict[str, Any]], 
                           save_path: str = None) -> str:
    """
    Генерирует подробный отчет об обучении
    
    Args:
        metrics_data: Метрики обучения
        trades: Список сделок
        save_path: Путь для сохранения отчета
        
    Returns:
        Текст отчета
    """
    
    # Анализ обучения
    training_analysis = analyze_training_performance(metrics_data)
    
    # Анализ торговли
    trade_analysis = analyze_trade_patterns(trades)
    
    # Формируем отчет
    report = []
    report.append("=" * 60)
    report.append("DQN TRAINING REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Общая статистика
    report.append("📊 TRAINING OVERVIEW:")
    report.append(f"  • Total Episodes: {training_analysis.get('total_episodes', 0)}")
    report.append(f"  • Average Reward: {training_analysis.get('avg_reward', 0):.4f}")
    report.append(f"  • Average Profit: {training_analysis.get('avg_profit', 0):.4f}")
    report.append(f"  • Success Rate: {training_analysis.get('success_rate', 0):.2%}")
    report.append(f"  • Current Epsilon: {training_analysis.get('epsilon_trend', 0):.4f}")
    
    if 'trend' in training_analysis:
        trend_emoji = "📈" if training_analysis['trend'] == 'improving' else "📉" if training_analysis['trend'] == 'declining' else "➡️"
        report.append(f"  • Performance Trend: {trend_emoji} {training_analysis['trend'].upper()}")
    
    report.append("")
    
    # Анализ торговли
    report.append("💼 TRADING ANALYSIS:")
    report.append(f"  • Total Trades: {trade_analysis.get('total_trades', 0)}")
    report.append(f"  • Winrate: {trade_analysis.get('winrate', 0):.2%}")
    report.append(f"  • Winning Trades: {trade_analysis.get('winning_trades', 0)}")
    report.append(f"  • Losing Trades: {trade_analysis.get('losing_trades', 0)}")
    
    if 'avg_profit' in trade_analysis:
        report.append(f"  • Average Profit: {trade_analysis['avg_profit']:.4f}")
        report.append(f"  • Average Loss: {trade_analysis['avg_loss']:.4f}")
        report.append(f"  • P/L Ratio: {trade_analysis.get('pl_ratio', 0):.2f}")
        report.append(f"  • Max Profit: {trade_analysis['max_profit']:.4f}")
        report.append(f"  • Max Loss: {trade_analysis['max_loss']:.4f}")
    
    if 'avg_hold_time' in trade_analysis:
        report.append(f"  • Average Hold Time: {trade_analysis['avg_hold_time']:.1f} minutes")
    
    report.append("")
    
    # Рекомендации
    report.append("💡 RECOMMENDATIONS:")
    
    if trade_analysis.get('winrate', 0) < 0.5:
        report.append("  • ⚠️ Low winrate detected. Consider:")
        report.append("    - Increasing exploration (epsilon)")
        report.append("    - Reviewing reward function")
        report.append("    - Adding more features to state")
    
    if trade_analysis.get('pl_ratio', 0) < 1.2:
        report.append("  • ⚠️ Low P/L ratio. Consider:")
        report.append("    - Improving stop-loss strategy")
        report.append("    - Better entry timing")
        report.append("    - Risk management rules")
    
    if training_analysis.get('epsilon_trend', 1) < 0.1:
        report.append("  • ℹ️ Low exploration rate. Consider:")
        report.append("    - Slower epsilon decay")
        report.append("    - Adaptive exploration")
    
    if training_analysis.get('trend') == 'declining':
        report.append("  • ⚠️ Performance declining. Consider:")
        report.append("    - Early stopping")
        report.append("    - Hyperparameter tuning")
        report.append("    - Model architecture review")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Отчет сохранен в {save_path}")
    
    return report_text

def optimize_hyperparameters(metrics_data: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Предлагает оптимизацию гиперпараметров на основе анализа
    
    Args:
        metrics_data: Метрики обучения
        trades: Список сделок
        
    Returns:
        Словарь с рекомендациями по гиперпараметрам
    """
    
    training_analysis = analyze_training_performance(metrics_data)
    trade_analysis = analyze_trade_patterns(trades)
    
    recommendations = {}
    
    # Анализ epsilon
    current_epsilon = training_analysis.get('epsilon_trend', 0.5)
    if current_epsilon < 0.1:
        recommendations['epsilon'] = {
            'current': current_epsilon,
            'suggested': min(0.3, current_epsilon * 2),
            'reason': 'Too low exploration rate, increasing for better exploration'
        }
    
    # Анализ learning rate
    if training_analysis.get('trend') == 'declining':
        recommendations['learning_rate'] = {
            'current': 'unknown',
            'suggested': 'decrease by 20%',
            'reason': 'Performance declining, may be learning too fast'
        }
    
    # Анализ batch size
    if trade_analysis.get('winrate', 0) < 0.4:
        recommendations['batch_size'] = {
            'current': 'unknown',
            'suggested': 'increase by 50%',
            'reason': 'Low winrate, larger batches may provide more stable learning'
        }
    
    # Анализ gamma
    if trade_analysis.get('pl_ratio', 0) < 1.0:
        recommendations['gamma'] = {
            'current': 'unknown',
            'suggested': 'decrease to 0.95',
            'reason': 'Low P/L ratio, focus on immediate rewards'
        }
    
    return recommendations

def analyze_bad_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Детальный анализ плохих сделок для понимания ошибок модели
    
    Args:
        trades: Список сделок
        
    Returns:
        Словарь с анализом плохих сделок
    """
    
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    # Определяем плохие сделки (ROI < 0.1%)
    bad_trades = df[abs(df['roi']) < 0.001]
    
    if len(bad_trades) == 0:
        return {'bad_trades_count': 0, 'message': 'Плохих сделок не найдено'}
    
    analysis = {
        'bad_trades_count': len(bad_trades),
        'bad_trades_percentage': len(bad_trades) / len(trades) * 100,
        'bad_trades_details': []
    }
    
    # Анализируем каждую плохую сделку
    for idx, trade in bad_trades.iterrows():
        trade_analysis = {
            'trade_id': idx,
            'roi': trade.get('roi', 0),
            'duration': trade.get('duration', 0),
            'entry_time': trade.get('entry_time', 'N/A'),
            'exit_time': trade.get('exit_time', 'N/A'),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'volume': trade.get('volume', 0),
            'action': trade.get('action', 'N/A'),
            'market_conditions': trade.get('market_conditions', 'N/A')
        }
        analysis['bad_trades_details'].append(trade_analysis)
    
    # Статистика плохих сделок
    bad_rois = bad_trades['roi']
    analysis.update({
        'avg_bad_roi': bad_rois.mean(),
        'min_bad_roi': bad_rois.min(),
        'max_bad_roi': bad_rois.max(),
        'bad_roi_std': bad_rois.std(),
        
        # Анализ по времени
        'avg_bad_duration': bad_trades['duration'].mean() if 'duration' in bad_trades.columns else 0,
        'min_bad_duration': bad_trades['duration'].min() if 'duration' in bad_trades.columns else 0,
        'max_bad_duration': bad_trades['duration'].max() if 'duration' in bad_trades.columns else 0,
        
        # Анализ по объему
        'avg_bad_volume': bad_trades['volume'].mean() if 'volume' in bad_trades.columns else 0,
        'min_bad_volume': bad_trades['volume'].min() if 'volume' in bad_trades.columns else 0,
        'max_bad_volume': bad_trades['volume'].max() if 'volume' in bad_trades.columns else 0
    })
    
    # Группировка по причинам
    very_small_losses = bad_trades[bad_trades['roi'] < 0]
    very_small_profits = bad_trades[bad_trades['roi'] > 0]
    
    analysis['loss_distribution'] = {
        'very_small_losses': len(very_small_losses),
        'very_small_profits': len(very_small_profits),
        'neutral_trades': len(bad_trades) - len(very_small_losses) - len(very_small_profits)
    }
    
    return analysis

def print_bad_trades_analysis(analysis: Dict[str, Any]):
    """
    Красивый вывод анализа плохих сделок
    """
    if not analysis or analysis.get('bad_trades_count', 0) == 0:
        print("✅ Плохих сделок не найдено!")
        return
    
    print(f"\n🔍 АНАЛИЗ ПЛОХИХ СДЕЛОК ({analysis['bad_trades_count']} шт.)")
    print("=" * 60)
    
    print(f"📊 Общая статистика:")
    print(f"  • Количество плохих сделок: {analysis['bad_trades_count']}")
    print(f"  • Процент от общих сделок: {analysis['bad_trades_percentage']:.2f}%")
    print(f"  • Средний ROI: {analysis['avg_bad_roi']:.6f} ({analysis['avg_bad_roi']*100:.4f}%)")
    print(f"  • Минимальный ROI: {analysis['min_bad_roi']:.6f} ({analysis['min_bad_roi']*100:.4f}%)")
    print(f"  • Максимальный ROI: {analysis['max_bad_roi']:.6f} ({analysis['max_bad_roi']*100:.4f}%)")
    
    print(f"\n⏱️ Временной анализ:")
    print(f"  • Средняя длительность: {analysis['avg_bad_duration']:.1f} минут")
    print(f"  • Минимальная длительность: {analysis['min_bad_duration']:.1f} минут")
    print(f"  • Максимальная длительность: {analysis['max_bad_duration']:.1f} минут")
    
    print(f"\n📈 Распределение по типам:")
    dist = analysis['loss_distribution']
    print(f"  • Очень маленькие убытки: {dist['very_small_losses']} ({dist['very_small_losses']/analysis['bad_trades_count']*100:.1f}%)")
    print(f"  • Очень маленькие прибыли: {dist['very_small_profits']} ({dist['very_small_profits']/analysis['bad_trades_count']*100:.1f}%)")
    print(f"  • Нейтральные сделки: {dist['neutral_trades']} ({dist['neutral_trades']/analysis['bad_trades_count']*100:.1f}%)")
    
    print(f"\n💡 Рекомендации:")
    if dist['very_small_losses'] > dist['very_small_profits']:
        print(f"  • Большинство плохих сделок - убыточные")
        print(f"  • Рекомендуется улучшить точность входа в позицию")
    else:
        print(f"  • Большинство плохих сделок - малоприбыльные")
        print(f"  • Рекомендуется улучшить управление выходом из позиции")
    
    if analysis['avg_bad_duration'] < 300:  # Меньше 5 часов
        print(f"  • Плохие сделки имеют короткую длительность")
        print(f"  • Возможно, модель слишком быстро закрывает позиции")
    elif analysis['avg_bad_duration'] > 600:  # Больше 10 часов
        print(f"  • Плохие сделки имеют длительную продолжительность")
        print(f"  • Возможно, модель слишком долго держит убыточные позиции")
