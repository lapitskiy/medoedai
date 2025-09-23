#!/usr/bin/env python3
"""
Анализатор QGate и предсказаний модели
Анализирует данные из БД для понимания почему не проходят qgate фильтры
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Импорты для работы с БД
from orm.database import get_db_session
from orm.models import ModelPrediction, Trade, Symbol, OHLCV

class QGateAnalyzer:
    """Анализатор QGate и предсказаний модели"""
    
    def __init__(self):
        self.session = get_db_session()
        self.qgate_stats = {}
        self.prediction_stats = {}
        
    def analyze_qgate_performance(self, days_back: int = 7) -> Dict[str, Any]:
        """Анализ производительности QGate за последние N дней"""
        print(f"🔍 Анализ QGate за последние {days_back} дней...")
        
        # Получаем все предсказания за период
        start_date = datetime.now() - timedelta(days=days_back)
        
        predictions = self.session.query(ModelPrediction).filter(
            ModelPrediction.created_at >= start_date
        ).order_by(ModelPrediction.created_at.desc()).all()
        
        if not predictions:
            print("❌ Нет предсказаний за указанный период")
            return {}
        
        print(f"📊 Найдено {len(predictions)} предсказаний")
        
        # Анализируем каждое предсказание
        qgate_analysis = {
            'total_predictions': len(predictions),
            'qgate_filtered': 0,
            'qgate_passed': 0,
            'q_values_stats': [],
            'confidence_stats': [],
            'action_distribution': {},
            'qgate_reasons': {},
            'time_analysis': {},
            'symbol_analysis': {}
        }
        
        for pred in predictions:
            # Парсим Q-values
            try:
                q_values = json.loads(pred.q_values) if pred.q_values else []
                if q_values:
                    max_q = max(q_values)
                    sorted_q = sorted(q_values, reverse=True)
                    second_q = sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]
                    gap_q = max_q - second_q
                    
                    qgate_analysis['q_values_stats'].append({
                        'max_q': max_q,
                        'gap_q': gap_q,
                        'q_values': q_values
                    })
            except:
                continue
            
            # Анализ уверенности
            if pred.confidence:
                qgate_analysis['confidence_stats'].append(pred.confidence)
            
            # Распределение действий
            action = pred.action
            if action not in qgate_analysis['action_distribution']:
                qgate_analysis['action_distribution'][action] = 0
            qgate_analysis['action_distribution'][action] += 1
            
            # Анализ по символам
            symbol = pred.symbol
            if symbol not in qgate_analysis['symbol_analysis']:
                qgate_analysis['symbol_analysis'][symbol] = {
                    'total': 0,
                    'qgate_filtered': 0,
                    'avg_confidence': 0,
                    'avg_max_q': 0
                }
            
            qgate_analysis['symbol_analysis'][symbol]['total'] += 1
            
            # Проверяем, было ли предсказание отфильтровано QGate
            market_conditions = {}
            if pred.market_conditions:
                try:
                    market_conditions = json.loads(pred.market_conditions)
                except:
                    pass
            
            is_qgate_filtered = market_conditions.get('qgate_filtered', False)
            
            if is_qgate_filtered:
                qgate_analysis['qgate_filtered'] += 1
                qgate_analysis['symbol_analysis'][symbol]['qgate_filtered'] += 1
                
                # Причина фильтрации
                reason = market_conditions.get('qgate_reason', 'unknown')
                if reason not in qgate_analysis['qgate_reasons']:
                    qgate_analysis['qgate_reasons'][reason] = 0
                qgate_analysis['qgate_reasons'][reason] += 1
            else:
                qgate_analysis['qgate_passed'] += 1
            
            # Временной анализ
            hour = pred.created_at.hour
            if hour not in qgate_analysis['time_analysis']:
                qgate_analysis['time_analysis'][hour] = {'total': 0, 'filtered': 0}
            qgate_analysis['time_analysis'][hour]['total'] += 1
            if is_qgate_filtered:
                qgate_analysis['time_analysis'][hour]['filtered'] += 1
        
        # Вычисляем статистики
        if qgate_analysis['q_values_stats']:
            max_q_values = [stat['max_q'] for stat in qgate_analysis['q_values_stats']]
            gap_q_values = [stat['gap_q'] for stat in qgate_analysis['q_values_stats']]
            
            qgate_analysis['q_values_summary'] = {
                'max_q_mean': np.mean(max_q_values),
                'max_q_std': np.std(max_q_values),
                'max_q_min': np.min(max_q_values),
                'max_q_max': np.max(max_q_values),
                'gap_q_mean': np.mean(gap_q_values),
                'gap_q_std': np.std(gap_q_values),
                'gap_q_min': np.min(gap_q_values),
                'gap_q_max': np.max(gap_q_values)
            }
        
        if qgate_analysis['confidence_stats']:
            qgate_analysis['confidence_summary'] = {
                'mean': np.mean(qgate_analysis['confidence_stats']),
                'std': np.std(qgate_analysis['confidence_stats']),
                'min': np.min(qgate_analysis['confidence_stats']),
                'max': np.max(qgate_analysis['confidence_stats'])
            }
        
        # Вычисляем проценты для символов
        for symbol, stats in qgate_analysis['symbol_analysis'].items():
            if stats['total'] > 0:
                stats['qgate_filter_rate'] = stats['qgate_filtered'] / stats['total']
                stats['avg_confidence'] = np.mean([p.confidence for p in predictions if p.symbol == symbol and p.confidence])
                stats['avg_max_q'] = np.mean([max(json.loads(p.q_values)) for p in predictions if p.symbol == symbol and p.q_values])
        
        return qgate_analysis
    
    def analyze_trading_performance(self, days_back: int = 7) -> Dict[str, Any]:
        """Анализ торговой производительности"""
        print(f"📈 Анализ торговой производительности за последние {days_back} дней...")
        
        start_date = datetime.now() - timedelta(days=days_back)
        
        trades = self.session.query(Trade).filter(
            Trade.created_at >= start_date
        ).order_by(Trade.created_at.desc()).all()
        
        if not trades:
            print("❌ Нет сделок за указанный период")
            return {}
        
        print(f"📊 Найдено {len(trades)} сделок")
        
        trading_analysis = {
            'total_trades': len(trades),
            'successful_trades': 0,
            'failed_trades': 0,
            'action_distribution': {},
            'symbol_distribution': {},
            'pnl_analysis': [],
            'confidence_vs_success': []
        }
        
        for trade in trades:
            # Успешность сделки
            if trade.is_successful:
                trading_analysis['successful_trades'] += 1
            else:
                trading_analysis['failed_trades'] += 1
            
            # Распределение по действиям
            action = trade.action
            if action not in trading_analysis['action_distribution']:
                trading_analysis['action_distribution'][action] = 0
            trading_analysis['action_distribution'][action] += 1
            
            # Распределение по символам
            symbol = trade.symbol.name if trade.symbol else 'Unknown'
            if symbol not in trading_analysis['symbol_distribution']:
                trading_analysis['symbol_distribution'][symbol] = 0
            trading_analysis['symbol_distribution'][symbol] += 1
            
            # P&L анализ
            if trade.position_pnl is not None:
                trading_analysis['pnl_analysis'].append(trade.position_pnl)
            
            # Связь уверенности и успеха
            if trade.confidence is not None:
                trading_analysis['confidence_vs_success'].append({
                    'confidence': trade.confidence,
                    'success': trade.is_successful
                })
        
        # Вычисляем статистики
        if trading_analysis['pnl_analysis']:
            pnl_values = trading_analysis['pnl_analysis']
            trading_analysis['pnl_summary'] = {
                'mean': np.mean(pnl_values),
                'std': np.std(pnl_values),
                'min': np.min(pnl_values),
                'max': np.max(pnl_values),
                'positive_count': sum(1 for pnl in pnl_values if pnl > 0),
                'negative_count': sum(1 for pnl in pnl_values if pnl < 0)
            }
        
        trading_analysis['success_rate'] = trading_analysis['successful_trades'] / trading_analysis['total_trades'] if trading_analysis['total_trades'] > 0 else 0
        
        return trading_analysis
    
    def get_current_qgate_thresholds(self) -> Dict[str, Any]:
        """Получение текущих порогов QGate"""
        import os
        
        thresholds = {
            'QGATE_MAXQ': os.environ.get('QGATE_MAXQ', 'not_set'),
            'QGATE_GAPQ': os.environ.get('QGATE_GAPQ', 'not_set'),
            'QGATE_SELL_MAXQ': os.environ.get('QGATE_SELL_MAXQ', 'not_set'),
            'QGATE_SELL_GAPQ': os.environ.get('QGATE_SELL_GAPQ', 'not_set')
        }
        
        return thresholds
    
    def suggest_qgate_improvements(self, qgate_analysis: Dict[str, Any]) -> List[str]:
        """Предложения по улучшению QGate на основе анализа"""
        suggestions = []
        
        if not qgate_analysis:
            return ["❌ Нет данных для анализа"]
        
        # Анализ статистик Q-values
        if 'q_values_summary' in qgate_analysis:
            qv = qgate_analysis['q_values_summary']
            
            # Если средний max_q очень низкий
            if qv['max_q_mean'] < 0.1:
                suggestions.append(f"🔴 Критично низкий средний max_q: {qv['max_q_mean']:.3f}. Модель не уверена в предсказаниях.")
            
            # Если gap_q очень мал
            if qv['gap_q_mean'] < 0.01:
                suggestions.append(f"🔴 Критично малый средний gap_q: {qv['gap_q_mean']:.3f}. Модель не различает действия.")
            
            # Если max_q имеет большой разброс
            if qv['max_q_std'] > qv['max_q_mean']:
                suggestions.append(f"🟡 Высокая нестабильность max_q (std={qv['max_q_std']:.3f} > mean={qv['max_q_mean']:.3f})")
        
        # Анализ фильтрации
        total = qgate_analysis['total_predictions']
        filtered = qgate_analysis['qgate_filtered']
        filter_rate = filtered / total if total > 0 else 0
        
        if filter_rate > 0.8:
            suggestions.append(f"🔴 Слишком высокая фильтрация QGate: {filter_rate:.1%}. Пороги слишком строгие.")
        elif filter_rate < 0.1:
            suggestions.append(f"🟡 Слишком низкая фильтрация QGate: {filter_rate:.1%}. Пороги слишком мягкие.")
        
        # Анализ по символам
        for symbol, stats in qgate_analysis['symbol_analysis'].items():
            if stats['total'] > 5:  # Только для символов с достаточным количеством данных
                filter_rate = stats['qgate_filter_rate']
                if filter_rate > 0.9:
                    suggestions.append(f"🔴 {symbol}: критично высокая фильтрация {filter_rate:.1%}")
                elif filter_rate < 0.1:
                    suggestions.append(f"🟡 {symbol}: очень низкая фильтрация {filter_rate:.1%}")
        
        # Анализ причин фильтрации
        if qgate_analysis['qgate_reasons']:
            most_common_reason = max(qgate_analysis['qgate_reasons'].items(), key=lambda x: x[1])
            suggestions.append(f"📊 Наиболее частая причина фильтрации: {most_common_reason[0]} ({most_common_reason[1]} раз)")
        
        # Рекомендации по порогам
        if 'q_values_summary' in qgate_analysis:
            qv = qgate_analysis['q_values_summary']
            suggestions.append(f"💡 Рекомендуемые пороги:")
            suggestions.append(f"   QGATE_MAXQ: {qv['max_q_mean'] * 0.7:.3f} (70% от среднего max_q)")
            suggestions.append(f"   QGATE_GAPQ: {qv['gap_q_mean'] * 0.5:.3f} (50% от среднего gap_q)")
        
        return suggestions
    
    def print_analysis_report(self, qgate_analysis: Dict[str, Any], trading_analysis: Dict[str, Any]):
        """Вывод отчета анализа"""
        print("\n" + "="*80)
        print("📊 ОТЧЕТ АНАЛИЗА QGATE И ПРЕДСКАЗАНИЙ")
        print("="*80)
        
        # Общая статистика
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего предсказаний: {qgate_analysis.get('total_predictions', 0)}")
        print(f"  Отфильтровано QGate: {qgate_analysis.get('qgate_filtered', 0)}")
        print(f"  Прошло QGate: {qgate_analysis.get('qgate_passed', 0)}")
        print(f"  Процент фильтрации: {qgate_analysis.get('qgate_filtered', 0) / max(qgate_analysis.get('total_predictions', 1), 1) * 100:.1f}%")
        
        # Q-values статистика
        if 'q_values_summary' in qgate_analysis:
            qv = qgate_analysis['q_values_summary']
            print(f"\n🎯 Q-VALUES СТАТИСТИКА:")
            print(f"  Средний max_q: {qv['max_q_mean']:.3f} ± {qv['max_q_std']:.3f}")
            print(f"  Диапазон max_q: [{qv['max_q_min']:.3f}, {qv['max_q_max']:.3f}]")
            print(f"  Средний gap_q: {qv['gap_q_mean']:.3f} ± {qv['gap_q_std']:.3f}")
            print(f"  Диапазон gap_q: [{qv['gap_q_min']:.3f}, {qv['gap_q_max']:.3f}]")
        
        # Уверенность
        if 'confidence_summary' in qgate_analysis:
            conf = qgate_analysis['confidence_summary']
            print(f"\n🎯 УВЕРЕННОСТЬ МОДЕЛИ:")
            print(f"  Средняя уверенность: {conf['mean']:.1f}% ± {conf['std']:.1f}%")
            print(f"  Диапазон: [{conf['min']:.1f}%, {conf['max']:.1f}%]")
        
        # Распределение действий
        print(f"\n🎯 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ:")
        for action, count in qgate_analysis.get('action_distribution', {}).items():
            percentage = count / max(qgate_analysis.get('total_predictions', 1), 1) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        # Анализ по символам
        print(f"\n🎯 АНАЛИЗ ПО СИМВОЛАМ:")
        for symbol, stats in qgate_analysis.get('symbol_analysis', {}).items():
            if stats['total'] > 0:
                filter_rate = stats['qgate_filter_rate']
                print(f"  {symbol}: {stats['total']} предсказаний, фильтрация {filter_rate:.1%}")
                if stats['avg_confidence']:
                    print(f"    Средняя уверенность: {stats['avg_confidence']:.1f}%")
                if stats['avg_max_q']:
                    print(f"    Средний max_q: {stats['avg_max_q']:.3f}")
        
        # Причины фильтрации
        if qgate_analysis.get('qgate_reasons'):
            print(f"\n🎯 ПРИЧИНЫ ФИЛЬТРАЦИИ QGATE:")
            for reason, count in sorted(qgate_analysis['qgate_reasons'].items(), key=lambda x: x[1], reverse=True):
                percentage = count / max(qgate_analysis.get('qgate_filtered', 1), 1) * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        # Торговая статистика
        if trading_analysis:
            print(f"\n📈 ТОРГОВАЯ СТАТИСТИКА:")
            print(f"  Всего сделок: {trading_analysis.get('total_trades', 0)}")
            print(f"  Успешных: {trading_analysis.get('successful_trades', 0)}")
            print(f"  Неудачных: {trading_analysis.get('failed_trades', 0)}")
            print(f"  Процент успеха: {trading_analysis.get('success_rate', 0) * 100:.1f}%")
            
            if 'pnl_summary' in trading_analysis:
                pnl = trading_analysis['pnl_summary']
                print(f"  Средний P&L: {pnl['mean']:.2f} ± {pnl['std']:.2f}")
                print(f"  Положительных сделок: {pnl['positive_count']}")
                print(f"  Отрицательных сделок: {pnl['negative_count']}")
        
        # Текущие пороги
        thresholds = self.get_current_qgate_thresholds()
        print(f"\n🎯 ТЕКУЩИЕ ПОРОГИ QGATE:")
        for key, value in thresholds.items():
            print(f"  {key}: {value}")
        
        # Рекомендации
        suggestions = self.suggest_qgate_improvements(qgate_analysis)
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    def close(self):
        """Закрытие соединения с БД"""
        self.session.close()

def main():
    """Основная функция"""
    print("🔍 АНАЛИЗАТОР QGATE И ПРЕДСКАЗАНИЙ")
    print("="*50)
    
    analyzer = QGateAnalyzer()
    
    try:
        # Анализ QGate
        qgate_analysis = analyzer.analyze_qgate_performance(days_back=7)
        
        # Анализ торговли
        trading_analysis = analyzer.analyze_trading_performance(days_back=7)
        
        # Вывод отчета
        analyzer.print_analysis_report(qgate_analysis, trading_analysis)
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()
