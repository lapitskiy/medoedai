"""
Модуль анализа результатов валидации CNN моделей
Предоставляет детальный анализ ошибок и рекомендации по улучшению
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ValidationAnalyzer:
    """Анализатор результатов валидации CNN моделей"""
    
    def __init__(self):
        """Инициализация анализатора"""
        self.accuracy_thresholds = {
            'excellent': 0.75,  # Отличная точность
            'good': 0.65,       # Хорошая точность
            'acceptable': 0.55, # Приемлемая точность
            'poor': 0.45        # Плохая точность
        }
        
        self.confidence_thresholds = {
            'high': 0.8,        # Высокая уверенность
            'medium': 0.6,      # Средняя уверенность
            'low': 0.4          # Низкая уверенность
        }
    
    def analyze_validation_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Комплексный анализ результатов валидации
        
        Args:
            validation_results: Результаты валидации модели
            
        Returns:
            Детальный анализ с рекомендациями
        """
        try:
            logger.info("🔍 Начинаем анализ результатов валидации")
            
            # Базовый анализ
            basic_analysis = self._analyze_basic_metrics(validation_results)
            
            # Анализ по символам
            symbol_analysis = self._analyze_symbol_performance(validation_results)
            
            # Анализ ошибок
            error_analysis = self._analyze_errors(validation_results)
            
            # Анализ паттернов
            pattern_analysis = self._analyze_patterns(validation_results)
            
            # Генерация рекомендаций
            recommendations = self._generate_recommendations(
                basic_analysis, symbol_analysis, error_analysis, pattern_analysis
            )
            
            # Оценка готовности к продакшену
            production_readiness = self._assess_production_readiness(
                basic_analysis, symbol_analysis
            )
            
            analysis_result = {
                'basic_analysis': basic_analysis,
                'symbol_analysis': symbol_analysis,
                'error_analysis': error_analysis,
                'pattern_analysis': pattern_analysis,
                'recommendations': recommendations,
                'production_readiness': production_readiness,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Анализ результатов валидации завершен")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа результатов: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _analyze_basic_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ базовых метрик"""
        try:
            overall_accuracy = results.get('overall_accuracy', 0)
            total_samples = results.get('total_samples', 0)
            symbol_results = results.get('symbol_results', [])
            
            # Вычисляем статистики
            accuracies = [sr.get('accuracy', 0) for sr in symbol_results if sr.get('success', False)]
            confidences = [sr.get('avg_confidence', 0) for sr in symbol_results if sr.get('success', False)]
            
            analysis = {
                'overall_accuracy': overall_accuracy,
                'accuracy_grade': self._grade_accuracy(overall_accuracy),
                'total_samples': total_samples,
                'symbols_tested': len(symbol_results),
                'successful_symbols': len([sr for sr in symbol_results if sr.get('success', False)]),
                'accuracy_stats': {
                    'mean': np.mean(accuracies) if accuracies else 0,
                    'std': np.std(accuracies) if accuracies else 0,
                    'min': np.min(accuracies) if accuracies else 0,
                    'max': np.max(accuracies) if accuracies else 0
                },
                'confidence_stats': {
                    'mean': np.mean(confidences) if confidences else 0,
                    'std': np.std(confidences) if confidences else 0,
                    'min': np.min(confidences) if confidences else 0,
                    'max': np.max(confidences) if confidences else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа базовых метрик: {str(e)}")
            return {}
    
    def _analyze_symbol_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ производительности по символам"""
        try:
            symbol_results = results.get('symbol_results', [])
            
            symbol_analysis = {
                'best_performing': None,
                'worst_performing': None,
                'consistent_performers': [],
                'inconsistent_performers': [],
                'symbol_details': {}
            }
            
            if not symbol_results:
                return symbol_analysis
            
            # Анализируем каждый символ
            for symbol_result in symbol_results:
                if not symbol_result.get('success', False):
                    continue
                
                symbol = symbol_result.get('symbol', 'Unknown')
                accuracy = symbol_result.get('accuracy', 0)
                confidence = symbol_result.get('avg_confidence', 0)
                samples = symbol_result.get('samples_tested', 0)
                
                symbol_analysis['symbol_details'][symbol] = {
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'samples': samples,
                    'grade': self._grade_accuracy(accuracy),
                    'confidence_level': self._grade_confidence(confidence),
                    'reliability': self._assess_reliability(accuracy, confidence, samples)
                }
            
            # Находим лучший и худший символы
            if symbol_analysis['symbol_details']:
                best_symbol = max(
                    symbol_analysis['symbol_details'].items(),
                    key=lambda x: x[1]['accuracy']
                )
                worst_symbol = min(
                    symbol_analysis['symbol_details'].items(),
                    key=lambda x: x[1]['accuracy']
                )
                
                symbol_analysis['best_performing'] = {
                    'symbol': best_symbol[0],
                    'accuracy': best_symbol[1]['accuracy'],
                    'grade': best_symbol[1]['grade']
                }
                
                symbol_analysis['worst_performing'] = {
                    'symbol': worst_symbol[0],
                    'accuracy': worst_symbol[1]['accuracy'],
                    'grade': worst_symbol[1]['grade']
                }
                
                # Классифицируем символы по стабильности
                for symbol, details in symbol_analysis['symbol_details'].items():
                    if details['reliability'] == 'high':
                        symbol_analysis['consistent_performers'].append(symbol)
                    elif details['reliability'] == 'low':
                        symbol_analysis['inconsistent_performers'].append(symbol)
            
            return symbol_analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа производительности по символам: {str(e)}")
            return {}
    
    def _analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ ошибок модели"""
        try:
            symbol_results = results.get('symbol_results', [])
            
            error_analysis = {
                'common_error_patterns': [],
                'error_distribution': {},
                'confidence_vs_accuracy': {},
                'samples_vs_accuracy': {}
            }
            
            # Анализируем ошибки по символам
            for symbol_result in symbol_results:
                if not symbol_result.get('success', False):
                    continue
                
                symbol = symbol_result.get('symbol', 'Unknown')
                accuracy = symbol_result.get('accuracy', 0)
                confidence = symbol_result.get('avg_confidence', 0)
                samples = symbol_result.get('samples_tested', 0)
                error_analysis_data = symbol_result.get('error_analysis', {})
                
                # Анализ распределения ошибок
                error_rate = 1 - accuracy
                error_analysis['error_distribution'][symbol] = {
                    'error_rate': error_rate,
                    'error_type': self._classify_error_type(error_rate, confidence)
                }
                
                # Корреляция уверенности и точности
                error_analysis['confidence_vs_accuracy'][symbol] = {
                    'confidence': confidence,
                    'accuracy': accuracy,
                    'correlation': self._assess_confidence_accuracy_correlation(confidence, accuracy)
                }
                
                # Корреляция количества образцов и точности
                error_analysis['samples_vs_accuracy'][symbol] = {
                    'samples': samples,
                    'accuracy': accuracy,
                    'sufficiency': self._assess_sample_sufficiency(samples, accuracy)
                }
            
            # Выявляем общие паттерны ошибок
            error_analysis['common_error_patterns'] = self._identify_common_error_patterns(
                error_analysis['error_distribution']
            )
            
            return error_analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа ошибок: {str(e)}")
            return {}
    
    def _analyze_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ обнаруженных паттернов"""
        try:
            symbol_results = results.get('symbol_results', [])
            
            pattern_analysis = {
                'pattern_frequency': {},
                'pattern_effectiveness': {},
                'pattern_coverage': {},
                'recommended_patterns': []
            }
            
            # Собираем все обнаруженные паттерны
            all_patterns = []
            for symbol_result in symbol_results:
                if not symbol_result.get('success', False):
                    continue
                
                patterns = symbol_result.get('patterns_detected', [])
                all_patterns.extend(patterns)
            
            # Анализируем частоту паттернов
            if all_patterns:
                from collections import Counter
                pattern_counts = Counter(all_patterns)
                pattern_analysis['pattern_frequency'] = dict(pattern_counts)
                
                # Определяем наиболее эффективные паттерны
                pattern_analysis['recommended_patterns'] = self._identify_effective_patterns(
                    pattern_counts, symbol_results
                )
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов: {str(e)}")
            return {}
    
    def _generate_recommendations(self, basic_analysis: Dict, symbol_analysis: Dict, 
                                error_analysis: Dict, pattern_analysis: Dict) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        try:
            recommendations = []
            
            # Рекомендации на основе общей точности
            overall_accuracy = basic_analysis.get('overall_accuracy', 0)
            if overall_accuracy < self.accuracy_thresholds['acceptable']:
                recommendations.append(
                    "🔴 КРИТИЧНО: Общая точность модели ниже приемлемого уровня. "
                    "Требуется переобучение с дополнительными данными."
                )
            elif overall_accuracy < self.accuracy_thresholds['good']:
                recommendations.append(
                    "🟡 ВНИМАНИЕ: Точность модели приемлема, но может быть улучшена. "
                    "Рекомендуется дообучение на проблемных символах."
                )
            else:
                recommendations.append(
                    "🟢 ОТЛИЧНО: Модель показывает хорошую обобщающую способность. "
                    "Готова к использованию в торговле."
                )
            
            # Рекомендации на основе стабильности по символам
            inconsistent_symbols = symbol_analysis.get('inconsistent_performers', [])
            if inconsistent_symbols:
                recommendations.append(
                    f"⚠️ Нестабильная работа на символах: {', '.join(inconsistent_symbols)}. "
                    "Рекомендуется дополнительное обучение на этих активах."
                )
            
            # Рекомендации на основе анализа ошибок
            high_error_symbols = [
                symbol for symbol, data in error_analysis.get('error_distribution', {}).items()
                if data.get('error_rate', 0) > 0.4
            ]
            if high_error_symbols:
                recommendations.append(
                    f"🔧 Высокий уровень ошибок на символах: {', '.join(high_error_symbols)}. "
                    "Рассмотрите возможность специализированного обучения для этих активов."
                )
            
            # Рекомендации по паттернам
            recommended_patterns = pattern_analysis.get('recommended_patterns', [])
            if recommended_patterns:
                recommendations.append(
                    f"📈 Эффективные паттерны: {', '.join(recommended_patterns[:3])}. "
                    "Рекомендуется фокус на этих типах паттернов при обучении."
                )
            
            # Рекомендации по количеству данных
            total_samples = basic_analysis.get('total_samples', 0)
            if total_samples < 1000:
                recommendations.append(
                    "📊 Недостаточно тестовых данных. Рекомендуется увеличить период тестирования "
                    "или добавить больше символов для более надежной оценки."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации рекомендаций: {str(e)}")
            return ["Ошибка при генерации рекомендаций"]
    
    def _assess_production_readiness(self, basic_analysis: Dict, symbol_analysis: Dict) -> Dict[str, Any]:
        """Оценка готовности модели к продакшену"""
        try:
            overall_accuracy = basic_analysis.get('overall_accuracy', 0)
            successful_symbols = basic_analysis.get('successful_symbols', 0)
            total_symbols = basic_analysis.get('symbols_tested', 0)
            inconsistent_performers = len(symbol_analysis.get('inconsistent_performers', []))
            
            # Критерии готовности
            accuracy_ready = overall_accuracy >= self.accuracy_thresholds['good']
            stability_ready = inconsistent_performers <= total_symbols * 0.3  # Не более 30% нестабильных
            coverage_ready = successful_symbols >= total_symbols * 0.8  # Не менее 80% успешных тестов
            
            # Общая оценка
            readiness_score = sum([accuracy_ready, stability_ready, coverage_ready]) / 3
            
            if readiness_score >= 0.8:
                readiness_level = "production_ready"
                readiness_message = "✅ Модель готова к использованию в продакшене"
            elif readiness_score >= 0.6:
                readiness_level = "near_ready"
                readiness_message = "⚠️ Модель близка к готовности, требуется небольшая доработка"
            elif readiness_score >= 0.4:
                readiness_level = "needs_improvement"
                readiness_message = "🔧 Модель требует значительных улучшений"
            else:
                readiness_level = "not_ready"
                readiness_message = "❌ Модель не готова к продакшену"
            
            return {
                'readiness_level': readiness_level,
                'readiness_score': readiness_score,
                'readiness_message': readiness_message,
                'criteria': {
                    'accuracy_ready': accuracy_ready,
                    'stability_ready': stability_ready,
                    'coverage_ready': coverage_ready
                },
                'overall_accuracy': overall_accuracy,
                'inconsistent_symbols_count': inconsistent_performers,
                'success_rate': successful_symbols / total_symbols if total_symbols > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки готовности к продакшену: {str(e)}")
            return {
                'readiness_level': 'unknown',
                'readiness_score': 0,
                'readiness_message': 'Ошибка при оценке готовности'
            }
    
    # Вспомогательные методы
    
    def _grade_accuracy(self, accuracy: float) -> str:
        """Оценка точности"""
        if accuracy >= self.accuracy_thresholds['excellent']:
            return 'excellent'
        elif accuracy >= self.accuracy_thresholds['good']:
            return 'good'
        elif accuracy >= self.accuracy_thresholds['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def _grade_confidence(self, confidence: float) -> str:
        """Оценка уверенности"""
        if confidence >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_reliability(self, accuracy: float, confidence: float, samples: int) -> str:
        """Оценка надежности"""
        if accuracy >= 0.7 and confidence >= 0.7 and samples >= 100:
            return 'high'
        elif accuracy >= 0.6 and confidence >= 0.6 and samples >= 50:
            return 'medium'
        else:
            return 'low'
    
    def _classify_error_type(self, error_rate: float, confidence: float) -> str:
        """Классификация типа ошибок"""
        if error_rate > 0.4:
            return 'high_error'
        elif error_rate > 0.3:
            return 'medium_error'
        elif confidence < 0.5:
            return 'low_confidence'
        else:
            return 'acceptable'
    
    def _assess_confidence_accuracy_correlation(self, confidence: float, accuracy: float) -> str:
        """Оценка корреляции уверенности и точности"""
        if abs(confidence - accuracy) < 0.1:
            return 'well_calibrated'
        elif confidence > accuracy + 0.2:
            return 'overconfident'
        elif confidence < accuracy - 0.2:
            return 'underconfident'
        else:
            return 'moderate_correlation'
    
    def _assess_sample_sufficiency(self, samples: int, accuracy: float) -> str:
        """Оценка достаточности выборки"""
        if samples >= 500:
            return 'sufficient'
        elif samples >= 100:
            return 'moderate'
        else:
            return 'insufficient'
    
    def _identify_common_error_patterns(self, error_distribution: Dict) -> List[str]:
        """Выявление общих паттернов ошибок"""
        patterns = []
        
        high_error_symbols = [
            symbol for symbol, data in error_distribution.items()
            if data.get('error_rate', 0) > 0.4
        ]
        
        if high_error_symbols:
            patterns.append(f"Высокие ошибки на символах: {', '.join(high_error_symbols)}")
        
        return patterns
    
    def _identify_effective_patterns(self, pattern_counts: Dict, symbol_results: List) -> List[str]:
        """Выявление эффективных паттернов"""
        # Упрощенная логика - возвращаем наиболее частые паттерны
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:5]]


def analyze_validation_results(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Удобная функция для анализа результатов валидации
    
    Args:
        validation_results: Результаты валидации модели
        
    Returns:
        Детальный анализ с рекомендациями
    """
    analyzer = ValidationAnalyzer()
    return analyzer.analyze_validation_results(validation_results)
