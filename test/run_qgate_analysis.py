#!/usr/bin/env python3
"""
Быстрый запуск анализа QGate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qgate_analyzer import QGateAnalyzer

def quick_analysis():
    """Быстрый анализ QGate"""
    print("🚀 БЫСТРЫЙ АНАЛИЗ QGATE")
    print("=" * 40)
    
    analyzer = QGateAnalyzer()
    
    try:
        # Анализ за последние 3 дня для быстроты
        qgate_analysis = analyzer.analyze_qgate_performance(days_back=3)
        trading_analysis = analyzer.analyze_trading_performance(days_back=3)
        
        # Вывод отчета
        analyzer.print_analysis_report(qgate_analysis, trading_analysis)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    quick_analysis()
