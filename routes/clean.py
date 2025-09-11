from flask import Blueprint, jsonify
from utils.db_utils import clean_ohlcv_data, delete_ohlcv_for_symbol_timeframe

clean_bp = Blueprint('clean', __name__)

@clean_bp.post('/clean_data')
def clean_data():
    timeframes_to_clean = ['5m', '15m', '1h']
    symbol_name = 'BTCUSDT'

    max_close_change_percent = 15.0
    max_hl_range_percent = 20.0
    volume_multiplier = 10.0

    results = []
    for tf in timeframes_to_clean:
        try:
            result = clean_ohlcv_data(tf, symbol_name,
                                      max_close_change_percent,
                                      max_hl_range_percent,
                                      volume_multiplier)
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "message": f"Ошибка при очистке {tf} для {symbol_name}: {str(e)}"})

    return jsonify({'status': 'Очистка данных завершена для всех указанных таймфреймов.', 'results': results})


@clean_bp.post('/clean_db')
def clean_db():
    timeframes_to_clean = '5m'
    symbol_name = 'BTCUSDT'

    results = []
    try:
        delete_ohlcv_for_symbol_timeframe('BTCUSDT', timeframes_to_clean)
    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка при очистке для {symbol_name}: {str(e)}"})

    return jsonify({'status': 'Очистка базы от всех свечей завершена указанных таймфреймов.', 'results': results})


