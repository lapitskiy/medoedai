from tasks import celery  # type: ignore


@celery.task(name='optimize_trading_params', bind=True)
def optimize_trading_params(self, payload: dict):
    """
    Черновой таск оптимизации параметров торговли.
    Реальную симуляцию/бэктест добавим позже.
    """
    grid = (payload or {}).get('grid') or {}
    # TODO: заменить на реальную оптимизацию
    return {
        'best': {'T1': 0.30, 'T2': 0.60, 'position_frac': 0.20, 'metric': 'winrate', 'value': 0.57},
        'searched': grid,
        'params': payload,
    }


