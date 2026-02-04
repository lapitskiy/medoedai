# XGB: метрики, coverage и как их интерпретировать

## Термины

- **Task `directional`**: 3-классовая задача `hold/buy/sell` на будущий return (зависит от `horizon_steps` и `threshold`).
- **Task `entry_*` / `exit_*`**: бинарная задача (`0=hold`, `1=enter/exit`) на позиционную разметку, зависящую от `max_hold_steps`, `min_profit`, (и для `exit_*` ещё `label_delta`).

## Главная идея: coverage vs качество

На странице `/analitika/xgb` есть две похожие метрики:

- **`y_non_hold`** = `y_non_hold_rate_val` — доля `y!=0` на валидации, то есть *насколько часто label говорит “не hold”*.
- **`pred_non_hold`** = `pred_non_hold_rate_val` — доля `pred!=0` на валидации, то есть *насколько часто модель “сигналит”*.

Важно: `pred_non_hold` сам по себе **не метрика качества**, это метрика “активности”. Она почти всегда “следует” за тем, как часто класс 1/2 встречается в label (особенно если decision rule — argmax).

## Какие метрики смотреть по задачам

### `entry_*` / `exit_*` (binary)

Смотри в первую очередь метрики класса `1`:

- **`f1(1)`** (`f1_val[1]`) — баланс precision/recall по классу 1.
- **`prec(1)`** (`precision_val[1]`) — насколько “чистые” входы/выходы (меньше false positive).
- **`recall(1)`** (`recall_val[1]`) — насколько часто ловишь все возможности.

И обязательно рядом:

- **`y_non_hold`**: если близко к 0 — класс 1 слишком редкий (мало данных для обучения),
  если близко к 1 — класс 1 слишком широкий (“входить почти всегда”).
- **`pred_non_hold`**: фактическая частота сигналов модели.

### `directional` (3-class)

- **`f1_buy_sell`**: средний F1 по классам buy/sell (вне hold).
- `pred_non_hold` тоже полезен как “coverage” сигналов (buy/sell), но качество смотри через `f1_buy_sell`.

## Почему `val_acc` может обманывать

При сильном дисбалансе (например, `y_non_hold` очень маленький) модель, которая почти всегда предсказывает `0`,
будет иметь высокий `val_acc`, но при этом плохую полезность.

## Где это берётся в коде (для ориентира)

- `y_non_hold_rate_val` и `pred_non_hold_rate_val` считаются в `agents/xgb/trainer.py`.
- Разметка `entry_*` зависит от `min_profit` и `max_hold_steps` (и fee), см. `agents/xgb/features.py`.

## Grid auto “под цель” (редкие сделки)

В `entry/exit` grid можно включить жёсткие фильтры при выборе `grid_final` через env:

- `XGB_GRID_TRADING_FILTER=1`
- `XGB_GRID_TARGET_TRADES_PER_MONTH` (например 5 или 10)
- `XGB_GRID_BARS_PER_MONTH=8640` (для 5m)

Подробности — в `goal_prompt_5pct_month.md`.
