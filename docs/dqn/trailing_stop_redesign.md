# Trailing Stop Redesign (февраль 2026)

## Что было

- **Trailing stop** работал только при `epsilon <= 0.2` (exploitation) — при обучении агент не видел его эффекта.
- Нужно было **3 бара подряд** с drawdown > порога (`trailing_stop_counter >= 3`) — медленная реакция.
- Проверка по **Close** свечи, а не по Low.
- **Reward** при trailing exit: штраф `-0.03` **всегда** (даже если сделка в плюсе).
- Фиксированный SL = `-3%`, TP = `+1%` — часто срабатывали **раньше** trailing-а, делая его бесполезным.
- Агент **не видел** расстояние до trailing stop в state.

## Что сделано

### 1. Trailing — основной механизм выхода (всегда включён)
- Убрана проверка `epsilon <= 0.2` — trailing работает на **всех** фазах обучения.
- Убран `trailing_stop_counter` — срабатывает за **1 бар**.
- Проверка по **Low** свечи (ближе к реальности): `low <= trailing_level`.
- `trailing_level = max_price_during_hold * (1 - thr_trail)`.
- Порог: `thr_trail = ATR * atr_trail_mult` (cfg: `atr_trail_mult`, по умолчанию 1.5).

### 2. Reward при trailing exit
- Убран безусловный штраф `-0.03`.
- Если trailing закрывает сделку **в плюсе**: бонус `+0.02`.
- PnL сделки считается через `_force_sell` как обычно.

### 3. SL/TP — аварийные механизмы
- **SL** расширен до `-5%` (базовый) — срабатывает только при резком движении, когда trailing не успевает.
- **TP** расширен до `+8%` (базовый) — потолок, если trailing не подтянулся.
- **TP можно отключить**: `use_fixed_tp = False` в cfg → только trailing закрывает в плюсе.
- Лимиты динамических параметров: SL от -3% до -10%, TP от 3% до 20%.

### 4. distance_to_trailing в state
- В observation добавлена фича: `(current_price - trailing_level) / current_price`.
- Если нет позиции — `0.0`.
- `observation_space_shape` увеличен на +1 (было +2 за balance/crypto, стало +3).

### 5. Новые параметры в gconfig.py
| Параметр | Тип | Дефолт | Описание |
|----------|-----|--------|----------|
| `atr_trail_mult` | float | 1.5 | ATR * mult = порог trailing (тюнить для гипотез) |
| `trailing_activation` | float | 0.0 | % профита от входа для активации trailing (0 = сразу) |
| `use_fixed_tp` | bool | True | False = отключить фиксированный TP |

## Иерархия выходов

```
1. Trailing stop (основной)     — ATR * mult от пика, по Low, 1 бар
2. Агент (действие SELL)        — агент сам решает продать
3. Фиксированный SL (аварийный) — -5% от входа, защита от гэпа
4. Фиксированный TP (опционал)  — +8%, можно отключить (use_fixed_tp=False)
5. Timeout                      — конец эпизода
```

## Grid search по atr_trail_mult

`atr_trail_mult` добавлен в grid обучения для поиска гипотез.

### Цепочка проброса
1. **UI**: `templates/models.html` — поля `gridTrailFrom/To/Step`
2. **Route**: `routes/training.py` — парсинг `trail_from/to/step`, добавление в `risk_management`
3. **Celery → Trainer**: `v_train_model_optimized.py` — `cfg.atr_trail_mult = rm2['atr_trail_mult']`
4. **Сохранение**: `gym_snapshot.risk_management.atr_trail_mult` в `train_result.pkl`
5. **CSV export**: `routes/system_models.py` — колонка `atr_trail_mult` + `sell_trailing_pct`
6. **Analitika**: `templates/analitika/index.html` — отображение `trail_mult`

### Рекомендуемые значения для grid
- From: 0.8, To: 2.5, Step: 0.5 → [0.8, 1.3, 1.8, 2.3]

## Файлы

- `envs/dqn_model/gym/gconfig.py` — параметры
- `envs/dqn_model/gym/crypto_trading_env_optimized.py` — логика trailing, SL/TP, state
- `routes/training.py` — grid route handler
- `agents/vdqn/v_train_model_optimized.py` — проброс override + gym_snapshot
- `routes/system_models.py` — CSV export гипотез
- `templates/models.html` — UI grid формы
- `templates/analitika/index.html` — отображение результатов