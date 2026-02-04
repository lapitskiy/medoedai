# RL trading (DQN) — strategy + tuning reference

## Principles (avoid reward hacking)
- Primary objective is **net PnL after fees**. Reward must not create “free money” for actions like BUY/ENTER or idle HOLD.
- **No “activity reward”**: do not add extra positive reward just because the agent traded or because time passed.
- Holding a position for hours/days should be incentivized **only through mark-to-market PnL** (unrealized/realized), not via “patience bonus”.

## What was fixed/changed
- **Eval winrate**: baseline trade counter is taken **after** `env.reset()` so eval episodes compute winrate on trades from that episode (no more accidental `0.0` due to reset clearing lists).
- **Entry reward removed**:
  - LONG (`CryptoTradingEnvOptimized`): removed `base_reward/confidence_bonus` overwrite; entry reward is now ~0 (commission penalty only, optional *negative-only* volume penalty).
  - SHORT (`CryptoTradingEnvShort`): removed `0.03 + confidence_bonus` entry reward; entry reward is now commission penalty only.
  - Tianshou env (`crypto_trading_env_tainshou.py`): same entry bonus removed to keep behavior consistent.
- **No idle bonuses**:
  - Removed `reward = 0.001` for HOLD without position (LONG + Tianshou) and removed `+0.001` for HOLD without position (SHORT).
  - Removed “patience bonus” for holding a position (`+0.001` after long hold).
  - Removed “activity reward” (`base_activity_reward`) that was adding extra reward just for non-HOLD actions.
- **Bad-regime entries**:
  - `use_state_action_mask` default is now **True** (env forbids BUY when `market_state != NORMAL`).
  - Buy filters no longer fully bypass when `eps > 0.8` (still uses lenient thresholds, but keeps dead-market protection).
- **Encoder stability**:
  - Default `encoder_lr_scale` reduced to `0.03` to reduce encoder drift while Q-head is learning.

## Reward design (current intent)
### Entry (BUY / ENTER_SHORT)
- Reward on entry should be **≈ 0**.
- Only systematic component is **fee penalty** (small negative).
- Any shaping around entry must be **penalty-only** (e.g., discourage very low volume), no positive bonuses.

### Hold with position
- Reward should track **unrealized PnL** (mark-to-market).
- This allows “hold half-day / day” to be learned naturally when trend persists.

### Exit (SELL / COVER)
- Reward should be primarily a function of **net PnL after fees**.
- Micro-trade / churn control:
  - In LONG env, micro-profit threshold is controlled by:
    - `micro_profit_tau_mult` (default `4.0`)
    - `micro_profit_tau_min`  (default `0.003`)
  - These can be overridden via `symbol override -> gym_config` (allowlist).
  - Idea: if net PnL is below the “fee-like” threshold, treat it as churn and do not reward it.

## Risk management knobs (what to tune)
- **Percent SL/TP**: `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`
- **ATR SL/TP/Trailing** (if enabled):
  - `use_atr_stop=True`
  - `atr_sl_mult`, `atr_tp_mult`, `atr_trail_mult`

Rule of thumb: tune so expectancy is positive:
- target either **winrate ≥ ~45%** *or* (avg_loss ↓ / avg_profit ↑)
- reduce churn: increase micro-profit threshold; avoid rewards that encourage frequent closes.

## Metrics to watch (sanity + quality)
- **eval.winrate**: greedy evaluation (ε=0), should be non-zero when trades exist.
- **winrate (train)**: per-episode, plus exploit-only (`eps <= threshold`).
- **avg_roi / total_profit**: must improve together with reward changes (avoid reward-only improvements).
- **churn**: trades/episode, close frequency, buy_accept_rate.
- **mask impact**: fraction of steps in non-NORMAL regimes; masked BUY rate.

## Short run checklist (200–400 episodes)
- Confirm:
  - eval.winrate is stable (not stuck at 0.0)
  - entry reward is not positive on average
  - HOLD(no position) reward average ~0 (not +)
  - churn decreases vs baseline
- Only after that run a full training.

