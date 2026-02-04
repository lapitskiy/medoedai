from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _fmt_pct(x: float) -> str:
    try:
        return f"{x:+.2f}%"
    except Exception:
        return "0.00%"


def build_semantic_snapshot(
    df_5m: pd.DataFrame,
    *,
    symbol: str,
    market_regime: str,
    market_regime_details: Optional[dict] = None,
    decision: Optional[str] = None,
    votes: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Converts OHLCV tail into compact, LLM-friendly summary (no raw arrays).
    Designed to be stable and cheap.
    """
    out: Dict[str, Any] = {
        "symbol": str(symbol),
        "timeframe": "5m",
        "decision": str(decision) if decision is not None else None,
        "market_regime": str(market_regime) if market_regime is not None else None,
        "regime_windows": [],
        "regime_labels": [],
    }
    if isinstance(market_regime_details, dict):
        out["regime_windows"] = list(market_regime_details.get("windows") or [])
        out["regime_labels"] = list(market_regime_details.get("labels") or [])
        out["regime_votes_map"] = market_regime_details.get("votes_map")
        out["regime_metrics"] = market_regime_details.get("metrics")
    if isinstance(votes, dict):
        out["ensemble_votes"] = {k: int(votes.get(k, 0)) for k in ("buy", "sell", "hold")}

    if df_5m is None or getattr(df_5m, "empty", False):
        return out

    df = df_5m.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            return out
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    last = _safe_float(close.iloc[-1], 0.0)
    out["last_price"] = last

    # Channel position on long window (2880 candles if possible).
    n_long = int(min(2880, len(df)))
    c_long = close.tail(n_long)
    lo = float(np.nanmin(c_long.values)) if len(c_long) else last
    hi = float(np.nanmax(c_long.values)) if len(c_long) else last
    out["range_long_n"] = n_long
    out["range_long_low"] = lo
    out["range_long_high"] = hi
    pos = 0.5
    try:
        if hi > lo:
            pos = float((last - lo) / (hi - lo))
            pos = float(np.clip(pos, 0.0, 1.0))
    except Exception:
        pos = 0.5
    out["price_position_0_1"] = float(pos)

    # Momentum: 15m (3 candles) and 1h (12 candles)
    def _chg(n: int) -> float:
        try:
            if len(close) <= n:
                return 0.0
            base = float(close.iloc[-(n + 1)])
            if base <= 0:
                return 0.0
            return float((last / base - 1.0) * 100.0)
        except Exception:
            return 0.0

    out["momentum_15m_pct"] = float(_chg(3))
    out["momentum_1h_pct"] = float(_chg(12))
    out["momentum_4h_pct"] = float(_chg(48))

    # Volatility proxy: std of returns over last 100
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tail_n = int(min(100, len(ret)))
    vol_std = float(np.nanstd(ret.tail(tail_n).values)) if tail_n > 1 else 0.0
    out["volatility_std_100"] = float(vol_std)

    # Volume spike: last vol / mean(60)
    n_v = int(min(60, len(vol)))
    v_mean = float(np.nanmean(vol.tail(n_v).values)) if n_v > 0 else 0.0
    v_last = float(vol.iloc[-1]) if len(vol) else 0.0
    out["volume_last"] = float(v_last)
    out["volume_mean_60"] = float(v_mean)
    out["volume_spike_x"] = float((v_last / v_mean) if v_mean > 0 else 1.0)

    # A simple RSI(14) implementation (no external deps)
    rsi = None
    try:
        n = 14
        if len(close) >= n + 2:
            delta = close.diff().fillna(0.0)
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            avg_gain = gain.rolling(n, min_periods=n).mean().iloc[-1]
            avg_loss = loss.rolling(n, min_periods=n).mean().iloc[-1]
            if avg_loss and float(avg_loss) > 0:
                rs = float(avg_gain) / float(avg_loss)
                rsi = 100.0 - (100.0 / (1.0 + rs))
            else:
                rsi = 100.0
    except Exception:
        rsi = None
    out["rsi14"] = float(rsi) if rsi is not None else None

    # A short, human-ish summary string (for logs/UI)
    out["summary"] = (
        f"price={last:.6f} pos={pos:.2f} "
        f"mom15m={_fmt_pct(out['momentum_15m_pct'])} mom1h={_fmt_pct(out['momentum_1h_pct'])} "
        f"vol_std100={vol_std:.4f} vol_spike={out['volume_spike_x']:.2f}x"
    )

    return out


def build_gigachat_prompt(snapshot: Dict[str, Any]) -> str:
    """
    Build a compact prompt from semantic snapshot. Keep it short to avoid token bloat.
    """
    sym = snapshot.get("symbol")
    dec = snapshot.get("decision")
    mr = snapshot.get("market_regime")
    rw = snapshot.get("regime_windows") or []
    rl = snapshot.get("regime_labels") or []
    regimes = ", ".join([f"{w}={l}" for w, l in zip(rw, rl)]) if rw and rl else None
    txt = [
        f"Symbol: {sym}",
        f"Decision from models: {str(dec).upper() if dec else 'N/A'}",
        f"Market regime: {mr}",
    ]
    if regimes:
        txt.append(f"Regimes: {regimes}")
    txt.append(f"Snapshot: {snapshot.get('summary')}")
    txt.append(
        "Return ONLY valid JSON (no markdown, no extra text). "
        "Schema: {\"decision\":\"BUY|SELL|HOLD\",\"confidence\":0.0-1.0,"
        "\"reason\":\"short\",\"risk\":\"short\"}."
    )
    return "\n".join([t for t in txt if t])

