from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request  # type: ignore

atr_bp = Blueprint("atr", __name__)


@atr_bp.get("/analitika/atr")
def atr_page():
    return render_template("analitika/atr.html")


@atr_bp.get("/api/atr/1h")
def atr_1h_api():
    """
    Возвращает ATR по 1h свечам (по умолчанию строго из БД).

    Query:
      symbols: "BTCUSDT,ETHUSDT" или повторяющиеся symbols=BTCUSDT&symbols=ETHUSDT
      length: период ATR (default 21)
      db_only: 1/0 (default 1) - если 1, берём свечи строго из Postgres, без докачки с биржи
    """
    try:
        length = request.args.get("length", 21, type=int)
        db_only = request.args.get("db_only", 1, type=int) == 1

        symbols_list: list[str] = []
        # 1) repeated args: symbols=BTCUSDT&symbols=ETHUSDT
        for s in request.args.getlist("symbols"):
            if s:
                symbols_list.append(str(s).strip())
        # 2) csv: symbols=BTCUSDT,ETHUSDT
        if not symbols_list:
            raw = (request.args.get("symbols") or "").strip()
            if raw:
                symbols_list = [p.strip() for p in raw.split(",") if p.strip()]

        # fallback defaults (как в сайдбаре)
        if not symbols_list:
            symbols_list = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TONUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]

        from utils.indicators import get_atr_1h

        rows = []
        for sym in symbols_list:
            sym_u = str(sym).strip().upper().replace("/", "")
            if not sym_u:
                continue
            try:
                atr_abs, atr_norm, last_close = get_atr_1h(sym_u, length=length, db_only=db_only)
                rows.append(
                    {
                        "symbol": sym_u,
                        "atr_abs": atr_abs,
                        "atr_norm": atr_norm,
                        "last_close": last_close,
                        "ok": True,
                    }
                )
            except Exception as e:
                rows.append({"symbol": sym_u, "ok": False, "error": str(e)[:200]})

        return jsonify(
            {
                "success": True,
                "timeframe": "1h",
                "length": int(length),
                "db_only": bool(db_only),
                "rows": rows,
                "total": len(rows),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

