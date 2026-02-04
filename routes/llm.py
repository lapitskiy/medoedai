from __future__ import annotations

from flask import Blueprint, jsonify, request

from utils.gigachat_client import GigaChatClient, load_gigachat_config
from utils.gigachat_features import build_gigachat_prompt, build_semantic_snapshot
from utils.gigachat_settings import ensure_gigachat_settings
from utils.settings_store import get_setting_value, upsert_setting


llm_bp = Blueprint("llm", __name__)


def _mask_secret(v: str | None) -> str:
    try:
        s = (v or "").strip()
        if not s:
            return ""
        if len(s) <= 10:
            return "***"
        return f"{s[:6]}…{s[-4:]}"
    except Exception:
        return "***"


@llm_bp.get("/api/llm/gigachat/config")
def gigachat_get_config():
    try:
        ensure_gigachat_settings()
        cfg = load_gigachat_config()
        return jsonify(
            {
                "success": True,
                "config": {
                    "enabled": bool(cfg.enabled),
                    "only_on_buy": bool(cfg.only_on_buy),
                    "auth_type": cfg.auth_type,
                    "oauth_url": cfg.oauth_url,
                    "oauth_scope": cfg.oauth_scope,
                    "chat_url": cfg.chat_url,
                    "model": cfg.model,
                    "timeout_sec": cfg.timeout_sec,
                    "temperature": cfg.temperature,
                    "max_tokens": cfg.max_tokens,
                    "verify_ssl": bool(cfg.verify_ssl),
                    "system_prompt": cfg.system_prompt,
                    "credentials_masked": _mask_secret(get_setting_value("llm", "gigachat", "CREDENTIALS")),
                    "has_credentials": bool((get_setting_value("llm", "gigachat", "CREDENTIALS") or "").strip()),
                    "has_access_token": bool((get_setting_value("llm", "gigachat", "ACCESS_TOKEN") or "").strip()),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@llm_bp.post("/api/llm/gigachat/config")
def gigachat_set_config():
    """
    Saves config into Postgres (app_settings).
    Secret fields are NOT returned back.
    """
    try:
        ensure_gigachat_settings()
        data = request.get_json(silent=True) or {}

        def _set(key: str, value, *, value_type: str, secret: bool, label: str, description: str):
            if value is None:
                return
            upsert_setting(
                scope="llm",
                group="gigachat",
                key=key,
                value_type=value_type,
                label=label,
                description=description,
                is_secret=bool(secret),
                value=str(value),
            )

        _set("ENABLED", data.get("enabled"), value_type="bool", secret=False, label="GigaChat enabled", description="Enable/disable GigaChat calls in trading pipeline.")
        _set("ONLY_ON_BUY", data.get("only_on_buy"), value_type="bool", secret=False, label="Only on BUY", description="Call GigaChat only when final decision is BUY.")
        _set("AUTH_TYPE", data.get("auth_type"), value_type="string", secret=False, label="Auth type", description="oauth or bearer")
        _set("OAUTH_URL", data.get("oauth_url"), value_type="string", secret=False, label="OAuth URL", description="Token endpoint URL.")
        _set("OAUTH_SCOPE", data.get("oauth_scope"), value_type="string", secret=False, label="OAuth scope", description="Scope for token request.")
        _set("CHAT_URL", data.get("chat_url"), value_type="string", secret=False, label="Chat completions URL", description="Chat completions endpoint URL.")
        _set("MODEL", data.get("model"), value_type="string", secret=False, label="Model", description="Model name.")
        _set("TIMEOUT_SEC", data.get("timeout_sec"), value_type="number", secret=False, label="Timeout (sec)", description="HTTP timeout for GigaChat requests.")
        _set("TEMPERATURE", data.get("temperature"), value_type="number", secret=False, label="Temperature", description="Sampling temperature.")
        _set("MAX_TOKENS", data.get("max_tokens"), value_type="number", secret=False, label="Max tokens", description="Max tokens in completion.")
        _set("VERIFY_SSL", data.get("verify_ssl"), value_type="bool", secret=False, label="Verify SSL", description="Enable SSL cert verification.")
        _set("SYSTEM_PROMPT", data.get("system_prompt"), value_type="string", secret=False, label="System prompt", description="System instruction for GigaChat.")

        # Secrets: only update when provided explicitly
        if "credentials" in data:
            _set("CREDENTIALS", data.get("credentials"), value_type="string", secret=True, label="OAuth credentials (Basic)", description="Secret: base64 credentials for OAuth.")
        if "access_token" in data:
            _set("ACCESS_TOKEN", data.get("access_token"), value_type="string", secret=True, label="Bearer token", description="Secret: Bearer token.")

        return jsonify({"success": True, "message": "Saved to Postgres (app_settings)."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@llm_bp.post("/api/llm/gigachat/chat")
def gigachat_chat():
    """
    Debug endpoint: send prompt/snapshot and return response.
    """
    try:
        ensure_gigachat_settings()
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip() or None
        snapshot = data.get("snapshot") if isinstance(data.get("snapshot"), dict) else None

        if snapshot and not prompt:
            prompt = build_gigachat_prompt(snapshot)
        if not prompt:
            return jsonify({"success": False, "error": "prompt is required (or provide snapshot)"}), 400

        cli = GigaChatClient()
        res = cli.chat(prompt=prompt)
        return jsonify({"success": True, "result": res, "prompt": prompt}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@llm_bp.post("/api/llm/gigachat/snapshot_from_ohlcv")
def gigachat_snapshot_from_ohlcv():
    """
    Helper: build semantic snapshot from OHLCV-like array.
    Payload:
      - symbol
      - ohlcv: [{open,high,low,close,volume}, ...]  (last N)
    """
    try:
        data = request.get_json(silent=True) or {}
        sym = str(data.get("symbol") or "").strip().upper()
        ohlcv = data.get("ohlcv") or []
        if not sym:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        if not isinstance(ohlcv, list) or len(ohlcv) < 30:
            return jsonify({"success": False, "error": "ohlcv list is required (>=30)"}), 400

        import pandas as pd

        df = pd.DataFrame(ohlcv)
        snap = build_semantic_snapshot(df, symbol=sym, market_regime="unknown", market_regime_details=None, decision=None, votes=None)
        prompt = build_gigachat_prompt(snap)
        return jsonify({"success": True, "snapshot": snap, "prompt": prompt}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

