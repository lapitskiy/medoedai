from __future__ import annotations

from typing import Any, Dict, Optional

from utils.settings_store import ensure_settings_table, list_settings, upsert_setting


SCOPE = "llm"
GROUP = "gigachat"


DEFAULTS: list[dict[str, Any]] = [
    {
        "key": "ENABLED",
        "type": "bool",
        "label": "GigaChat enabled",
        "description": "Enable/disable GigaChat calls in trading pipeline.",
        "secret": False,
        "value": "0",
    },
    {
        "key": "ONLY_ON_BUY",
        "type": "bool",
        "label": "Only on BUY",
        "description": "If enabled, call GigaChat only when final decision is BUY. (Future switch.)",
        "secret": False,
        "value": "0",
    },
    {
        "key": "AUTH_TYPE",
        "type": "string",
        "label": "Auth type",
        "description": "oauth (Authorization: Basic <credentials>) or bearer (Authorization: Bearer <token>).",
        "secret": False,
        "value": "oauth",
    },
    {
        "key": "CREDENTIALS",
        "type": "string",
        "label": "OAuth credentials (Basic)",
        "description": "Secret: base64(client_id:client_secret) or provider-specific credentials for OAuth.",
        "secret": True,
        "value": None,
    },
    {
        "key": "ACCESS_TOKEN",
        "type": "string",
        "label": "Bearer token",
        "description": "Secret: Bearer token (if AUTH_TYPE=bearer).",
        "secret": True,
        "value": None,
    },
    {
        "key": "OAUTH_URL",
        "type": "string",
        "label": "OAuth URL",
        "description": "Token endpoint URL.",
        "secret": False,
        "value": "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
    },
    {
        "key": "OAUTH_SCOPE",
        "type": "string",
        "label": "OAuth scope",
        "description": "Scope for token request (provider-specific).",
        "secret": False,
        "value": "GIGACHAT_API_PERS",
    },
    {
        "key": "CHAT_URL",
        "type": "string",
        "label": "Chat completions URL",
        "description": "Chat completions endpoint URL.",
        "secret": False,
        "value": "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
    },
    {
        "key": "MODEL",
        "type": "string",
        "label": "Model",
        "description": "Model name (provider-specific).",
        "secret": False,
        "value": "GigaChat",
    },
    {
        "key": "TIMEOUT_SEC",
        "type": "number",
        "label": "Timeout (sec)",
        "description": "HTTP timeout for GigaChat requests.",
        "secret": False,
        "value": "10",
    },
    {
        "key": "TEMPERATURE",
        "type": "number",
        "label": "Temperature",
        "description": "Sampling temperature for the model.",
        "secret": False,
        "value": "0.2",
    },
    {
        "key": "MAX_TOKENS",
        "type": "number",
        "label": "Max tokens",
        "description": "Max tokens in completion.",
        "secret": False,
        "value": "256",
    },
    {
        "key": "VERIFY_SSL",
        "type": "bool",
        "label": "Verify SSL",
        "description": "Enable SSL cert verification. If container lacks CA bundle, set to 0.",
        "secret": False,
        "value": "0",
    },
    {
        "key": "SYSTEM_PROMPT",
        "type": "string",
        "label": "System prompt",
        "description": "System instruction for GigaChat (keep short).",
        "secret": False,
        "value": "Ты опытный трейдер. Ответь кратко: 1) что видишь, 2) риск, 3) что бы сделал.",
    },
]


_BOOTSTRAPPED = False


def ensure_gigachat_settings() -> None:
    """Creates missing (scope=llm, group=gigachat) settings rows for UI/consumption."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    ensure_settings_table()
    try:
        rows = list_settings(scope=SCOPE, group=GROUP)
        existing_keys = {str(r.get("key") or "") for r in rows}
    except Exception:
        existing_keys = set()

    for d in DEFAULTS:
        if d["key"] in existing_keys:
            continue
        try:
            upsert_setting(
                scope=SCOPE,
                group=GROUP,
                key=str(d["key"]),
                value_type=str(d["type"]),
                label=str(d.get("label") or ""),
                description=d.get("description"),
                is_secret=bool(d.get("secret")),
                value=d.get("value"),
            )
        except Exception:
            # Never break the app due to settings bootstrapping.
            continue

    _BOOTSTRAPPED = True

