from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from utils.gigachat_settings import ensure_gigachat_settings
from utils.settings_store import get_setting_value


def _truthy(v: Any) -> bool:
    try:
        return str(v).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _to_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


def _to_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return int(default)
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


@dataclass
class GigaChatConfig:
    enabled: bool
    only_on_buy: bool
    auth_type: str
    credentials: Optional[str]
    access_token: Optional[str]
    oauth_url: str
    oauth_scope: str
    chat_url: str
    model: str
    timeout_sec: float
    temperature: float
    max_tokens: int
    verify_ssl: bool
    system_prompt: str


def load_gigachat_config() -> GigaChatConfig:
    ensure_gigachat_settings()
    scope, group = "llm", "gigachat"

    enabled = _truthy(get_setting_value(scope, group, "ENABLED"))
    only_on_buy = _truthy(get_setting_value(scope, group, "ONLY_ON_BUY"))
    auth_type = (get_setting_value(scope, group, "AUTH_TYPE") or "oauth").strip().lower()
    credentials = (get_setting_value(scope, group, "CREDENTIALS") or None)
    access_token = (get_setting_value(scope, group, "ACCESS_TOKEN") or None)
    oauth_url = (get_setting_value(scope, group, "OAUTH_URL") or "").strip()
    oauth_scope = (get_setting_value(scope, group, "OAUTH_SCOPE") or "GIGACHAT_API_PERS").strip()
    chat_url = (get_setting_value(scope, group, "CHAT_URL") or "").strip()
    model = (get_setting_value(scope, group, "MODEL") or "GigaChat").strip()
    timeout_sec = _to_float(get_setting_value(scope, group, "TIMEOUT_SEC"), 10.0)
    temperature = _to_float(get_setting_value(scope, group, "TEMPERATURE"), 0.2)
    max_tokens = _to_int(get_setting_value(scope, group, "MAX_TOKENS"), 256)
    verify_ssl = _truthy(get_setting_value(scope, group, "VERIFY_SSL"))
    system_prompt = (get_setting_value(scope, group, "SYSTEM_PROMPT") or "").strip()

    return GigaChatConfig(
        enabled=bool(enabled),
        only_on_buy=bool(only_on_buy),
        auth_type=str(auth_type or "oauth"),
        credentials=str(credentials).strip() if credentials and str(credentials).strip() else None,
        access_token=str(access_token).strip() if access_token and str(access_token).strip() else None,
        oauth_url=oauth_url,
        oauth_scope=oauth_scope,
        chat_url=chat_url,
        model=model,
        timeout_sec=float(timeout_sec),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        verify_ssl=bool(verify_ssl),
        system_prompt=system_prompt,
    )


class GigaChatClient:
    """
    Minimal requests-based client.

    Supports:
    - auth_type=oauth: fetch token via oauth_url using Authorization: Basic <credentials>
    - auth_type=bearer: use ACCESS_TOKEN directly
    """

    def __init__(self):
        self._token: Optional[str] = None
        self._token_expire_ts: float = 0.0

    def _get_bearer(self, cfg: GigaChatConfig) -> Tuple[Optional[str], Optional[str]]:
        if cfg.auth_type == "bearer":
            if not cfg.access_token:
                return None, "ACCESS_TOKEN is empty"
            return str(cfg.access_token), None

        # oauth
        if not cfg.credentials:
            return None, "CREDENTIALS is empty"
        if not cfg.oauth_url:
            return None, "OAUTH_URL is empty"

        now = time.time()
        if self._token and now < (self._token_expire_ts - 15):
            return self._token, None

        rq_uid = str(uuid.uuid4())
        headers = {
            "Authorization": f"Basic {cfg.credentials}",
            "RqUID": rq_uid,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"scope": cfg.oauth_scope or "GIGACHAT_API_PERS"}
        try:
            r = requests.post(
                cfg.oauth_url,
                headers=headers,
                data=data,
                timeout=float(cfg.timeout_sec),
                verify=bool(cfg.verify_ssl),
            )
            if not r.ok:
                return None, f"oauth HTTP {r.status_code}: {r.text[:300]}"
            j = r.json()
            token = j.get("access_token") or j.get("token")
            if not token:
                return None, "oauth response has no access_token"
            expires_in = j.get("expires_in")
            try:
                exp = now + float(expires_in)
            except Exception:
                exp = now + 25 * 60
            self._token = str(token)
            self._token_expire_ts = float(exp)
            return self._token, None
        except Exception as e:
            return None, f"oauth error: {e}"

    def chat(self, *, prompt: str, cfg: Optional[GigaChatConfig] = None) -> Dict[str, Any]:
        cfg = cfg or load_gigachat_config()
        started = time.time()

        if not cfg.chat_url:
            return {"ok": False, "error": "CHAT_URL is empty", "latency_ms": int((time.time() - started) * 1000)}

        bearer, err = self._get_bearer(cfg)
        if err:
            return {"ok": False, "error": err, "latency_ms": int((time.time() - started) * 1000)}

        system_prompt = cfg.system_prompt or "Ты опытный трейдер. Ответь кратко."
        payload = {
            "model": cfg.model or "GigaChat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(prompt or "")},
            ],
            "temperature": float(cfg.temperature),
            "max_tokens": int(cfg.max_tokens),
        }

        headers = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            r = requests.post(
                cfg.chat_url,
                headers=headers,
                json=payload,
                timeout=float(cfg.timeout_sec),
                verify=bool(cfg.verify_ssl),
            )
            raw_text = None
            try:
                raw_text = r.text
            except Exception:
                raw_text = None
            if not r.ok:
                return {
                    "ok": False,
                    "error": f"chat HTTP {r.status_code}",
                    "raw": (raw_text[:1000] if isinstance(raw_text, str) else None),
                    "latency_ms": int((time.time() - started) * 1000),
                }

            j = r.json()
            content = None
            try:
                choices = j.get("choices") or []
                if choices:
                    msg = (choices[0] or {}).get("message") or {}
                    content = msg.get("content")
            except Exception:
                content = None
            if content is None and isinstance(raw_text, str):
                content = raw_text

            parsed = None
            try:
                parsed = parse_gigachat_text(str(content or ""))
            except Exception:
                parsed = None

            return {
                "ok": True,
                "text": str(content or "").strip(),
                "parsed": parsed,
                "raw": j,
                "latency_ms": int((time.time() - started) * 1000),
                "model": cfg.model,
            }
        except Exception as e:
            return {"ok": False, "error": f"chat error: {e}", "latency_ms": int((time.time() - started) * 1000)}


def parse_gigachat_text(text: str) -> Dict[str, Any]:
    """
    Extracts final decision from LLM output.
    Preferred: strict JSON. Fallback: keyword heuristics (RU/EN).
    """
    t = (text or "").strip()

    # 1) Try JSON extraction (LLM sometimes wraps JSON with extra text — try best-effort slice)
    j = None
    try:
        a = t.find("{")
        b = t.rfind("}")
        if a != -1 and b != -1 and b > a:
            j = json.loads(t[a : b + 1])
    except Exception:
        j = None

    decision = None
    confidence = None
    reason = None
    risk = None

    if isinstance(j, dict):
        decision = j.get("decision") or j.get("action") or j.get("recommendation")
        confidence = j.get("confidence") or j.get("score")
        reason = j.get("reason") or j.get("analysis")
        risk = j.get("risk")

    def _norm_dec(d: Any) -> Optional[str]:
        try:
            s = str(d or "").strip().upper()
            if s in ("BUY", "SELL", "HOLD"):
                return s
            if s in ("LONG",):
                return "BUY"
            if s in ("SHORT",):
                return "SELL"
        except Exception:
            return None
        return None

    decision_n = _norm_dec(decision)

    # 2) Fallback heuristics
    if not decision_n:
        tl = t.lower()
        if any(x in tl for x in ("не вход", "не заход", "воздерж", "подожд", "hold", "держ")):
            decision_n = "HOLD"
        elif any(x in tl for x in ("прод", "шорт", "sell", "продавать", "продать")):
            decision_n = "SELL"
        elif any(x in tl for x in ("куп", "лонг", "buy", "покупать", "покупка", "купить")):
            decision_n = "BUY"
        else:
            decision_n = "HOLD"

    # confidence best-effort
    try:
        if confidence is not None:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = None

    return {
        "decision": decision_n,
        "confidence": confidence,
        "reason": (str(reason).strip()[:500] if reason is not None else None),
        "risk": (str(risk).strip()[:500] if risk is not None else None),
        "json": j if isinstance(j, dict) else None,
    }

