from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROXY_URL = "http://telegram-vless-proxy:1080"
CONFIG_PATH = Path("/workspace/infra/telegram-vless-proxy/generated.json")


def write_sing_box_config_from_vless(vless_url: str) -> str:
    raw = str(vless_url or "").strip()
    if not raw:
        raise ValueError("vless_url is required")

    parsed = urlparse(raw)
    if parsed.scheme != "vless":
        raise ValueError("vless_url must start with vless://")
    if not parsed.hostname or not parsed.port or not parsed.username:
        raise ValueError("vless_url must include uuid, host and port")

    params = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
    security = str(params.get("security") or "reality").strip().lower()
    flow = str(params.get("flow") or "").strip()
    sni = str(params.get("sni") or "").strip()
    public_key = str(params.get("pbk") or "").strip()
    short_id = str(params.get("sid") or "").strip()
    fingerprint = str(params.get("fp") or "chrome").strip() or "chrome"

    outbound = {
        "type": "vless",
        "tag": "vless-out",
        "server": parsed.hostname,
        "server_port": int(parsed.port),
        "uuid": unquote(parsed.username),
        "flow": flow,
        "tls": {
            "enabled": security in ("tls", "reality"),
            "server_name": sni,
            "utls": {
                "enabled": True,
                "fingerprint": fingerprint,
            },
            "reality": {
                "enabled": security == "reality",
                "public_key": public_key,
                "short_id": short_id,
            },
        },
    }

    config = {
        "log": {
            "level": "info",
            "timestamp": True,
        },
        "inbounds": [
            {
                "type": "mixed",
                "tag": "telegram-proxy",
                "listen": "0.0.0.0",
                "listen_port": 1080,
            }
        ],
        "outbounds": [outbound],
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return PROXY_URL
