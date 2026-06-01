#!/bin/sh
set -eu

if [ -s /config/generated.json ]; then
  echo "telegram-vless-proxy: using generated config from /config/generated.json" >&2
  exec sing-box run -c /config/generated.json
fi

if [ -z "${VLESS_SERVER:-}" ] || [ -z "${VLESS_PORT:-}" ] || [ -z "${VLESS_UUID:-}" ]; then
  echo "VLESS_SERVER, VLESS_PORT and VLESS_UUID are required" >&2
  echo "telegram-vless-proxy is idle until VLESS_* env vars are configured" >&2
  tail -f /dev/null
fi

SECURITY="${VLESS_SECURITY:-reality}"
FLOW="${VLESS_FLOW:-}"
SNI="${VLESS_SNI:-}"
PUBLIC_KEY="${VLESS_PUBLIC_KEY:-}"
SHORT_ID="${VLESS_SHORT_ID:-}"
FINGERPRINT="${VLESS_FINGERPRINT:-chrome}"

cat > /tmp/sing-box.json <<EOF
{
  "log": {
    "level": "info",
    "timestamp": true
  },
  "inbounds": [
    {
      "type": "mixed",
      "tag": "telegram-proxy",
      "listen": "0.0.0.0",
      "listen_port": 1080
    }
  ],
  "outbounds": [
    {
      "type": "vless",
      "tag": "vless-out",
      "server": "$VLESS_SERVER",
      "server_port": $VLESS_PORT,
      "uuid": "$VLESS_UUID",
      "flow": "$FLOW",
      "tls": {
        "enabled": true,
        "server_name": "$SNI",
        "utls": {
          "enabled": true,
          "fingerprint": "$FINGERPRINT"
        },
        "reality": {
          "enabled": $(if [ "$SECURITY" = "reality" ]; then echo true; else echo false; fi),
          "public_key": "$PUBLIC_KEY",
          "short_id": "$SHORT_ID"
        }
      }
    }
  ]
}
EOF

exec sing-box run -c /tmp/sing-box.json
