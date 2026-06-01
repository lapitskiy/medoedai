import os
from dataclasses import dataclass


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class NotifierConfig:
    redis_url: str
    stream_name: str
    consumer_group: str
    consumer_name: str
    channels: list[str]
    telegram_bot_token: str | None
    telegram_chat_ids: list[str]
    max_api_url: str | None
    max_bot_token: str | None
    max_chat_ids: list[str]


def load_config() -> NotifierConfig:
    channels = [ch.lower() for ch in _csv_env("SIGNAL_NOTIFIER_CHANNELS")]
    cfg = NotifierConfig(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        stream_name=os.getenv("SIGNAL_STREAM_NAME", "signals:events"),
        consumer_group=os.getenv("SIGNAL_CONSUMER_GROUP", "signal-notifier"),
        consumer_name=os.getenv("SIGNAL_CONSUMER_NAME", "signal-notifier-1"),
        channels=channels,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_ids=_csv_env("TELEGRAM_CHAT_IDS"),
        max_api_url=os.getenv("MAX_API_URL") or None,
        max_bot_token=os.getenv("MAX_BOT_TOKEN") or None,
        max_chat_ids=_csv_env("MAX_CHAT_IDS"),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: NotifierConfig) -> None:
    allowed = {"telegram", "max"}
    if not cfg.channels:
        raise RuntimeError("SIGNAL_NOTIFIER_CHANNELS is empty; set telegram,max or disable the container")
    unknown = sorted(set(cfg.channels) - allowed)
    if unknown:
        raise RuntimeError(f"Unknown notifier channels: {unknown}")
    if "telegram" in cfg.channels and (not cfg.telegram_bot_token or not cfg.telegram_chat_ids):
        raise RuntimeError("Telegram channel enabled, but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_IDS are empty")
    if "max" in cfg.channels and (not cfg.max_api_url or not cfg.max_bot_token or not cfg.max_chat_ids):
        raise RuntimeError("MAX channel enabled, but MAX_API_URL/MAX_BOT_TOKEN/MAX_CHAT_IDS are empty")

