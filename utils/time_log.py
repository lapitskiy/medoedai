from datetime import datetime, timedelta


def msk_now_str() -> str:
    return (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")


def msk_tag(message: str) -> str:
    return f"{msk_now_str()} | {message}"
