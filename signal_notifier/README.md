# signal-notifier

Отдельный контейнер для доставки торговых сигналов во внешние API.

Поток:

```text
celery-trade -> Redis Stream signals:events -> signal-notifier -> Telegram/MAX
```

Минимальные переменные:

```text
SIGNAL_NOTIFIER_CHANNELS=telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_IDS=123456,789012
```

Для MAX включите канал `max` и задайте `MAX_API_URL`, `MAX_BOT_TOKEN`, `MAX_CHAT_IDS`.
Если канал включён, но его ключи пустые, контейнер завершится с ошибкой.

