"use client";

import { useEffect, useState } from "react";

type TelegramSettings = {
  bot_token_masked?: string | null;
  has_bot_token?: boolean;
  chat_ids?: string | null;
  proxy_url?: string | null;
  vless_url?: string | null;
};

type Status = {
  type: "info" | "success" | "error";
  message: string;
} | null;

export function TelegramBotSettingsForm() {
  const [botToken, setBotToken] = useState("");
  const [chatIds, setChatIds] = useState("");
  const [proxyUrl, setProxyUrl] = useState("");
  const [vlessUrl, setVlessUrl] = useState("");
  const [settings, setSettings] = useState<TelegramSettings | null>(null);
  const [status, setStatus] = useState<Status>({ type: "info", message: "Загрузка..." });
  const [isSaving, setIsSaving] = useState(false);

  async function loadSettings() {
    try {
      setStatus({ type: "info", message: "Загрузка..." });
      const response = await fetch("/api/settings/bots/telegram", { cache: "no-store" });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "load failed");
      setSettings(data.telegram || null);
      setChatIds(data.telegram?.chat_ids || "");
      setProxyUrl(data.telegram?.proxy_url || "");
      setVlessUrl(data.telegram?.vless_url || "");
      setStatus(null);
    } catch (error) {
      setStatus({
        type: "error",
        message: error instanceof Error ? error.message : "Ошибка загрузки",
      });
    }
  }

  async function saveSettings() {
    try {
      setIsSaving(true);
      const response = await fetch("/api/settings/bots/telegram", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_token: botToken.trim(),
          chat_ids: chatIds.trim(),
          proxy_url: proxyUrl.trim(),
          vless_url: vlessUrl.trim(),
        }),
      });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "save failed");
      setBotToken("");
      setSettings(data.telegram || null);
      setChatIds(data.telegram?.chat_ids || "");
      setProxyUrl(data.telegram?.proxy_url || "");
      setVlessUrl(data.telegram?.vless_url || "");
      setStatus({ type: "success", message: "Сохранено в Postgres" });
    } catch (error) {
      setStatus({
        type: "error",
        message: error instanceof Error ? error.message : "Ошибка сохранения",
      });
    } finally {
      setIsSaving(false);
    }
  }

  useEffect(() => {
    void loadSettings();
  }, []);

  const tokenLabel = settings?.has_bot_token
    ? `Текущий token: ${settings.bot_token_masked || "***"}`
    : "Текущий token: не задан";

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-900">Telegram</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Сначала можно сохранить только Bot token. Chat IDs добавь после того, как узнаешь свой chat_id
          (например через getUpdates). Поле Chat IDs можно оставить пустым.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Bot token</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            type="password"
            value={botToken}
            onChange={(event) => setBotToken(event.target.value)}
            placeholder="123456:ABC..."
            autoComplete="off"
          />
          <span className="mt-2 block text-xs text-slate-500">{tokenLabel}</span>
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">Chat IDs (необязательно)</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            value={chatIds}
            onChange={(event) => setChatIds(event.target.value)}
            placeholder="123456789,-1001234567890"
          />
          <span className="mt-2 block text-xs text-slate-500">
            Несколько значений через запятую. Цифра 1 не является chat_id — нужен реальный id из Telegram.
          </span>
        </label>

        <label className="block lg:col-span-2">
          <span className="text-sm font-medium text-slate-700">
            Telegram proxy URL
          </span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            value={proxyUrl}
            onChange={(event) => setProxyUrl(event.target.value)}
            placeholder="http://telegram-vless-proxy:1080"
          />
          <span className="mt-2 block text-xs text-slate-500">
            Используется только для Telegram Bot API. Остальной сайт через proxy не пойдёт.
          </span>
        </label>
      </div>

      <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <p className="text-sm font-semibold text-slate-900">VPN (VLESS)</p>
        <p className="mt-1 text-xs text-slate-600">
          Одна строка VLESS для твоей VPN-конфигурации Telegram.
        </p>
        <label className="mt-3 block">
          <span className="text-sm font-medium text-slate-700">VLESS URL</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            value={vlessUrl}
            onChange={(event) => setVlessUrl(event.target.value)}
            placeholder="vless://uuid@host:443?encryption=none&flow=xtls-rprx-vision&security=reality..."
            autoComplete="off"
          />
        </label>
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void saveSettings()}
          disabled={isSaving}
          className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isSaving ? "Сохраняю..." : "Сохранить"}
        </button>
        <button
          type="button"
          onClick={() => void loadSettings()}
          className="rounded-2xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Обновить
        </button>
        {status ? (
          <span
            className={`rounded-2xl px-4 py-3 text-sm ${
              status.type === "error"
                ? "bg-red-50 text-red-700"
                : status.type === "success"
                  ? "bg-emerald-50 text-emerald-700"
                  : "bg-slate-50 text-slate-600"
            }`}
          >
            {status.message}
          </span>
        ) : null}
      </div>
    </section>
  );
}
