"use client";

import { useEffect, useState } from "react";

type MaxSettings = {
  bot_token_masked?: string | null;
  has_bot_token?: boolean;
  api_url?: string | null;
  chat_ids?: string | null;
  enable_max_bot_polling?: boolean;
};

type Status = {
  type: "info" | "success" | "error";
  message: string;
} | null;

export function MaxBotSettingsForm() {
  const [botToken, setBotToken] = useState("");
  const [apiUrl, setApiUrl] = useState("");
  const [chatIds, setChatIds] = useState("");
  const [enablePolling, setEnablePolling] = useState(false);
  const [settings, setSettings] = useState<MaxSettings | null>(null);
  const [status, setStatus] = useState<Status>({ type: "info", message: "Загрузка..." });
  const [isSaving, setIsSaving] = useState(false);

  async function loadSettings() {
    try {
      setStatus({ type: "info", message: "Загрузка..." });
      const response = await fetch("/api/settings/bots/max", { cache: "no-store" });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "load failed");
      setSettings(data.max || null);
      setApiUrl(data.max?.api_url || "");
      setChatIds(data.max?.chat_ids || "");
      setEnablePolling(Boolean(data.max?.enable_max_bot_polling));
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
      const response = await fetch("/api/settings/bots/max", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_token: botToken.trim(),
          api_url: apiUrl.trim(),
          chat_ids: chatIds.trim(),
          enable_max_bot_polling: enablePolling,
        }),
      });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "save failed");
      setBotToken("");
      setSettings(data.max || null);
      setApiUrl(data.max?.api_url || "");
      setChatIds(data.max?.chat_ids || "");
      setEnablePolling(Boolean(data.max?.enable_max_bot_polling));
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
        <h2 className="text-xl font-semibold text-slate-900">MAX</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Для MAX нужны Bot token, API URL и Chat IDs. Значения сохраняются в app_settings.
        </p>
      </div>

      <label className="mb-6 flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
        <input
          type="checkbox"
          className="mt-1 h-4 w-4 rounded border-slate-300"
          checked={enablePolling}
          onChange={(e) => setEnablePolling(e.target.checked)}
        />
        <span className="text-sm leading-6 text-slate-700">
          <span className="font-medium text-slate-900">Long polling (/updates)</span>
          — регистрация пользователей через GET /updates. Не включай одновременно с Webhook в
          MAX. Хранится только в Postgres, не в ENV.
        </span>
      </label>

      <div className="grid gap-5 lg:grid-cols-2">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Bot token</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            type="password"
            value={botToken}
            onChange={(event) => setBotToken(event.target.value)}
            placeholder="MAX bot token"
            autoComplete="off"
          />
          <span className="mt-2 block text-xs text-slate-500">{tokenLabel}</span>
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">API URL</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            value={apiUrl}
            onChange={(event) => setApiUrl(event.target.value)}
            placeholder="https://..."
          />
          <span className="mt-2 block text-xs text-slate-500">Base URL для MAX Bot API.</span>
        </label>

        <label className="block lg:col-span-2">
          <span className="text-sm font-medium text-slate-700">Chat IDs</span>
          <input
            className="mt-2 w-full rounded-2xl border border-slate-200 px-4 py-3 font-mono text-sm outline-none transition focus:border-slate-400"
            value={chatIds}
            onChange={(event) => setChatIds(event.target.value)}
            placeholder="123456789,987654321"
          />
          <span className="mt-2 block text-xs text-slate-500">Несколько значений через запятую.</span>
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
