"use client";

import { useEffect, useMemo, useState } from "react";

type Status = {
  type: "info" | "success" | "error";
  message: string;
} | null;

type ActiveAgent = {
  session_id?: string | null;
  symbol?: string | null;
  is_xgb?: boolean;
  model_path?: string | null;
  model_paths?: string[] | null;
  xgb_task?: string | null;
};

type ModelItem = {
  path: string;
  symbol?: string | null;
  sessionId?: string | null;
  task?: string | null;
  active: boolean;
};

function normalizePath(value: unknown) {
  return String(value ?? "").replace(/\\/g, "/").trim();
}

function modelLabel(model: ModelItem) {
  const parts = model.path.split("/").filter(Boolean);
  const version = parts.find((part) => /^v\d+$/i.test(part));
  const file = parts.at(-1) ?? "model";
  return [model.symbol, model.task, version, file].filter(Boolean).join(" / ");
}

export function NotificationModelsSettingsForm() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [configured, setConfigured] = useState(false);
  const [status, setStatus] = useState<Status>({ type: "info", message: "Загрузка..." });
  const [isSaving, setIsSaving] = useState(false);

  async function loadSettings() {
    try {
      setStatus({ type: "info", message: "Загрузка активных XGB-моделей..." });
      const [settingsResponse, agentsResponse] = await Promise.all([
        fetch("/api/settings/bots/notification-models", { cache: "no-store" }),
        fetch("/api/trading/status_all", { cache: "no-store" }),
      ]);
      const settingsData = await settingsResponse.json();
      const agentsData = await agentsResponse.json();
      if (!settingsData.success) throw new Error(settingsData.error || "settings load failed");
      if (!agentsData.success) throw new Error(agentsData.error || "active models load failed");

      const savedPaths = new Set(
        (settingsData.selected_model_paths ?? []).map((path: unknown) => normalizePath(path)).filter(Boolean),
      );
      const byPath = new Map<string, ModelItem>();
      for (const agent of (agentsData.active_agents ?? []) as ActiveAgent[]) {
        if (!agent.is_xgb) continue;
        const paths = agent.model_paths?.length ? agent.model_paths : [agent.model_path];
        for (const rawPath of paths) {
          const path = normalizePath(rawPath);
          if (!path) continue;
          byPath.set(path, {
            path,
            symbol: agent.symbol,
            sessionId: agent.session_id,
            task: agent.xgb_task,
            active: true,
          });
        }
      }
      for (const path of savedPaths) {
        if (!byPath.has(path)) byPath.set(path, { path, active: false });
      }

      const items = [...byPath.values()].sort((a, b) => modelLabel(a).localeCompare(modelLabel(b)));
      setConfigured(Boolean(settingsData.configured));
      setModels(items);
      setSelected(settingsData.configured ? savedPaths : new Set(items.map((item) => item.path)));
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
      const response = await fetch("/api/settings/bots/notification-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_paths: [...selected] }),
      });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "save failed");
      setConfigured(true);
      setSelected(new Set((data.selected_model_paths ?? []).map((path: unknown) => normalizePath(path))));
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

  const selectedCount = useMemo(() => selected.size, [selected]);

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-slate-900">Оповещения по XGB-моделям</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Список берётся из активных XGB trading sessions. Если фильтр не сохранён, уведомления разрешены всем моделям.
        </p>
      </div>

      <div className="space-y-3">
        {models.length === 0 ? (
          <div className="rounded-2xl bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Активных XGB-моделей не найдено.
          </div>
        ) : (
          models.map((model) => (
            <label key={model.path} className="flex cursor-pointer items-start gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
              <input
                type="checkbox"
                className="mt-1 h-4 w-4 rounded border-slate-300"
                checked={selected.has(model.path)}
                onChange={(event) => {
                  const next = new Set(selected);
                  if (event.target.checked) next.add(model.path);
                  else next.delete(model.path);
                  setSelected(next);
                }}
              />
              <span className="min-w-0 text-sm leading-6 text-slate-700">
                <span className="font-semibold text-slate-900">{modelLabel(model)}</span>
                {!model.active ? <span className="ml-2 text-xs text-amber-700">(сейчас не активна)</span> : null}
                <span className="block break-all font-mono text-xs text-slate-500">{model.path}</span>
              </span>
            </label>
          ))
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void saveSettings()}
          disabled={isSaving}
          className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isSaving ? "Сохраняю..." : "Сохранить выбор"}
        </button>
        <button
          type="button"
          onClick={() => void loadSettings()}
          className="rounded-2xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Обновить
        </button>
        <span className="text-sm text-slate-500">
          {configured ? `Выбрано: ${selectedCount}` : "Фильтр ещё не сохранён"}
        </span>
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
