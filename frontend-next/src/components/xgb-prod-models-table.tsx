"use client";

import { useState } from "react";

import type { XgbProdModel } from "@/lib/backend";

type DeleteResponse = {
  success?: boolean;
  error?: string;
  current_version?: string | null;
};

type XgbProdModelsTableProps = {
  initialModels: XgbProdModel[];
};

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function compactPath(value: string | null | undefined, parts = 6): string {
  if (!value) {
    return "-";
  }
  return value.replace(/\\/g, "/").split("/").filter(Boolean).slice(-parts).join("/");
}

function formatDate(value: string | null): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(date);
}

export function XgbProdModelsTable({ initialModels }: XgbProdModelsTableProps) {
  const [models, setModels] = useState(initialModels);
  const [deletingPath, setDeletingPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const deleteModel = async (model: XgbProdModel) => {
    const title = `${model.symbol} ${model.ensemble} ${model.version}`;
    if (!window.confirm(`Удалить production XGB модель ${title}?`)) {
      return;
    }

    setDeletingPath(model.path);
    setError(null);

    try {
      const response = await fetch("/api/xgb/prod/models", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: model.symbol,
          ensemble: model.ensemble,
          version: model.version,
        }),
      });
      const data = (await response.json()) as DeleteResponse;

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Delete failed");
      }

      setModels((current) =>
        current
          .filter((item) => item.path !== model.path)
          .map((item) =>
            item.symbol === model.symbol && item.ensemble === model.ensemble
              ? { ...item, isCurrent: item.version === data.current_version }
              : item,
          ),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingPath(null);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200">
      <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800">
        Production моделей: {models.length}
      </div>
      {error ? (
        <div className="border-b border-red-100 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      ) : null}
      {models.length === 0 ? (
        <div className="px-4 py-5 text-sm text-slate-500">
          В `models/xgb` пока нет production XGB моделей.
        </div>
      ) : (
        <div className="max-h-[72vh] overflow-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 bg-white text-slate-600">
              <tr>
                <th className="border-b border-slate-200 px-3 py-3">Symbol</th>
                <th className="border-b border-slate-200 px-3 py-3">Ensemble</th>
                <th className="border-b border-slate-200 px-3 py-3">Version</th>
                <th className="border-b border-slate-200 px-3 py-3">Task</th>
                <th className="border-b border-slate-200 px-3 py-3">Run</th>
                <th className="border-b border-slate-200 px-3 py-3">UUID</th>
                <th className="border-b border-slate-200 px-3 py-3">val_acc</th>
                <th className="border-b border-slate-200 px-3 py-3">OOS PnL</th>
                <th className="border-b border-slate-200 px-3 py-3">Trades</th>
                <th className="border-b border-slate-200 px-3 py-3">Created</th>
                <th className="border-b border-slate-200 px-3 py-3">Path</th>
                <th className="border-b border-slate-200 px-3 py-3">Action</th>
              </tr>
            </thead>
            <tbody className="bg-white">
              {models.map((model) => (
                <tr key={model.path}>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-semibold">
                    {model.symbol}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {model.ensemble}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    <span className="font-mono text-xs">{model.version}</span>
                    {model.isCurrent ? (
                      <span className="ml-2 rounded-full bg-emerald-100 px-2 py-1 text-xs font-semibold text-emerald-700">
                        current
                      </span>
                    ) : null}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {model.task || "-"} / {model.direction || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-600">
                    {model.runId || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-600">
                    {model.modelUuid || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(model.trainMetrics.valAcc, 5)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(model.bestOos.pnlTotal, 2)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(model.bestOos.tradesCount, 0)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatDate(model.createdAt)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500">
                    {compactPath(model.path)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    <button
                      type="button"
                      onClick={() => deleteModel(model)}
                      disabled={deletingPath === model.path}
                      className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-semibold text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {deletingPath === model.path ? "Удаляю..." : "Удалить"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
