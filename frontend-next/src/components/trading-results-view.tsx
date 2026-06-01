"use client";

import { useEffect, useMemo, useState } from "react";
import type { BybitAccount } from "@/lib/backend";

type ModelFamily = "all" | "xgb" | "dqn";

type RawTrade = {
  success?: boolean;
  symbol?: string | null;
  side?: string | null;
  entry_price?: number | null;
  exit_price?: number | null;
  quantity?: number | null;
  entry_time?: string | null;
  exit_time?: string | null;
  pnl?: number | null;
  pnl_pct?: number | null;
  leverage?: number | null;
  pnl_pct_leveraged?: number | null;
  margin_pnl?: number | null;
  exit_reason?: string | null;
  model_path?: string | null;
  model_family?: string | null;
  account_id?: string | null;
  account_label?: string | null;
  bal_before?: number | null;
  bal_after?: number | null;
  duration_min?: number | null;
  mae?: number | null;
  mfe?: number | null;
  xgb_shap?: XgbShap | null;
};

type XgbShapFeature = {
  feature_name?: string | null;
  shap_value?: number | null;
};

type XgbShap = {
  top_features?: XgbShapFeature[] | null;
  full_vector?: XgbShapFeature[] | null;
  base_value?: number | null;
};

type TradeHistoryResponse = {
  success?: boolean;
  trades?: RawTrade[];
  total?: number;
  error?: string;
};

type TradingTrade = {
  id: string;
  symbol: string;
  side: string | null;
  entryPrice: number | null;
  exitPrice: number | null;
  quantity: number | null;
  entryTime: string | null;
  exitTime: string | null;
  pnl: number | null;
  pnlPct: number | null;
  leverage: number | null;
  pnlPctLeveraged: number | null;
  marginPnl: number | null;
  exitReason: string | null;
  modelPath: string | null;
  modelFamily: ModelFamily | "unknown";
  accountId: string | null;
  accountLabel: string | null;
  balanceBefore: number | null;
  balanceAfter: number | null;
  durationMin: number | null;
  mae: number | null;
  mfe: number | null;
  xgbShap: XgbShap | null;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ru-RU", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "Europe/Moscow",
  }).format(date);
}

function formatNumber(value: number | null | undefined, digits = 4): string {
  if (!isFiniteNumber(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatSignedNumber(value: number | null | undefined, digits = 4): string {
  if (!isFiniteNumber(value)) {
    return "—";
  }
  return `${value >= 0 ? "+" : ""}${formatNumber(value, digits)}`;
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  const text = typeof value === "string" ? value : String(value);
  return `"${text.replace(/"/g, '""')}"`;
}

function toCsvNumber(value: number | null | undefined): string {
  return isFiniteNumber(value) ? String(value) : "";
}

function pickBaselineBalance(items: TradingTrade[]): number | null {
  const sorted = [...items].sort((a, b) => {
    const left = Date.parse(a.entryTime ?? a.exitTime ?? "") || 0;
    const right = Date.parse(b.entryTime ?? b.exitTime ?? "") || 0;
    return left - right;
  });

  for (const item of sorted) {
    if (isFiniteNumber(item.balanceBefore) && item.balanceBefore > 0) {
      return item.balanceBefore;
    }
    if (
      isFiniteNumber(item.balanceAfter) &&
      isFiniteNumber(item.pnl) &&
      item.balanceAfter - item.pnl > 0
    ) {
      return item.balanceAfter - item.pnl;
    }
  }

  return null;
}

function compactModelPath(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  return value.replace(/\\/g, "/").split("/").filter(Boolean).slice(-5).join("/");
}

function compactModelName(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const parts = value.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.slice(-4).join("/");
}

function pnlTone(value: number | null): string {
  if (isFiniteNumber(value) && value > 0) {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
  if (isFiniteNumber(value) && value < 0) {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }
  return "border-slate-200 bg-slate-100 text-slate-700";
}

function familyTone(family: "xgb" | "dqn" | "unknown"): string {
  return family === "xgb"
    ? "border-sky-200 bg-sky-50 text-sky-700"
    : family === "dqn"
      ? "border-violet-200 bg-violet-50 text-violet-700"
      : "border-slate-200 bg-slate-100 text-slate-700";
}

function prettyExitReason(value: string | null): string {
  const reason = String(value ?? "").trim().toLowerCase();
  if (!reason) {
    return "—";
  }
  if (reason === "take_profit" || reason === "tp") {
    return "tp";
  }
  if (reason === "stop_loss" || reason === "sl" || reason === "stop") {
    return "sl";
  }
  if (reason === "timeout" || reason === "hold_steps" || reason === "holdstep") {
    return "holdstep";
  }
  if (reason === "trailing") {
    return "trailing";
  }
  if (reason === "signal") {
    return "signal";
  }
  return reason;
}

type TradingResultsViewProps = {
  accounts: BybitAccount[];
};

export function TradingResultsView({ accounts: initialAccounts }: TradingResultsViewProps) {
  const [symbol, setSymbol] = useState("");
  const [modelFamily, setModelFamily] = useState<ModelFamily>("all");
  const [accountId, setAccountId] = useState("");
  const [results, setResults] = useState<TradingTrade[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [csvSaveStatus, setCsvSaveStatus] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function load() {
      try {
        setError(null);
        const query = new URLSearchParams();
        query.set("limit", "1000");
        if (symbol) {
          query.set("symbol", symbol);
        }
        if (accountId) {
          query.set("account_id", accountId);
        }
        const response = await fetch(`/api/trading/trade_history?${query.toString()}`, {
          method: "GET",
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const data = (await response.json()) as TradeHistoryResponse;
        if (!data.success) {
          throw new Error(data.error || "Failed to load trade history");
        }

        const mapped = (data.trades ?? []).map<TradingTrade>((item, index) => {
          const rawFamily = String(item.model_family ?? "").trim().toLowerCase();
          const modelPath = item.model_path ? String(item.model_path) : null;
          const family: TradingTrade["modelFamily"] =
            rawFamily === "xgb"
              ? "xgb"
              : rawFamily === "dqn"
                ? "dqn"
                : modelPath && modelPath.replace(/\\/g, "/").includes("/models/xgb/")
                  ? "xgb"
                  : modelPath
                    ? "dqn"
                    : "unknown";
          return {
            id: `${item.symbol ?? "no-symbol"}-${item.exit_time ?? "no-exit"}-${index}`,
            symbol: String(item.symbol ?? ""),
            side: item.side ? String(item.side).toLowerCase() : null,
            entryPrice: isFiniteNumber(item.entry_price) ? item.entry_price : null,
            exitPrice: isFiniteNumber(item.exit_price) ? item.exit_price : null,
            quantity: isFiniteNumber(item.quantity) ? item.quantity : null,
            entryTime: item.entry_time ? String(item.entry_time) : null,
            exitTime: item.exit_time ? String(item.exit_time) : null,
            pnl: isFiniteNumber(item.pnl) ? item.pnl : null,
            pnlPct: isFiniteNumber(item.pnl_pct) ? item.pnl_pct : null,
            leverage: isFiniteNumber(item.leverage) ? item.leverage : null,
            pnlPctLeveraged: isFiniteNumber(item.pnl_pct_leveraged) ? item.pnl_pct_leveraged : null,
            marginPnl: isFiniteNumber(item.margin_pnl) ? item.margin_pnl : null,
            exitReason: item.exit_reason ? String(item.exit_reason) : null,
            modelPath,
            modelFamily: family,
            accountId: item.account_id ? String(item.account_id) : null,
            accountLabel: item.account_label ? String(item.account_label) : null,
            balanceBefore: isFiniteNumber(item.bal_before) ? item.bal_before : null,
            balanceAfter: isFiniteNumber(item.bal_after) ? item.bal_after : null,
            durationMin: isFiniteNumber(item.duration_min) ? item.duration_min : null,
            mae: isFiniteNumber(item.mae) ? item.mae : null,
            mfe: isFiniteNumber(item.mfe) ? item.mfe : null,
            xgbShap: item.xgb_shap ?? null,
          };
        });

        if (!cancelled) {
          setResults(mapped);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(
            loadError instanceof Error ? loadError.message : "Failed to load trade history",
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          timer = setTimeout(() => {
            void load();
          }, 5000);
        }
      }
    }

    setIsLoading(true);
    void load();

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [accountId, symbol]);

  const symbols = useMemo(
    () =>
      Array.from(new Set(results.map((item) => item.symbol).filter(Boolean))).sort((a, b) =>
        a.localeCompare(b),
      ),
    [results],
  );

  const accounts = useMemo(() => {
    const fromHistory = results
      .filter((item) => item.accountId)
      .map((item) => [
        item.accountId as string,
        item.accountLabel?.trim() || `Account ${item.accountId}`,
      ] as const);
    const fromSettings = initialAccounts.map((item) => [item.id, item.label] as const);
    return Array.from(new Map([...fromSettings, ...fromHistory]).entries()).sort((a, b) =>
      a[1].localeCompare(b[1]),
    );
  }, [initialAccounts, results]);

  const filteredResults = useMemo(() => {
    return results.filter((item) => {
      if (symbol && item.symbol !== symbol) {
        return false;
      }
      if (modelFamily !== "all" && item.modelFamily !== modelFamily) {
        return false;
      }
      if (accountId && item.accountId !== accountId) {
        return false;
      }
      return true;
    });
  }, [accountId, modelFamily, results, symbol]);

  const selectedRows = useMemo(
    () => filteredResults.filter((item) => selectedIds.has(item.id)),
    [filteredResults, selectedIds],
  );

  const allFilteredSelected =
    filteredResults.length > 0 && filteredResults.every((item) => selectedIds.has(item.id));

  const summary = useMemo(
    () => {
      const totalPnl = filteredResults.reduce(
        (acc, item) => acc + (isFiniteNumber(item.pnl) ? item.pnl : 0),
        0,
      );
      const winPnl = filteredResults.reduce(
        (acc, item) => acc + (isFiniteNumber(item.pnl) && item.pnl > 0 ? item.pnl : 0),
        0,
      );
      const lossPnl = filteredResults.reduce(
        (acc, item) => acc + (isFiniteNumber(item.pnl) && item.pnl < 0 ? item.pnl : 0),
        0,
      );
      const baselineBalance = pickBaselineBalance(filteredResults);
      const totalPnlPct =
        isFiniteNumber(baselineBalance) && baselineBalance > 0
          ? (totalPnl / baselineBalance) * 100
          : null;

      return {
        total: filteredResults.length,
        wins: filteredResults.filter((item) => isFiniteNumber(item.pnl) && item.pnl > 0).length,
        losses: filteredResults.filter((item) => isFiniteNumber(item.pnl) && item.pnl < 0).length,
        winPnl,
        lossPnl,
        totalPnl,
        totalPnlPct,
      };
    },
    [filteredResults],
  );

  function toggleRow(id: string, checked: boolean) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(id);
      } else {
        next.delete(id);
      }
      return next;
    });
  }

  function toggleAllFiltered(checked: boolean) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      for (const item of filteredResults) {
        if (checked) {
          next.add(item.id);
        } else {
          next.delete(item.id);
        }
      }
      return next;
    });
  }

  async function exportSelectedCsv() {
    const rows = selectedRows.length > 0 ? selectedRows : filteredResults;
    if (!rows.length) {
      return;
    }
    const headers = [
      "symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price",
      "quantity", "pnl", "pnl_pct", "leverage", "pnl_pct_leveraged", "margin_pnl",
      "mae", "mfe", "exit_reason",
      "model_family", "model_path", "account_id", "account_label",
      "shap_1_feature", "shap_1_value", "shap_2_feature", "shap_2_value",
      "shap_3_feature", "shap_3_value", "shap_4_feature", "shap_4_value",
      "shap_5_feature", "shap_5_value", "shap_full_json",
    ];
    const lines = rows.map((item) => {
      const top = item.xgbShap?.top_features ?? [];
      const shapCols = Array.from({ length: 5 }).flatMap((_, index) => [
        top[index]?.feature_name ?? "",
        toCsvNumber(top[index]?.shap_value),
      ]);
      return [
        item.symbol, item.side, item.entryTime, item.exitTime,
        toCsvNumber(item.entryPrice), toCsvNumber(item.exitPrice),
        toCsvNumber(item.quantity), toCsvNumber(item.pnl), toCsvNumber(item.pnlPct),
        toCsvNumber(item.leverage), toCsvNumber(item.pnlPctLeveraged), toCsvNumber(item.marginPnl),
        toCsvNumber(item.mae), toCsvNumber(item.mfe), item.exitReason,
        item.modelFamily, item.modelPath, item.accountId, item.accountLabel,
        ...shapCols, JSON.stringify(item.xgbShap?.full_vector ?? []),
      ].map(csvCell).join(",");
    });
    const csvText = [headers.join(","), ...lines].join("\n");
    const filename = `trading-shap-results-${new Date().toISOString().slice(0, 10)}.csv`;
    try {
      setCsvSaveStatus("Сохраняю CSV на сервер...");
      const response = await fetch("/api/trading/save_shap_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        body: JSON.stringify({ csv: csvText, rows_count: rows.length }),
      });
      const data = (await response.json()) as { success?: boolean; path?: string; error?: string };
      if (!response.ok || !data.success) {
        throw new Error(data.error || `Backend returned ${response.status}`);
      }
      setCsvSaveStatus(`CSV сохранен: ${data.path ?? "predict_test/xgb_shap"}`);
    } catch (saveError) {
      setCsvSaveStatus(
        saveError instanceof Error ? `CSV не сохранен на сервер: ${saveError.message}` : "CSV не сохранен на сервер",
      );
    }
    const blob = new Blob([csvText], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 rounded-3xl border border-slate-200 bg-slate-50 p-5 lg:grid-cols-3">
        <div className="space-y-2">
          <p className="text-sm font-semibold text-slate-900">Тип статистики</p>
          <select
            value={modelFamily}
            onChange={(event) => setModelFamily(event.target.value as ModelFamily)}
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
          >
            <option value="all">Все результаты</option>
            <option value="xgb">XGB</option>
            <option value="dqn">DQN / SAC</option>
          </select>
        </div>
        <div className="space-y-2">
          <p className="text-sm font-semibold text-slate-900">Символ</p>
          <select
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
          >
            <option value="">Все символы</option>
            {symbols.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-2">
          <p className="text-sm font-semibold text-slate-900">API аккаунт</p>
          <select
            value={accountId}
            onChange={(event) => setAccountId(event.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 outline-none transition focus:border-slate-300"
          >
            <option value="">Все аккаунты</option>
            {accounts.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Всего</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{summary.total}</p>
        </div>
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-emerald-700">Win</p>
          <p className="mt-2 text-2xl font-semibold text-emerald-700">{summary.wins}</p>
          <p className="mt-1 text-sm text-emerald-700">
            PnL: {summary.winPnl > 0 ? `+${formatNumber(summary.winPnl, 2)}` : formatNumber(summary.winPnl, 2)}
          </p>
        </div>
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
          <p className="text-xs uppercase tracking-[0.2em] text-rose-700">Loss</p>
          <p className="mt-2 text-2xl font-semibold text-rose-700">{summary.losses}</p>
          <p className="mt-1 text-sm text-rose-700">PnL: {formatNumber(summary.lossPnl, 2)}</p>
        </div>
        <div className={`rounded-2xl border p-4 ${pnlTone(summary.totalPnl)}`}>
          <p className="text-xs uppercase tracking-[0.2em]">Total PnL</p>
          <p className="mt-2 text-2xl font-semibold">
            {isFiniteNumber(summary.totalPnl)
              ? `${summary.totalPnl >= 0 ? "+" : ""}${formatNumber(summary.totalPnl, 2)} USDT`
              : "—"}
          </p>
          <p className="mt-1 text-sm opacity-80">
            {isFiniteNumber(summary.totalPnlPct)
              ? `${summary.totalPnlPct >= 0 ? "+" : ""}${formatNumber(summary.totalPnlPct, 2)}% от счета`
              : "—"}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-3xl border border-slate-200 bg-white p-4">
        <div className="text-sm text-slate-600">
          Выбрано: <span className="font-semibold text-slate-900">{selectedRows.length}</span> из{" "}
          <span className="font-semibold text-slate-900">{filteredResults.length}</span>
          {csvSaveStatus ? <span className="ml-3 text-xs text-slate-500">{csvSaveStatus}</span> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => toggleAllFiltered(true)}
            className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Выбрать все
          </button>
          <button
            type="button"
            onClick={() => setSelectedIds(new Set())}
            className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Снять выбор
          </button>
          <button
            type="button"
            onClick={exportSelectedCsv}
            disabled={filteredResults.length === 0}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Скачать CSV
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
          Загружаю результаты торговли...
        </div>
      ) : error ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-700">
          Ошибка загрузки: {error}
        </div>
      ) : filteredResults.length === 0 ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
          Закрытых сделок пока нет.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-3xl border border-slate-200 bg-white">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr className="text-left text-xs uppercase tracking-[0.18em] text-slate-500">
                <th className="px-4 py-3 font-semibold">
                  <input
                    type="checkbox"
                    checked={allFilteredSelected}
                    onChange={(event) => toggleAllFiltered(event.target.checked)}
                    aria-label="Выбрать все сделки"
                    className="h-4 w-4 rounded border-slate-300"
                  />
                </th>
                <th className="px-4 py-3 font-semibold">Символ</th>
                <th className="px-4 py-3 font-semibold">Модель</th>
                <th className="px-4 py-3 font-semibold">Аккаунт</th>
                <th className="px-4 py-3 font-semibold">Тип</th>
                <th className="px-4 py-3 font-semibold">Вход</th>
                <th className="px-4 py-3 font-semibold">Выход</th>
                <th className="px-4 py-3 font-semibold">Цена входа</th>
                <th className="px-4 py-3 font-semibold">Цена выхода</th>
                <th className="px-4 py-3 font-semibold">Размер</th>
                <th className="px-4 py-3 font-semibold">Причина</th>
                <th className="px-4 py-3 font-semibold">P&L</th>
                <th className="px-4 py-3 font-semibold">P&L %</th>
                <th className="px-4 py-3 font-semibold">MAE / MFE</th>
                <th className="px-4 py-3 font-semibold">SHAP top-5</th>
                <th className="px-4 py-3 font-semibold">Длительность</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredResults.map((item) => (
                <tr key={item.id} className="align-top">
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(item.id)}
                      onChange={(event) => toggleRow(item.id, event.target.checked)}
                      aria-label={`Выбрать сделку ${item.symbol}`}
                      className="h-4 w-4 rounded border-slate-300"
                    />
                  </td>
                  <td className="px-4 py-3 font-semibold text-slate-900">{item.symbol || "—"}</td>
                  <td className="px-4 py-3">
                    <div className="flex flex-col gap-2">
                      <span
                        className={`w-fit rounded-full border px-2.5 py-1 text-xs font-semibold ${familyTone(item.modelFamily)}`}
                      >
                        {item.modelFamily === "xgb"
                          ? "XGB"
                          : item.modelFamily === "dqn"
                            ? "DQN / SAC"
                            : "UNKNOWN"}
                      </span>
                      <span className="text-slate-700">{compactModelName(item.modelPath)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {item.accountLabel || (item.accountId ? `Account ${item.accountId}` : "—")}
                  </td>
                  <td className="px-4 py-3 text-slate-700">{item.side || "—"}</td>
                  <td className="px-4 py-3 text-slate-700">{formatDateTime(item.entryTime)}</td>
                  <td className="px-4 py-3 text-slate-700">{formatDateTime(item.exitTime)}</td>
                  <td className="px-4 py-3 text-slate-700">
                    {isFiniteNumber(item.entryPrice)
                      ? `${formatNumber(item.entryPrice, 4)} USDT`
                      : "—"}
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {isFiniteNumber(item.exitPrice)
                      ? `${formatNumber(item.exitPrice, 4)} USDT`
                      : "—"}
                  </td>
                  <td className="px-4 py-3 text-slate-700">{formatNumber(item.quantity, 6)}</td>
                  <td className="px-4 py-3 text-slate-700">{prettyExitReason(item.exitReason)}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${pnlTone(item.pnl)}`}
                    >
                      {isFiniteNumber(item.pnl)
                        ? `${item.pnl >= 0 ? "+" : ""}${formatNumber(item.pnl, 2)}`
                        : "—"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {isFiniteNumber(item.pnlPct) ? (
                      <div className="space-y-1 whitespace-nowrap">
                        <div>{`${item.pnlPct >= 0 ? "+" : ""}${formatNumber(item.pnlPct, 2)}%`}</div>
                        {isFiniteNumber(item.leverage) && item.leverage > 1 && isFiniteNumber(item.pnlPctLeveraged) ? (
                          <div className="text-xs text-slate-500">
                            {`x${formatNumber(item.leverage, 0)}: ${formatSignedNumber(item.pnlPctLeveraged, 2)}%`}
                            {isFiniteNumber(item.marginPnl) ? `, ${formatSignedNumber(item.marginPnl, 2)} USDT` : ""}
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {isFiniteNumber(item.mae) || isFiniteNumber(item.mfe)
                      ? `${formatSignedNumber(item.mae, 2)}% / ${formatSignedNumber(item.mfe, 2)}%`
                      : "—"}
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {item.xgbShap?.top_features?.length ? (
                      <div className="min-w-64 space-y-1 text-xs">
                        {item.xgbShap.top_features.slice(0, 5).map((feature, index) => (
                          <div key={`${feature.feature_name ?? "feature"}-${index}`} className="flex justify-between gap-3">
                            <span className="max-w-48 truncate" title={feature.feature_name ?? ""}>
                              {feature.feature_name || "feature"}
                            </span>
                            <span className={isFiniteNumber(feature.shap_value) && feature.shap_value < 0 ? "text-rose-600" : "text-emerald-700"}>
                              {formatSignedNumber(feature.shap_value, 5)}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-4 py-3 text-slate-700">
                    {isFiniteNumber(item.durationMin)
                      ? `${formatNumber(item.durationMin, 1)} мин`
                      : "—"}
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
