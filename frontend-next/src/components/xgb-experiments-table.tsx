"use client";

import type { XgbExperimentSummary } from "@/lib/backend";

type XgbExperimentsTableProps = {
  initialExperiments: XgbExperimentSummary[];
};

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function compactPath(value: string | null | undefined, parts = 2): string {
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

export function XgbExperimentsTable({ initialExperiments }: XgbExperimentsTableProps) {
  const experiments = initialExperiments;

  return (
    <div className="rounded-2xl border border-slate-200">
      <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800">
        Экспериментов: {experiments.length}
      </div>
      {experiments.length === 0 ? (
        <div className="px-4 py-5 text-sm text-slate-500">
          В `predict_test/xgb_hypo/experiments` пока нет сохраненных XGB экспериментов.
        </div>
      ) : (
        <div className="max-h-[72vh] overflow-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead className="sticky top-0 bg-white text-slate-600">
              <tr>
                <th className="border-b border-slate-200 px-3 py-3">Symbol</th>
                <th className="border-b border-slate-200 px-3 py-3">Experiment</th>
                <th className="border-b border-slate-200 px-3 py-3">Created</th>
                <th className="border-b border-slate-200 px-3 py-3">Top-1 UUID</th>
                <th className="border-b border-slate-200 px-3 py-3">Top-1 ROI</th>
                <th className="border-b border-slate-200 px-3 py-3">Top-1 PF</th>
                <th className="border-b border-slate-200 px-3 py-3">Top-1 MaxDD</th>
                <th className="border-b border-slate-200 px-3 py-3">Top-3 UUID</th>
                <th className="border-b border-slate-200 px-3 py-3">Avg ROI</th>
                <th className="border-b border-slate-200 px-3 py-3">Avg PF</th>
                <th className="border-b border-slate-200 px-3 py-3">Worst DD</th>
                <th className="border-b border-slate-200 px-3 py-3">Trades</th>
                <th className="border-b border-slate-200 px-3 py-3">OOS CSV</th>
                <th className="border-b border-slate-200 px-3 py-3">Path</th>
              </tr>
            </thead>
            <tbody className="bg-white">
              {experiments.map((experiment) => (
                <tr key={experiment.path} className="hover:bg-slate-50">
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-semibold">
                    {experiment.symbol || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs">
                    {experiment.experimentName || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatDate(experiment.createdAt)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs">
                    {experiment.topUuid || "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.topRoiPct, 4)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.topProfitFactor, 4)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.topMaxDd, 6)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs">
                    {experiment.top3Uuids.length ? experiment.top3Uuids.join(", ") : "-"}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.avgRoiTop, 4)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.avgProfitFactorTop, 4)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.worstMaxDdTop, 6)}
                  </td>
                  <td className="border-b border-slate-100 px-3 py-3 align-top">
                    {formatNumber(experiment.tradesSum, 0)}
                  </td>
                  <td
                    className="max-w-[220px] truncate border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500"
                    title={experiment.oosCsv ?? ""}
                  >
                    {compactPath(experiment.oosCsv, 1)}
                  </td>
                  <td
                    className="max-w-[220px] truncate border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500"
                    title={experiment.path}
                  >
                    {compactPath(experiment.path, 2)}
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
