import Link from "next/link";
import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { getSidebarData, getXgbWfRuns } from "@/lib/backend";

export const metadata: Metadata = {
  title: "WF XGB",
};

function formatNumber(value: unknown, digits = 4): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

function compactPath(value: string | null | undefined, parts = 6): string {
  if (!value) {
    return "—";
  }
  return value.replace(/\\/g, "/").split("/").filter(Boolean).slice(-parts).join("/");
}

export default async function WfXgbPage() {
  const [sidebarData, wfRuns] = await Promise.all([getSidebarData(), getXgbWfRuns()]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">OOS</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">WF XGB</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Отдельная страница под Walk Forward. Сюда должны попадать кандидаты,
          которые вы сохранили кнопкой <span className="font-medium">Copy to WF</span> из `OOS`.
        </p>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6">
        <p className="text-sm leading-7 text-slate-600">
          Архив кандидатов сохраняется в `result/wf/xgb/&lt;symbol&gt;/&lt;run_name&gt;`.
        </p>
        <div className="mt-6 rounded-2xl border border-slate-200">
          <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800">
            Модели в WF архиве: {wfRuns.length}
          </div>
          {wfRuns.length === 0 ? (
            <div className="px-4 py-5 text-sm text-slate-500">
              В архиве пока нет моделей. Выберите run в `OOS` и нажмите `Copy to WF`.
            </div>
          ) : (
            <div className="max-h-[70vh] overflow-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead className="sticky top-0 bg-white text-slate-600">
                  <tr>
                    <th className="border-b border-slate-200 px-3 py-3">Symbol</th>
                    <th className="border-b border-slate-200 px-3 py-3">Run</th>
                    <th className="border-b border-slate-200 px-3 py-3">Task</th>
                    <th className="border-b border-slate-200 px-3 py-3">Dir</th>
                    <th className="border-b border-slate-200 px-3 py-3">val_acc</th>
                    <th className="border-b border-slate-200 px-3 py-3">f1(1)</th>
                    <th className="border-b border-slate-200 px-3 py-3">pred_non_hold</th>
                    <th className="border-b border-slate-200 px-3 py-3">Source path</th>
                  </tr>
                </thead>
                <tbody className="bg-white">
                  {wfRuns.map((run) => (
                    <tr key={run.resultDir}>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">{run.symbol}</td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">{run.runName}</td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">
                        {run.task || "—"} / {run.direction || "—"}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500">
                        {compactPath(run.resultDir)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">
                        {formatNumber(run.metrics.val_acc, 5)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">
                        {Array.isArray(run.metrics.f1_val) && typeof run.metrics.f1_val[1] === "number"
                          ? formatNumber(run.metrics.f1_val[1], 4)
                          : "—"}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top">
                        {formatNumber(run.metrics.pred_non_hold_rate_val, 4)}
                      </td>
                      <td className="border-b border-slate-100 px-3 py-3 align-top font-mono text-xs text-slate-500">
                        {compactPath(run.sourceRunPath)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <div className="mt-6">
          <Link
            href="/oos_xgb"
            className="inline-flex rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
          >
            Вернуться в XGB OOS
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
