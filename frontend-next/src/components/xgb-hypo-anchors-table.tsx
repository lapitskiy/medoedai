"use client";

import type { XgbShortAnchorData } from "@/lib/backend";

type XgbHypoAnchorsTableProps = {
  data: XgbShortAnchorData | null;
};

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat("ru-RU", {
    maximumFractionDigits: digits,
  }).format(value);
}

export function XgbHypoAnchorsTable({ data }: XgbHypoAnchorsTableProps) {
  if (!data || !data.models || data.models.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200">
        <div className="px-4 py-5 text-sm text-slate-500">
          Данные отсутствуют или файл пуст.
        </div>
      </div>
    );
  }

  const models = data.models;

  return (
    <div className="rounded-2xl border border-slate-200">
      <div className="border-b border-slate-200 bg-slate-50 px-4 py-3 text-sm font-semibold text-slate-800 flex justify-between">
        <span>Anchors: {models.length}</span>
        <span className="font-normal text-slate-500">
          Символ: {data.symbol || "-"} | Task: {data.task || "-"}
        </span>
      </div>
      
      <div className="max-h-[72vh] overflow-auto">
        <table className="min-w-full border-collapse text-left text-sm whitespace-nowrap">
          <thead className="sticky top-0 bg-white text-slate-600 shadow-sm">
            <tr>
              <th className="border-b border-slate-200 px-3 py-3" colSpan={4}>Идентификация</th>
              <th className="border-b border-slate-200 border-l px-3 py-3 bg-slate-50" colSpan={9}>Train (из compare_keys)</th>
              <th className="border-b border-slate-200 border-l px-3 py-3" colSpan={3}>Val</th>
              <th className="border-b border-slate-200 border-l px-3 py-3 bg-slate-50" colSpan={5}>OOS (90d / first batch)</th>
            </tr>
            <tr className="text-xs">
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Run Name</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Status</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Rank Note</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Grid ID</th>
              
              <th className="border-b border-slate-200 border-l px-3 py-2 font-medium bg-slate-50">Hold</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Min Prof %</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">TP / SL / Trail %</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Depth</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">LR</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Reg λ</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Min Child Wt</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">ColSample</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">p_enter th</th>
              
              <th className="border-b border-slate-200 border-l px-3 py-2 font-medium">Val Acc</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Val PnL (sum)</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium">Val F1</th>
              
              <th className="border-b border-slate-200 border-l px-3 py-2 font-medium bg-slate-50">ROI %</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">PF</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">MaxDD %</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Trades</th>
              <th className="border-b border-slate-200 px-3 py-2 font-medium bg-slate-50">Best p_enter</th>
            </tr>
          </thead>
          <tbody className="bg-white">
            {models.map((model, idx) => {
              const train = model.train || {};
              const label = train.label || {};
              const xgb = train.xgb || {};
              const exitPolicy = train.exit_policy || {};
              const valMetrics = train.val_metrics || {};
              const paths = model.paths || {};
              
              // OOS logic: either direct oos.90d or first batch's 90d
              let oos90d = (model.oos && model.oos["90d"]) || null;
              if (!oos90d && model.oos && Array.isArray(model.oos.batches) && model.oos.batches.length > 0) {
                oos90d = model.oos.batches[0]["90d"];
              }

              return (
                <tr key={`${model.run_name}-${idx}`} className="hover:bg-slate-50 border-b border-slate-100 last:border-0">
                  <td className="px-3 py-2 font-mono text-xs">{model.run_name || "-"}</td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10">
                      {model.status || "-"}
                    </span>
                  </td>
                  <td className="px-3 py-2 max-w-[200px] truncate" title={model.rank_note}>{model.rank_note || "-"}</td>
                  <td className="px-3 py-2 font-mono text-xs text-slate-500">{paths.grid_id || "-"}</td>
                  
                  <td className="border-l border-slate-100 px-3 py-2 bg-slate-50/50">{label.max_hold_steps ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{label.min_profit_pct ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50 text-xs">
                    {exitPolicy.entry_tp_pct ?? "-"} / {exitPolicy.entry_sl_pct ?? "-"} / {exitPolicy.entry_trail_pct ?? "-"}
                  </td>
                  <td className="px-3 py-2 bg-slate-50/50">{xgb.max_depth ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{xgb.learning_rate ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{xgb.reg_lambda ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{xgb.min_child_weight ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{xgb.colsample_bytree ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{label.p_enter_threshold ?? "-"}</td>
                  
                  <td className="border-l border-slate-100 px-3 py-2">{formatNumber(valMetrics.val_acc, 3)}</td>
                  <td className="px-3 py-2">{formatNumber(valMetrics.proxy_pnl_val_sum, 3)}</td>
                  <td className="px-3 py-2">{formatNumber(valMetrics.f1_enter_val, 3)}</td>
                  
                  <td className="border-l border-slate-100 px-3 py-2 bg-slate-50/50 font-semibold">{formatNumber(oos90d?.roi_pct)}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{formatNumber(oos90d?.profit_factor)}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{formatNumber(oos90d?.max_dd_pct)}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{oos90d?.trades ?? "-"}</td>
                  <td className="px-3 py-2 bg-slate-50/50">{formatNumber(oos90d?.best_p_enter, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {data.description && (
        <div className="border-t border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500">
          {data.description}
        </div>
      )}
    </div>
  );
}
