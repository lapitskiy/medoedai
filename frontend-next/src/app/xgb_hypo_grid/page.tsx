import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbHypoAnchorsTable } from "@/components/xgb-hypo-anchors-table";
import { getSidebarData, getXgbShortAnchors } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Гипотезы grid",
};

export default async function XgbHypoGridPage() {
  const [sidebarData, anchorsData] = await Promise.all([
    getSidebarData(),
    getXgbShortAnchors(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Гипотезы grid</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Сравнение якорных моделей из обучающих батчей. Данные читаются из{" "}
          <code className="rounded bg-slate-100 px-1 py-0.5 text-xs text-slate-800">
            predict_test/xgb_hypo/xgb_short_model_anchors.json
          </code>
          .
        </p>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6">
        <XgbHypoAnchorsTable data={anchorsData} />
      </div>
    </AppShell>
  );
}
