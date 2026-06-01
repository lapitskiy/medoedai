import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbExperimentsTable } from "@/components/xgb-experiments-table";
import { getSidebarData, getXgbExperiments } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Эксперименты XGB",
};

export default async function XgbExperimentsPage() {
  const [sidebarData, experiments] = await Promise.all([
    getSidebarData(),
    getXgbExperiments(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Эксперименты</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Сохраненные OOS эксперименты из `predict_test/xgb_hypo/experiments`:
          snapshot preset и top-модели из CSV.
        </p>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6">
        <XgbExperimentsTable initialExperiments={experiments} />
      </div>
    </AppShell>
  );
}
