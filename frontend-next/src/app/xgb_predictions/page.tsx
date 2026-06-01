import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbPredictionsView } from "@/components/xgb-predictions-view";
import { getSidebarData, getXgbSymbols } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Предсказания XGB",
};

export default async function XgbPredictionsPage() {
  const [sidebarData, symbols] = await Promise.all([
    getSidebarData(),
    getXgbSymbols(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Предсказания</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Отдельная страница только для XGB предсказаний без перехода в legacy UI.
        </p>
      </div>

      <XgbPredictionsView
        symbols={symbols}
        initialSymbol={symbols[0] ?? ""}
      />
    </AppShell>
  );
}
