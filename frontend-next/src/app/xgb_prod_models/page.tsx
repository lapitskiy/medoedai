import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbProdModelsTable } from "@/components/xgb-prod-models-table";
import { getSidebarData, getXgbProdModels } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Продакшн XGB",
};

export default async function XgbProdModelsPage() {
  const [sidebarData, models] = await Promise.all([
    getSidebarData(),
    getXgbProdModels(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Модели</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Production XGB модели из `models/xgb`: все symbol, ensemble и версии,
          которые сейчас доступны торговому агенту.
        </p>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6">
        <XgbProdModelsTable initialModels={models} />
      </div>
    </AppShell>
  );
}
