import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbTrainingView } from "@/components/xgb-training-view";
import { getSidebarData } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Обучение XGB",
};

export default async function XgbTrainingPage() {
  const sidebarData = await getSidebarData();

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">
          Обучение
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Полный перенос legacy `xgb_models.html` в Next (без iframe): запуск grid
          задач, Full Grid Search, мониторинг активных обучений и создание
          versions.
        </p>
      </div>

      <XgbTrainingView />
    </AppShell>
  );
}
