import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbOosView } from "@/components/xgb-oos-view";
import { getSidebarData, getXgbOosRuns } from "@/lib/backend";

export const metadata: Metadata = {
  title: "XGB OOS",
};

export default async function XgbOosPage() {
  const [sidebarData, initialRuns] = await Promise.all([
    getSidebarData(),
    getXgbOosRuns(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">OOS</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">XGB OOS</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Отдельный Next UI для OOS-оценки XGB run: запуск одиночных и batch
          тестов, сохранение CSV и управление run без перехода в legacy шаблон.
        </p>
      </div>

      <XgbOosView initialRuns={initialRuns} />
    </AppShell>
  );
}
