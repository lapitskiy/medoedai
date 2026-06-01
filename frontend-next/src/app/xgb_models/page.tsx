import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbAgentsMonitor } from "@/components/xgb-agents-monitor";
import { getSidebarData } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Мониторинг XGB",
};

export default async function XgbModelsPage() {
  const sidebarData = await getSidebarData();

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">XGB</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Мониторинг</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Здесь показываются только активные XGB торговые агенты: кто запущен,
          какие правила у него выставлены и активен он сейчас или нет.
        </p>
      </div>

      <XgbAgentsMonitor />
    </AppShell>
  );
}
