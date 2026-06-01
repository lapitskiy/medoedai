import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { TradingResultsView } from "@/components/trading-results-view";
import { getBybitAccounts, getSidebarData } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Результаты торговли",
};

export default async function TradingResultsPage() {
  const [sidebarData, accounts] = await Promise.all([getSidebarData(), getBybitAccounts()]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
          Trading
        </p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">
          Результаты торговли
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Новая страница в `Next`: слева обычное меню, справа список последних
          результатов торговли с переключением между `XGB` и `DQN/SAC`.
        </p>
      </div>

      <TradingResultsView accounts={accounts} />
    </AppShell>
  );
}
