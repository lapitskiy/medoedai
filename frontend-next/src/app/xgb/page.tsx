import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { XgbLaunchForm } from "@/components/xgb-launch-form";
import {
  getBybitAccounts,
  getSelectedBybitAccountId,
  getSidebarData,
  getXgbSymbols,
} from "@/lib/backend";

export const metadata: Metadata = {
  title: "Запустить XGB",
};

export default async function XgbPage() {
  const [sidebarData, accounts, selectedAccountId, symbols] = await Promise.all([
    getSidebarData(),
    getBybitAccounts(),
    getSelectedBybitAccountId(),
    getXgbSymbols(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
          XGB
        </p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">
          Запустить
        </h1>
      </div>

      <XgbLaunchForm
        accounts={accounts}
        selectedAccountId={selectedAccountId}
        symbols={symbols}
      />
    </AppShell>
  );
}
