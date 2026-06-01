import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { PromoCodesForm } from "@/components/promo-codes-form";
import { PromoCodesList } from "@/components/promo-codes-list";
import { getSidebarData, getPromoCodes, type PromoCode } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Бот - Промокоды",
};

export default async function PromoCodesPage() {
  const [sidebarData, promoCodes] = await Promise.all([
    getSidebarData(),
    getPromoCodes(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Бот</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Промокоды</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Выпуск и управление промокодами для предоставления бесплатного доступа пользователям бота.
        </p>
      </div>

      <section className="mb-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="mb-5 text-xl font-semibold text-slate-900">Выпуск новых кодов</h2>
        <PromoCodesForm />
      </section>

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Все промокоды</h2>
            <p className="mt-1 text-sm text-slate-500">
              Всего выпущено кодов: {promoCodes.length}
            </p>
          </div>
        </div>

        <PromoCodesList promoCodes={promoCodes} />
      </section>
    </AppShell>
  );
}
