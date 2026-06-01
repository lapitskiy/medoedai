import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { MaxBotSettingsForm } from "@/components/max-bot-settings-form";
import { NotificationModelsSettingsForm } from "@/components/notification-models-settings-form";
import { TelegramBotSettingsForm } from "@/components/telegram-bot-settings-form";
import { getSidebarData } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Боты",
};

export default async function BotsSettingsPage() {
  const sidebarData = await getSidebarData();

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
          Система / Настройки
        </p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">
          Боты
        </h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Здесь задаются Telegram и MAX API для уведомлений и сигналов.
        </p>
      </div>

      <div className="space-y-6">
        <TelegramBotSettingsForm />
        <MaxBotSettingsForm />
        <NotificationModelsSettingsForm />
      </div>
    </AppShell>
  );
}
