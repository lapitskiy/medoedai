import type { Metadata } from "next";
import { AppShell } from "@/components/app-shell";
import { getSidebarData, getTelegramBotAccounts } from "@/lib/backend";
import { SupportChat } from "./support-chat";

export const metadata: Metadata = {
  title: "Поддержка пользователя",
};

export default async function SupportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const resolvedParams = await params;
  const [sidebarData, tgAccounts] = await Promise.all([
    getSidebarData(),
    getTelegramBotAccounts(),
  ]);

  const user = tgAccounts.find((a) => a.userId === resolvedParams.id);
  const userName = user?.displayName || user?.username || `ID ${resolvedParams.id}`;

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Поддержка</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">
          Чат с {userName}
        </h1>
      </div>

      <div className="mx-auto max-w-4xl">
        <SupportChat userId={resolvedParams.id} />
      </div>
    </AppShell>
  );
}
