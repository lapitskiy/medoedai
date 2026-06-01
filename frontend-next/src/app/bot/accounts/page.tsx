import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { BotBroadcastForm } from "@/components/bot-broadcast-form";
import {
  getSidebarData,
  getMaxBotAccounts,
  getTelegramBotAccounts,
  type MaxBotAccount,
  type TelegramBotAccount,
} from "@/lib/backend";

export const metadata: Metadata = {
  title: "Бот - Аккаунты",
};

function formatDate(value: string | null) {
  if (!value) {
    return "нет данных";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("ru-RU", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function getAccountName(account: TelegramBotAccount) {
  if (account.displayName) {
    return account.displayName;
  }
  if (account.username) {
    return `@${account.username}`;
  }
  return `Telegram ID ${account.telegramId}`;
}

function getMaxAccountName(account: MaxBotAccount) {
  if (account.displayName) {
    return account.displayName;
  }
  if (account.username) {
    return `@${account.username}`;
  }
  return `MAX user ${account.maxUserId}`;
}

export default async function BotAccountsPage() {
  const [sidebarData, tgAccounts, maxAccounts] = await Promise.all([
    getSidebarData(),
    getTelegramBotAccounts(),
    getMaxBotAccounts(),
  ]);

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-6 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Бот</p>
        <h1 className="mt-2 text-3xl font-semibold text-slate-900">Аккаунты</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
          Пользователи, зарегистрированные через Telegram и MAX ботов (long polling /
          webhook).
        </p>
      </div>

      <BotBroadcastForm />

      <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-slate-900">Telegram</h2>
            <p className="mt-1 text-sm text-slate-500">
              Всего аккаунтов: {tgAccounts.length}
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3 font-semibold">Пользователь</th>
                <th className="px-4 py-3 font-semibold">Telegram ID</th>
                <th className="px-4 py-3 font-semibold">Username</th>
                <th className="px-4 py-3 font-semibold">Ключи Bybit</th>
                <th className="px-4 py-3 font-semibold">Плечо</th>
                <th className="px-4 py-3 font-semibold">Оплачен до</th>
                <th className="px-4 py-3 font-semibold">Активированные промокоды</th>
                <th className="px-4 py-3 font-semibold">Дата регистрации</th>
                <th className="px-4 py-3 font-semibold">Последняя активность</th>
                <th className="px-4 py-3 font-semibold">Статус</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {tgAccounts.length > 0 ? (
                tgAccounts.map((account) => (
                  <tr key={account.identityId} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-medium text-slate-900">
                      {getAccountName(account)}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-700">
                      {account.telegramId}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {account.username ? `@${account.username}` : "нет данных"}
                    </td>
                    <td className="px-4 py-3">
                      {account.hasActiveKeys ? (
                        <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
                          ✅ Активны
                        </span>
                      ) : (
                        <span className="inline-flex items-center rounded-full bg-slate-50 px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-inset ring-slate-500/10">
                          ❌ Нет
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-700 text-center">
                      x{account.leverage}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {account.paidUntil ? (
                        <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-600/20">
                          {formatDate(account.paidUntil)}
                        </span>
                      ) : (
                        <span className="text-slate-400">Не оплачено</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-700 max-w-[200px] truncate" title={account.activePromos.join(", ")}>
                      {account.activePromos.length > 0 ? (
                        account.activePromos.join(", ")
                      ) : (
                        <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {formatDate(account.registeredAt)}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {formatDate(account.lastSeenAt)}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {account.status || "нет данных"}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="px-4 py-6 text-center text-slate-500" colSpan={10}>
                    Telegram-регистраций пока нет.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-slate-900">MAX</h2>
            <p className="mt-1 text-sm text-slate-500">
              Всего аккаунтов: {maxAccounts.length}
            </p>
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-3 font-semibold">Пользователь</th>
                <th className="px-4 py-3 font-semibold">MAX user ID</th>
                <th className="px-4 py-3 font-semibold">Username</th>
                <th className="px-4 py-3 font-semibold">Дата регистрации</th>
                <th className="px-4 py-3 font-semibold">Последняя активность</th>
                <th className="px-4 py-3 font-semibold">Статус</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {maxAccounts.length > 0 ? (
                maxAccounts.map((account) => (
                  <tr key={account.identityId} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-medium text-slate-900">
                      {getMaxAccountName(account)}
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-700">
                      {account.maxUserId}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {account.username ? `@${account.username}` : "нет данных"}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {formatDate(account.registeredAt)}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {formatDate(account.lastSeenAt)}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {account.status || "нет данных"}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="px-4 py-6 text-center text-slate-500" colSpan={6}>
                    Регистраций через MAX пока нет.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  );
}
