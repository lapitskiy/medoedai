"use client";

import Link from "next/link";
import { useState } from "react";

type SidebarProps = {
  backendUrl: string;
};

type NavItem = {
  label: string;
  href?: string;
  children?: Array<{ label: string; href: string }>;
};

type NavGroup = {
  title: string;
  items: NavItem[];
};

const navGroups: NavGroup[] = [
  {
    title: "Торговый агент",
    items: [
      { label: "Результаты торговли", href: "/trading/results" },
      {
        label: "XGB",
        children: [
          { label: "Запустить", href: "/xgb" },
          { label: "Мониторинг", href: "/xgb_models" },
          { label: "Предсказания", href: "/xgb_predictions" },
        ],
      },
    ],
  },
  {
    title: "Модели",
    items: [
      {
        label: "XGB",
        children: [
          { label: "Обучение", href: "/xgb_training" },
          { label: "Продакшн", href: "/xgb_prod_models" },
          { label: "Эксперименты", href: "/xgb_experiments" },
        ],
      },
    ],
  },
  {
    title: "OOS",
    items: [
      { label: "XGB OOS", href: "/oos_xgb" },
      { label: "WF", href: "/wf_xgb" },
    ],
  },
  {
    title: "Бот",
    items: [
      { label: "Аккаунты", href: "/bot/accounts" },
      { label: "Промокоды", href: "/bot/promo-codes" },
    ],
  },
];

export function Sidebar({ backendUrl }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const legacyUiUrl = backendUrl
    .replace("localhost", "192.168.0.21")
    .replace("127.0.0.1", "192.168.0.21");
  const legacyHref = (path: string) => `${legacyUiUrl.replace(/\/$/, "")}${path}`;
  const navGroupsWithSystem: NavGroup[] = [
    ...navGroups,
    {
      title: "Система",
      items: [
        {
          label: "Настройки",
          children: [
            { label: "Настройки", href: legacyHref("/settings") },
            { label: "API", href: legacyHref("/settings/api") },
            { label: "Боты", href: "/settings/bots" },
            { label: "Работа с энкодерами", href: legacyHref("/settings/encoders") },
          ],
        },
        { label: "Очистить данные", href: legacyHref("/clean_data") },
        { label: "Очистить Redis", href: legacyHref("/clear_redis") },
        { label: "Работа с моделями", href: legacyHref("/system/models") },
        { label: "Адаптивные", href: legacyHref("/system/adaptive") },
        { label: "Гипотезы", href: legacyHref("/system/hypotheses") },
        { label: "Статус задач", href: legacyHref("/task-status/") },
      ],
    },
  ];
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    "Торговый агент": true,
    Модели: true,
    OOS: false,
    Бот: false,
    Система: false,
    "Торговый агент:XGB": true,
    "Модели:XGB": true,
    "Система:Настройки": false,
  });

  const toggleGroup = (group: string) => {
    setOpenGroups((current) => ({
      ...current,
      [group]: !current[group],
    }));
  };

  const closeSidebar = () => setIsOpen(false);
  const renderNavLink = (
    href: string,
    label: string,
    className: string,
  ) => {
    if (href.startsWith("http://") || href.startsWith("https://")) {
      return (
        <a className={className} href={href} onClick={closeSidebar}>
          {label}
        </a>
      );
    }

    return (
      <Link className={className} href={href} onClick={closeSidebar}>
        {label}
      </Link>
    );
  };

  return (
    <>
      <button
        type="button"
        aria-controls="default-sidebar"
        aria-label="Open sidebar"
        onClick={() => setIsOpen(true)}
        className="fixed left-3 top-3 z-50 inline-flex items-center rounded-lg border border-slate-200 bg-white p-2 text-slate-500 shadow-sm transition hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-200 lg:hidden"
      >
        <svg
          className="h-6 w-6"
          aria-hidden="true"
          fill="currentColor"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            clipRule="evenodd"
            fillRule="evenodd"
            d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zm0 10.5a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5h-7.5a.75.75 0 01-.75-.75zM2 10a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 10z"
          />
        </svg>
      </button>

      {isOpen ? (
        <button
          type="button"
          aria-label="Close sidebar overlay"
          onClick={closeSidebar}
          className="fixed inset-0 z-30 bg-slate-900/35 lg:hidden"
        />
      ) : null}

      <aside
        id="default-sidebar"
        aria-label="Sidenav"
        className={`fixed left-0 top-0 z-40 h-screen w-72 border-r border-slate-200 bg-white transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } lg:translate-x-0`}
      >
        <div className="flex h-full flex-col overflow-y-auto px-3 py-5">
          <div className="mb-4 flex items-start justify-between gap-3 px-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                MedoedAI
              </p>
              <h2 className="mt-1 text-lg font-semibold text-slate-900">
                Navigation
              </h2>
            </div>
            <button
              type="button"
              onClick={closeSidebar}
              className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 lg:hidden"
            >
              <span className="sr-only">Close sidebar</span>
              <svg
                className="h-5 w-5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <nav className="space-y-2">
            <ul className="space-y-2">
              <li>
                <Link
                  className="group flex items-center rounded-lg p-2 text-base font-medium text-slate-900 transition hover:bg-slate-100"
                  href="/"
                  onClick={closeSidebar}
                >
                  <span className="mr-3 text-slate-400 transition group-hover:text-slate-900">
                    <svg
                      aria-hidden="true"
                      className="h-6 w-6"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
                      <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
                    </svg>
                  </span>
                  <span>Overview</span>
                </Link>
              </li>
              <li>
                <a
                  className="group flex items-center rounded-lg p-2 text-base font-medium text-slate-900 transition hover:bg-slate-100"
                  href={legacyUiUrl}
                  target="_blank"
                  rel="noreferrer"
                  onClick={closeSidebar}
                >
                  <span className="mr-3 text-slate-400 transition group-hover:text-slate-900">
                    <svg
                      aria-hidden="true"
                      className="h-6 w-6"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7A1 1 0 003 11h1v5a2 2 0 002 2h2a1 1 0 001-1v-3h2v3a1 1 0 001 1h2a2 2 0 002-2v-5h1a1 1 0 00.707-1.707l-7-7z" />
                    </svg>
                  </span>
                  <span>Старый UI</span>
                </a>
              </li>

              {navGroupsWithSystem.map((group) => (
                <li key={group.title}>
                  <button
                    type="button"
                    onClick={() => toggleGroup(group.title)}
                    className="group flex w-full items-center rounded-lg p-2 text-base font-normal text-slate-900 transition hover:bg-slate-100"
                  >
                    <span className="mr-3 text-slate-400 transition group-hover:text-slate-900">
                      <svg
                        aria-hidden="true"
                        className="h-6 w-6"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          fillRule="evenodd"
                          d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
                          clipRule="evenodd"
                        />
                      </svg>
                    </span>
                    <span className="flex-1 text-left whitespace-nowrap">
                      {group.title}
                    </span>
                    <svg
                      aria-hidden="true"
                      className={`h-5 w-5 text-slate-500 transition-transform ${
                        openGroups[group.title] ? "rotate-180" : ""
                      }`}
                      fill="currentColor"
                      viewBox="0 0 20 20"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        fillRule="evenodd"
                        d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>

                  {openGroups[group.title] ? (
                    <ul className="space-y-1 py-2">
                      {group.items.map((item) => (
                        <li key={item.label}>
                          {item.children?.length ? (
                            <>
                              <button
                                type="button"
                                onClick={() =>
                                  toggleGroup(`${group.title}:${item.label}`)
                                }
                                className="flex w-full items-center rounded-lg py-2 pl-11 pr-2 text-base font-normal text-slate-700 transition hover:bg-slate-100 hover:text-slate-900"
                              >
                                <span className="flex-1 text-left">
                                  {item.label}
                                </span>
                                <svg
                                  aria-hidden="true"
                                  className={`h-4 w-4 text-slate-500 transition-transform ${
                                    openGroups[`${group.title}:${item.label}`]
                                      ? "rotate-180"
                                      : ""
                                  }`}
                                  fill="currentColor"
                                  viewBox="0 0 20 20"
                                  xmlns="http://www.w3.org/2000/svg"
                                >
                                  <path
                                    fillRule="evenodd"
                                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                                    clipRule="evenodd"
                                  />
                                </svg>
                              </button>

                              {openGroups[`${group.title}:${item.label}`] ? (
                                <ul className="space-y-1 py-1">
                                  {item.children.map((child) => (
                                    <li key={child.href}>
                                      {renderNavLink(
                                        child.href,
                                        child.label,
                                        "block rounded-lg py-2 pl-16 pr-2 text-sm font-normal text-slate-600 transition hover:bg-slate-100 hover:text-slate-900",
                                      )}
                                    </li>
                                  ))}
                                </ul>
                              ) : null}
                            </>
                          ) : (
                            renderNavLink(
                              item.href ?? "/",
                              item.label,
                              "block rounded-lg py-2 pl-11 pr-2 text-base font-normal text-slate-700 transition hover:bg-slate-100 hover:text-slate-900",
                            )
                          )}
                        </li>
                      ))}
                    </ul>
                  ) : null}
                </li>
              ))}
            </ul>

          </nav>
        </div>
      </aside>
    </>
  );
}
