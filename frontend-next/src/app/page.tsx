import type { Metadata } from "next";

import { AppShell } from "@/components/app-shell";
import { getSidebarData } from "@/lib/backend";

export const metadata: Metadata = {
  title: "Overview - Medoed",
};

const migrationCards = [
  {
    title: "Шаблоны",
    text: "Jinja partial `templates/include/_sidebar.html` уже перенесен в React-компонент. Остальные страницы можно переносить блоками и подключать рядом с legacy UI.",
  },
  {
    title: "Backend API",
    text: "Next.js ходит в Flask по внутреннему Docker URL, а legacy страницы открываются по внешнему `http://localhost:5050`.",
  },
  {
    title: "Дальше",
    text: "Следующий нормальный шаг: вынести общий layout, sidebar, таблицы и формы в `src/components`, затем по одной переносить страницы `/settings`, `/models`, `/oos`.",
  },
];

export default async function Home() {
  const sidebarData = await getSidebarData();

  return (
    <AppShell sidebarData={sidebarData}>
      <div className="mb-8 flex flex-col gap-4 border-b border-slate-200 pb-8">
        <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
          Frontend migration
        </p>
        <h1 className="max-w-4xl text-3xl font-semibold text-slate-900 lg:text-4xl">
          Белый layout под Next.js с левым меню и правым блоком контента
        </h1>
        <p className="max-w-3xl text-base leading-8 text-slate-600 lg:text-lg">
          Теперь структура ближе к тому шаблону, который вы прислали:
          фиксированный sidebar слева, мобильный drawer и большая белая
          рабочая область справа под страницы.
        </p>
        <div className="flex flex-wrap gap-3 text-sm text-slate-600">
          <span className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2">
            Next.js: http://localhost:3001
          </span>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-4 py-2">
            Legacy Flask: {sidebarData.backendUrl}
          </span>
        </div>
      </div>

      <section className="grid gap-4 md:grid-cols-3">
        {migrationCards.map((card) => (
          <article
            key={card.title}
            className="rounded-3xl border border-slate-200 bg-slate-50 p-5"
          >
            <h2 className="mb-3 text-lg font-semibold text-slate-900">
              {card.title}
            </h2>
            <p className="text-sm leading-7 text-slate-600">{card.text}</p>
          </article>
        ))}
      </section>

      <section className="mt-6 grid gap-4 lg:grid-cols-[1.4fr_0.6fr]">
        <div className="rounded-3xl border border-slate-200 bg-white p-5">
          <h2 className="mb-3 text-lg font-semibold text-slate-900">
            Контентная зона
          </h2>
          <p className="mb-4 text-sm leading-7 text-slate-600">
            Сюда можно уже переносить реальные страницы. Логика теперь
            разделена: меню живёт отдельно, контент справа отдельно.
          </p>
          <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-500">
            Здесь будет страница `settings`, `models`, `oos` или любой другой
            экран после переноса из Flask.
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
          <h2 className="mb-3 text-lg font-semibold text-slate-900">
            Статус backend
          </h2>
          <p className="text-sm leading-7 text-slate-600">
            {sidebarData.symbolsError
              ? `Не удалось получить активные символы: ${sidebarData.symbolsError}`
              : `Backend доступен, найдено активных символов: ${sidebarData.symbols.length}`}
          </p>
        </div>
      </section>
    </AppShell>
  );
}
