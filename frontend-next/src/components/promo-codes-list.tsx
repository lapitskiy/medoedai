"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { PromoCode } from "@/lib/backend";

type PromoCodesListProps = {
  promoCodes: PromoCode[];
};

type GroupedPromoCode = {
  key: string;
  note: string;
  createdAtStr: string;
  createdAtDate: Date;
  codes: PromoCode[];
  durationDays: number;
  totalCodes: number;
  usedCodes: number;
  isActive: boolean;
};

function formatDate(value: string | null) {
  if (!value) {
    return "—";
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

function getGroupKey(code: PromoCode): string {
  const date = new Date(code.createdAt);
  // Группируем с точностью до минуты
  const timeKey = date.toISOString().substring(0, 16);
  const noteKey = code.note || "Без заметки";
  return `${timeKey}_${noteKey}_${code.durationDays}`;
}

export function PromoCodesList({ promoCodes }: PromoCodesListProps) {
  const router = useRouter();
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [isDeleting, setIsDeleting] = useState<Record<number, boolean>>({});

  const toggleGroup = (key: string) => {
    setExpandedGroups((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const copyCodes = async (codes: PromoCode[], e: React.MouseEvent) => {
    e.stopPropagation();
    const textToCopy = codes.map((c) => c.code).join("\n");
    try {
      await navigator.clipboard.writeText(textToCopy);
      alert(`Скопировано ${codes.length} кодов`);
    } catch (err) {
      console.error("Failed to copy codes:", err);
      alert("Не удалось скопировать коды");
    }
  };

  const deleteCode = async (id: number) => {
    if (!confirm("Удалить этот промокод?")) {
      return;
    }

    setIsDeleting((prev) => ({ ...prev, [id]: true }));

    try {
      const response = await fetch(`/api/bot/promo-codes/${id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete promo code");
      }

      router.refresh();
    } catch (err) {
      console.error("Delete error:", err);
      alert("Ошибка при удалении промокода");
    } finally {
      setIsDeleting((prev) => ({ ...prev, [id]: false }));
    }
  };

  const deleteGroup = async (codes: PromoCode[], e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Удалить все ${codes.length} кодов в этой партии?`)) {
      return;
    }

    for (const code of codes) {
      setIsDeleting((prev) => ({ ...prev, [code.id]: true }));
    }

    try {
      await Promise.all(
        codes.map((code) =>
          fetch(`/api/bot/promo-codes/${code.id}`, { method: "DELETE" })
        )
      );
      router.refresh();
    } catch (err) {
      console.error("Batch delete error:", err);
      alert("Ошибка при массовом удалении");
    } finally {
      const newDeleting = { ...isDeleting };
      for (const code of codes) {
        delete newDeleting[code.id];
      }
      setIsDeleting(newDeleting);
    }
  };

  if (promoCodes.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 p-8 text-center text-slate-500 bg-slate-50">
        Промокодов пока нет.
      </div>
    );
  }

  const groupsMap: Record<string, GroupedPromoCode> = {};
  for (const code of promoCodes) {
    const key = getGroupKey(code);
    if (!groupsMap[key]) {
      groupsMap[key] = {
        key,
        note: code.note || "—",
        createdAtStr: formatDate(code.createdAt),
        createdAtDate: new Date(code.createdAt),
        durationDays: code.durationDays,
        codes: [],
        totalCodes: 0,
        usedCodes: 0,
        isActive: code.isActive,
      };
    }
    groupsMap[key].codes.push(code);
    groupsMap[key].totalCodes += 1;
    if (code.usedCount > 0) {
      groupsMap[key].usedCodes += 1;
    }
  }

  const groups = Object.values(groupsMap).sort(
    (a, b) => b.createdAtDate.getTime() - a.createdAtDate.getTime()
  );

  return (
    <div className="space-y-4">
      {groups.map((group) => {
        const isExpanded = expandedGroups[group.key];

        return (
          <div
            key={group.key}
            className="overflow-hidden rounded-2xl border border-slate-200 bg-white"
          >
            {/* Header (Group Summary) */}
            <div
              className="flex cursor-pointer items-center justify-between bg-slate-50 px-5 py-4 transition hover:bg-slate-100"
              onClick={() => toggleGroup(group.key)}
            >
              <div className="flex items-center gap-4">
                <button
                  type="button"
                  className="text-slate-400 hover:text-slate-600 focus:outline-none"
                >
                  <svg
                    className={`h-5 w-5 transform transition-transform ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
                <div>
                  <h3 className="text-base font-medium text-slate-900">
                    Партия: {group.note}
                  </h3>
                  <div className="mt-1 flex items-center gap-3 text-sm text-slate-500">
                    <span>{group.createdAtStr}</span>
                    <span>•</span>
                    <span className="font-medium text-indigo-600">
                      {group.totalCodes} шт
                    </span>
                    <span>•</span>
                    <span>Срок: {group.durationDays} дн.</span>
                    <span>•</span>
                    <span>Использовано: {group.usedCodes}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={(e) => copyCodes(group.codes, e)}
                  className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  Копировать все
                </button>
                <button
                  onClick={(e) => deleteGroup(group.codes, e)}
                  className="rounded-md border border-red-200 bg-white px-3 py-1.5 text-sm font-medium text-red-600 shadow-sm transition hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  Удалить партию
                </button>
              </div>
            </div>

            {/* Expanded List of Codes */}
            {isExpanded && (
              <div className="border-t border-slate-200">
                <table className="min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                    <tr>
                      <th className="px-5 py-3 font-semibold">Промокод</th>
                      <th className="px-5 py-3 font-semibold text-center">
                        Использовано
                      </th>
                      <th className="px-5 py-3 font-semibold text-center">
                        Статус
                      </th>
                      <th className="px-5 py-3 font-semibold text-right">
                        Действия
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 bg-white">
                    {group.codes.map((code) => (
                      <tr key={code.id} className="hover:bg-slate-50">
                        <td className="px-5 py-3 font-mono font-medium text-slate-900">
                          {code.code}
                        </td>
                        <td className="px-5 py-3 text-center text-slate-700">
                          {code.usedCount} / {code.maxUses}
                        </td>
                        <td className="px-5 py-3 text-center">
                          {code.isActive ? (
                            <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
                              Активен
                            </span>
                          ) : (
                            <span className="inline-flex items-center rounded-full bg-red-50 px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-600/10">
                              Отключен
                            </span>
                          )}
                        </td>
                        <td className="px-5 py-3 text-right">
                          <button
                            type="button"
                            disabled={isDeleting[code.id]}
                            onClick={() => deleteCode(code.id)}
                            className="text-red-500 hover:text-red-700 transition disabled:opacity-50"
                            title="Удалить промокод"
                          >
                            <svg
                              className="h-5 w-5 inline-block"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                              />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
