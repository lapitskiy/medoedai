"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export function PromoCodesForm() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccessMessage(null);

    const formData = new FormData(event.currentTarget);
    const count = Number(formData.get("count"));
    const duration_days = Number(formData.get("durationDays"));
    const max_uses = Number(formData.get("maxUses"));
    const note = formData.get("note") as string;

    try {
      const response = await fetch("/api/bot/promo-codes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count, duration_days, max_uses, note }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Не удалось выпустить промокоды");
      }

      setSuccessMessage(`Успешно выпущено кодов: ${data.created}`);
      
      // Сброс формы (опционально)
      // event.currentTarget.reset();
      
      // Обновляем данные на странице (таблицу кодов)
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label htmlFor="count" className="block text-sm font-medium text-slate-700">
            Количество (шт)
          </label>
          <input
            type="number"
            id="count"
            name="count"
            min="1"
            max="100"
            defaultValue="1"
            required
            className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm px-3 py-2 border"
          />
        </div>

        <div>
          <label htmlFor="durationDays" className="block text-sm font-medium text-slate-700">
            Длительность (дней)
          </label>
          <input
            type="number"
            id="durationDays"
            name="durationDays"
            min="1"
            defaultValue="14"
            required
            className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm px-3 py-2 border"
          />
        </div>

        <div>
          <label htmlFor="maxUses" className="block text-sm font-medium text-slate-700">
            Лимит активаций (на код)
          </label>
          <input
            type="number"
            id="maxUses"
            name="maxUses"
            min="1"
            defaultValue="1"
            required
            className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm px-3 py-2 border"
          />
        </div>

        <div>
          <label htmlFor="note" className="block text-sm font-medium text-slate-700">
            Заметка (кому / зачем)
          </label>
          <input
            type="text"
            id="note"
            name="note"
            className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm px-3 py-2 border"
            placeholder="Например: приветственный бонус"
          />
        </div>
      </div>

      {error ? (
        <div className="rounded-md bg-red-50 p-4">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      ) : null}

      {successMessage ? (
        <div className="rounded-md bg-emerald-50 p-4">
          <p className="text-sm text-emerald-700">{successMessage}</p>
        </div>
      ) : null}

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={isSubmitting}
          className="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50"
        >
          {isSubmitting ? "Выпуск..." : "Выпустить промокоды"}
        </button>
      </div>
    </form>
  );
}
