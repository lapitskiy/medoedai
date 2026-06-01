"use client";

import { useState } from "react";

export function BotBroadcastForm() {
  const [target, setTarget] = useState("all");
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "success" | "error">("idle");
  const [statusText, setStatusText] = useState("");

  const handleSend = async () => {
    if (!message.trim()) {
      setStatus("error");
      setStatusText("Сообщение не может быть пустым.");
      return;
    }

    setStatus("sending");
    setStatusText("Отправка...");

    try {
      const res = await fetch("/api/bot/broadcast", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ target, message }),
      });

      const data = await res.json();
      if (data.success) {
        setStatus("success");
        setStatusText(`Успешно отправлено ${data.sent_count} пользователям (из ${data.total_targeted}).`);
        setMessage("");
      } else {
        setStatus("error");
        setStatusText(`Ошибка: ${data.error}`);
      }
    } catch (err: any) {
      setStatus("error");
      setStatusText(`Ошибка сети: ${err.message}`);
    }
  };

  return (
    <section className="mb-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-xl font-semibold text-slate-900">Рассылка сообщений</h2>
      <div className="space-y-4 max-w-2xl">
        <div>
          <label className="mb-1 block text-sm font-medium text-slate-700">
            Кому отправить
          </label>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="block w-full rounded-xl border border-slate-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">Всем пользователям</option>
            <option value="active">Только с активной подпиской</option>
            <option value="inactive">Только без активной подписки</option>
          </select>
        </div>
        
        <div>
          <label className="mb-1 block text-sm font-medium text-slate-700">
            Сообщение (поддерживает HTML-теги Telegram: &lt;b&gt;, &lt;i&gt;, &lt;code&gt;, &lt;a&gt;)
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={5}
            className="block w-full rounded-xl border border-slate-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            placeholder="Введите текст сообщения..."
          />
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={handleSend}
            disabled={status === "sending"}
            className="rounded-full bg-blue-600 px-6 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
          >
            {status === "sending" ? "Отправка..." : "Отправить"}
          </button>
          
          {status !== "idle" && (
            <p
              className={`text-sm ${
                status === "success" ? "text-emerald-600" : "text-red-600"
              }`}
            >
              {statusText}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
