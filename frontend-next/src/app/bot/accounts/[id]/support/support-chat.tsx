"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

type Message = {
  id: number;
  direction: "user_to_admin" | "admin_to_user";
  text: string;
  is_read: boolean;
  created_at: string;
};

export function SupportChat({ userId }: { userId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(true);
  const [replyText, setReplyText] = useState("");
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const fetchMessages = async () => {
    if (!userId) return;
    try {
      const res = await fetch(`/api/bot/support/${userId}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      } else {
        console.error("Failed to fetch messages:", await res.text());
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const markAsRead = async () => {
    if (!userId) return;
    try {
      await fetch(`/api/bot/support/${userId}/read`, { method: "POST" });
      router.refresh(); // Refresh layout to update unread badge
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchMessages();
    markAsRead();
    
    // Poll for new messages every 10 seconds
    const interval = setInterval(fetchMessages, 10000);
    return () => clearInterval(interval);
  }, [userId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyText.trim()) return;

    setSending(true);
    try {
      const res = await fetch(`/api/bot/support/${userId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: replyText }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setMessages((prev) => [...prev, data.message]);
        setReplyText("");
      }
    } catch (e) {
      console.error(e);
      alert("Ошибка при отправке");
    } finally {
      setSending(false);
    }
  };

  if (loading) {
    return <div className="p-8 text-center text-slate-500">Загрузка сообщений...</div>;
  }

  return (
    <div className="flex h-[600px] flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-50">
        {messages.length === 0 ? (
          <div className="text-center text-slate-500 mt-10">Нет сообщений</div>
        ) : (
          messages.map((msg) => {
            const isAdmin = msg.direction === "admin_to_user";
            return (
              <div
                key={msg.id}
                className={`flex ${isAdmin ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                    isAdmin
                      ? "bg-blue-600 text-white rounded-br-sm"
                      : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm">{msg.text}</p>
                  <p
                    className={`mt-1 text-[10px] ${
                      isAdmin ? "text-blue-200" : "text-slate-400"
                    }`}
                  >
                    {new Date(msg.created_at).toLocaleString("ru-RU")}
                  </p>
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-slate-200 bg-white p-4">
        <form onSubmit={handleSend} className="flex gap-3">
          <textarea
            value={replyText}
            onChange={(e) => setReplyText(e.target.value)}
            placeholder="Напишите ответ пользователю..."
            className="flex-1 resize-none rounded-xl border border-slate-300 p-3 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
            rows={3}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend(e);
              }
            }}
          />
          <button
            type="submit"
            disabled={sending || !replyText.trim()}
            className="rounded-xl bg-blue-600 px-6 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {sending ? "..." : "Отправить"}
          </button>
        </form>
        <p className="mt-2 text-xs text-slate-400 text-center">
          Enter — отправить, Shift+Enter — перенос строки
        </p>
      </div>
    </div>
  );
}
