import { Sidebar } from "@/components/sidebar";
import type { SidebarData } from "@/lib/backend";
import type { ReactNode } from "react";

type AppShellProps = {
  sidebarData: SidebarData;
  children: ReactNode;
};

export function AppShell({ sidebarData, children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <div className="flex min-h-screen w-full">
        <Sidebar {...sidebarData} />

        <main className="min-h-screen flex-1 p-3 pt-16 lg:ml-72 lg:p-6">
          <div className="min-h-[calc(100vh-1.5rem)] rounded-[28px] border border-slate-200 bg-white p-6 shadow-sm lg:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
