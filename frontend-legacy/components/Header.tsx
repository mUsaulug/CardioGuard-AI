import React from "react";
import { useTheme } from "./ThemeProvider";

interface HeaderProps {
  baseUrl: string;
  onBaseUrlChange: (url: string) => void;
  systemReady: boolean;
}

export default function Header({ baseUrl, onBaseUrlChange, systemReady }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-20 bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 bg-gradient-to-br from-blue-500 to-red-500 rounded-lg shadow-sm flex items-center justify-center text-white font-bold text-lg">
            C
          </div>
          <div>
            <h1 className="text-xl font-bold leading-tight">
              <span className="bg-gradient-to-r from-blue-500 to-red-500 bg-clip-text text-transparent">
                CardioGuard-AI
              </span>
            </h1>
            <p className="text-[10px] text-slate-500 dark:text-slate-400 font-medium tracking-wide uppercase">
              Klinik Karar Destek Sistemi
            </p>
          </div>
          <div className="ml-3 flex items-center gap-1.5">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                systemReady ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
            />
            <span className="text-xs text-slate-500 dark:text-slate-400">
              {systemReady ? "Aktif" : "Bağlantı Yok"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2 bg-slate-100 dark:bg-slate-700 p-1.5 rounded-md border border-slate-200 dark:border-slate-600 flex-1 md:flex-initial">
            <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 whitespace-nowrap px-2">
              API
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => onBaseUrlChange(e.target.value)}
              placeholder="http://localhost:8000"
              className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded px-3 py-1 text-sm text-slate-900 dark:text-slate-100 w-full md:w-64 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition font-mono"
            />
          </div>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-md bg-slate-100 dark:bg-slate-700 border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition"
            title="Tema Değiştir"
          >
            {theme === "dark" ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
