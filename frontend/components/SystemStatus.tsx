import React, { useState } from "react";
import { apiRequest } from "../lib/api";
import { HealthResponse, ReadyResponse } from "../lib/types";

interface SystemStatusProps {
  baseUrl: string;
  onReadyChange: (isReady: boolean) => void;
}

export default function SystemStatus({ baseUrl, onReadyChange }: SystemStatusProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readyData, setReadyData] = useState<ReadyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkSystem = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hRes, rRes] = await Promise.all([
        apiRequest<HealthResponse>(baseUrl, "/health"),
        apiRequest<ReadyResponse>(baseUrl, "/ready"),
      ]);
      setHealth(hRes);
      setReadyData(rRes);
      onReadyChange(rRes.ready);
    } catch (err: any) {
      setError(err.message);
      onReadyChange(false);
      setHealth(null);
      setReadyData(null);
    } finally {
      setLoading(false);
    }
  };

  const modelLabels: Record<string, string> = {
    superclass: "Superclass",
    localization: "Lokalizasyon",
    xgb: "XGBoost",
    thresholds: "Esikler",
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
          Sistem Durumu
        </h2>
        <button
          onClick={checkSystem}
          disabled={loading}
          className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1.5 rounded text-sm disabled:opacity-50 transition flex items-center gap-2 font-medium"
        >
          {loading && (
            <svg
              className="animate-spin h-3.5 w-3.5 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          )}
          {loading ? "Kontrol Ediliyor..." : "Baglanti Kontrol Et"}
        </button>
      </div>

      <div className="p-4 space-y-3">
        {error && (
          <div className="bg-red-500/10 text-red-500 dark:text-red-400 p-3 rounded text-sm border border-red-500/20">
            <strong>Baglanti Hatasi:</strong> {error}
          </div>
        )}

        {health && (
          <div className="flex items-center gap-2 text-sm">
            <span className="h-2.5 w-2.5 rounded-full bg-green-500" />
            <span className="text-slate-700 dark:text-slate-300 font-medium">
              Sunucu Aktif
            </span>
            <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono ml-auto">
              {health.timestamp}
            </span>
          </div>
        )}

        {readyData && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 mb-3">
              <span
                className={`text-xs font-bold px-2 py-1 rounded ${
                  readyData.ready
                    ? "bg-green-500/20 text-green-500 dark:text-green-400"
                    : "bg-amber-500/20 text-amber-500 dark:text-amber-400"
                }`}
              >
                {readyData.ready ? "Hazir" : "Hazir Degil"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {Object.entries(readyData.models_loaded).map(([key, loaded]) => (
                <div
                  key={key}
                  className={`flex items-center gap-2 text-xs p-2 rounded border ${
                    loaded
                      ? "border-green-500/20 bg-green-500/5 text-green-600 dark:text-green-400"
                      : "border-red-500/20 bg-red-500/5 text-red-600 dark:text-red-400"
                  }`}
                >
                  <span
                    className={`h-2 w-2 rounded-full flex-shrink-0 ${
                      loaded ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span className="font-medium">{modelLabels[key] || key}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {!health && !readyData && !error && (
          <p className="text-sm text-slate-400 dark:text-slate-500 italic text-center py-4">
            Sistem durumunu kontrol etmek icin butona basin.
          </p>
        )}
      </div>
    </div>
  );
}
