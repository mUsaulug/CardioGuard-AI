import React from "react";
import { LocalizationResponse } from "../lib/types";

interface LocalizationPanelProps {
  result: LocalizationResponse;
}

export default function LocalizationPanel({ result }: LocalizationPanelProps) {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
          MI Lokalizasyon
        </h2>
        {result.mi_detected ? (
          <span className="bg-red-500/20 text-red-500 dark:text-red-400 text-xs font-bold px-3 py-1 rounded-md border border-red-500/30 animate-pulse">
            MI TESPIT EDILDI
          </span>
        ) : (
          <span className="bg-green-500/20 text-green-500 dark:text-green-400 text-xs font-bold px-3 py-1 rounded-md border border-green-500/30">
            MI TESPIT EDILMEDI
          </span>
        )}
      </div>

      <div className="p-4 space-y-4">
        <div>
          <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block mb-2">
            Etkilenen Bölgeler
          </span>
          <div className="flex flex-wrap gap-1">
            {result.regions.map((r) => (
              <span
                key={r}
                className="bg-red-500/10 text-red-500 dark:text-red-400 border border-red-500/20 text-xs px-2.5 py-1 rounded font-medium"
              >
                {r}
              </span>
            ))}
            {result.regions.length === 0 && (
              <span className="text-slate-400 dark:text-slate-500 text-xs italic">
                Belirli bir bölge tespit edilmedi
              </span>
            )}
          </div>
        </div>

        <div className="space-y-3">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block">
            Bölge Olasılıkları
          </span>
          {(Object.entries(result.probabilities) as [string, number][]).map(
            ([region, prob]) => {
              const isHigh = result.regions.includes(region);
              return (
                <div key={region}>
                  <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
                    <span className={isHigh ? "font-bold text-red-500 dark:text-red-400" : ""}>
                      {region}
                    </span>
                    <span className="font-mono">{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
                    <div
                      className={`h-2.5 rounded-full transition-all duration-500 ${
                        isHigh ? "bg-red-500" : "bg-slate-400 dark:bg-slate-500"
                      }`}
                      style={{ width: `${Math.min(prob * 100, 100)}%` }}
                    />
                  </div>
                </div>
              );
            }
          )}
        </div>

        <div className="text-[10px] text-slate-400 dark:text-slate-500 border-t border-slate-200 dark:border-slate-700 pt-2 space-y-1 font-mono">
          <div className="flex gap-2">
            <span className="font-bold">Kaynak:</span> {result.mapping_source}
          </div>
          <div className="flex gap-2">
            <span className="font-bold">Bas Tipi:</span> {result.localization_head_type}
          </div>
        </div>
      </div>
    </div>
  );
}
