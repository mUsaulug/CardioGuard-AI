import React from "react";
import { ConsistencyInfo } from "../lib/types";

interface ConsistencyPanelProps {
  consistency: ConsistencyInfo;
}

export default function ConsistencyPanel({ consistency }: ConsistencyPanelProps) {
  const agreementConfig: Record<string, { label: string; className: string }> = {
    AGREE_MI: {
      label: "MI Onaylandı",
      className: "bg-red-500/20 text-red-500 dark:text-red-400 border-red-500/30",
    },
    AGREE_NO_MI: {
      label: "MI Yok",
      className: "bg-green-500/20 text-green-500 dark:text-green-400 border-green-500/30",
    },
    DISAGREE: {
      label: "İnceleme Gerekli",
      className: "bg-amber-500/20 text-amber-500 dark:text-amber-400 border-amber-500/30",
    },
  };

  const triageConfig: Record<string, { label: string; className: string }> = {
    HIGH: {
      label: "YÜKSEK",
      className: "bg-red-500/20 text-red-500 dark:text-red-400",
    },
    LOW: {
      label: "DÜŞÜK",
      className: "bg-green-500/20 text-green-500 dark:text-green-400",
    },
    REVIEW: {
      label: "İNCELEME",
      className: "bg-amber-500/20 text-amber-500 dark:text-amber-400",
    },
  };

  const agreement = agreementConfig[consistency.agreement] || {
    label: consistency.agreement,
    className: "bg-slate-500/20 text-slate-500",
  };

  const triage = triageConfig[consistency.triage_level] || {
    label: consistency.triage_level,
    className: "bg-slate-500/20 text-slate-500",
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
          Tutarlılık Analizi
        </h2>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex gap-3">
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block mb-1">
              Uyum Durumu
            </span>
            <span className={`inline-block text-sm font-bold px-3 py-1.5 rounded-md border ${agreement.className}`}>
              {agreement.label}
            </span>
          </div>
          <div className="flex-1">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block mb-1">
              Triaj Seviyesi
            </span>
            <span className={`inline-block text-sm font-bold px-3 py-1.5 rounded-md ${triage.className}`}>
              {triage.label}
            </span>
          </div>
        </div>

        <div className="space-y-3">
          <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block">
            MI Olasılık Karşılaştırması
          </span>

          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
                <span>Superclass MI</span>
                <span className="font-mono">{(consistency.superclass_mi_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${
                    consistency.superclass_mi_decision ? "bg-red-500" : "bg-slate-400"
                  }`}
                  style={{ width: `${Math.min(consistency.superclass_mi_prob * 100, 100)}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400 mb-1">
                <span>Binary MI</span>
                <span className="font-mono">{(consistency.binary_mi_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${
                    consistency.binary_mi_decision ? "bg-red-500" : "bg-slate-400"
                  }`}
                  style={{ width: `${Math.min(consistency.binary_mi_prob * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {consistency.warnings.length > 0 && (
          <div className="space-y-2">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wide font-bold block">
              Uyarılar
            </span>
            {consistency.warnings.map((warning, idx) => (
              <div
                key={`warning-${idx}-${warning.substring(0,15)}`}
                className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400 bg-amber-500/10 p-2 rounded border border-amber-500/20"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 flex-shrink-0">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                <span>{warning}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
