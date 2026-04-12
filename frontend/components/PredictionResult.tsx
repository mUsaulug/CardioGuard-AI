import React from "react";
import { SuperclassResponse } from "../lib/types";
import ProbabilityChart from "./ProbabilityChart";

interface PredictionResultProps {
  result: SuperclassResponse;
}

export default function PredictionResult({ result }: PredictionResultProps) {
  const confidenceColor =
    result.primary.confidence > 0.8
      ? "bg-green-500/20 text-green-500 dark:text-green-400"
      : result.primary.confidence > 0.5
      ? "bg-amber-500/20 text-amber-500 dark:text-amber-400"
      : "bg-red-500/20 text-red-500 dark:text-red-400";

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
          Birincil Tani
        </h2>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex justify-between items-start">
          <div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400 block uppercase tracking-wide font-bold mb-1">
              Tani
            </span>
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-extrabold text-slate-900 dark:text-slate-100">
                {result.primary.label}
              </span>
              <span className={`text-sm font-bold px-2.5 py-1 rounded-full ${confidenceColor}`}>
                {(result.primary.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-mono">
              Kural: {result.primary.rule}
            </div>
          </div>

          <div className="text-right max-w-[50%]">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 block uppercase tracking-wide font-bold mb-2">
              Tahmin Edilen Etiketler
            </span>
            <div className="flex flex-wrap gap-1 justify-end">
              {result.predicted_labels.map((l) => (
                <span
                  key={l}
                  className="bg-blue-500 text-white text-xs px-2.5 py-1 rounded shadow-sm font-medium"
                >
                  {l}
                </span>
              ))}
              {result.predicted_labels.length === 0 && (
                <span className="text-slate-400 dark:text-slate-500 text-xs italic">
                  Bulgu yok
                </span>
              )}
            </div>
          </div>
        </div>

        <ProbabilityChart
          probabilities={result.probabilities}
          thresholds={result.thresholds}
        />

        <div className="text-[10px] text-slate-400 dark:text-slate-500 bg-slate-50 dark:bg-slate-900 p-2 rounded border border-slate-200 dark:border-slate-700 font-mono flex justify-between">
          <span>v{result.versions.api_version}</span>
          <span>Model: {result.versions.model_hash.substring(0, 8)}</span>
        </div>
      </div>
    </div>
  );
}
