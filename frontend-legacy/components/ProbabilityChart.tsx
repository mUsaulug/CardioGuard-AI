import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { SuperclassProbabilities } from "../lib/types";

interface ProbabilityChartProps {
  probabilities: SuperclassProbabilities;
  thresholds: { MI: number; STTC: number; CD: number; HYP: number };
}

export default function ProbabilityChart({
  probabilities,
  thresholds,
}: ProbabilityChartProps) {
  const data = Object.entries(probabilities).map(([key, value]) => {
    const threshold = (thresholds as any)[key] ?? 0.5;
    return {
      name: key,
      value: Number((value * 100).toFixed(1)),
      threshold: Number((threshold * 100).toFixed(1)),
      isAbove: value > threshold,
    };
  });

  return (
    <div className="bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
      <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300 mb-3">
        Sınıf Olasılıkları (%)
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
          <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: "#94a3b8" }} />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: 12, fontWeight: 600, fill: "#94a3b8" }}
            width={50}
          />
          <Tooltip
            formatter={(value: number) => [`${value}%`, "Olasılık"]}
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              color: "#f1f5f9",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.isAbove ? "#3b82f6" : "#475569"}
              />
            ))}
          </Bar>
          {data.map((entry, index) => (
            <ReferenceLine
              key={index}
              x={entry.threshold}
              stroke="#ef4444"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 mt-2 text-[10px] text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-1">
          <span className="w-3 h-2 bg-blue-500 rounded-sm inline-block" />
          Eşik Üstü
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-2 bg-slate-500 rounded-sm inline-block" />
          Eşik Altı
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-red-500 inline-block" />
          Eşik Değeri
        </div>
      </div>
    </div>
  );
}
