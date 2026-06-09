import React from "react";

interface SanityBadgeProps {
  sanity: object | null;
}

export default function SanityBadge({ sanity }: SanityBadgeProps) {
  if (!sanity) return null;

  const data = sanity as any;
  const verdict = data.verdict || data.status || "UNKNOWN";
  const passed = data.passed ?? data.checks_passed ?? 0;
  const total = data.total ?? data.checks_total ?? 0;

  const config: Record<string, { label: string; className: string }> = {
    RELIABLE: {
      label: "GÜVENİLİR",
      className: "bg-green-500/20 text-green-500 dark:text-green-400 border-green-500/30",
    },
    ACCEPTABLE: {
      label: "KABUL EDİLEBİLİR",
      className: "bg-amber-500/20 text-amber-500 dark:text-amber-400 border-amber-500/30",
    },
    UNRELIABLE: {
      label: "GÜVENİLMEZ",
      className: "bg-red-500/20 text-red-500 dark:text-red-400 border-red-500/30",
    },
  };

  const style = config[verdict.toUpperCase()] || {
    label: verdict,
    className: "bg-slate-500/20 text-slate-500 border-slate-500/30",
  };

  return (
    <div className="flex items-center gap-2">
      <span className={`text-[10px] font-bold px-2 py-1 rounded border ${style.className}`}>
        {style.label}
      </span>
      {total > 0 && (
        <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
          {passed}/{total}
        </span>
      )}
    </div>
  );
}
