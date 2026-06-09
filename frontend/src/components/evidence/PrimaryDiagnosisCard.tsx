import { motion } from "framer-motion";
import { HeartPulse } from "lucide-react";
import type { AnalysisContext } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { PathologyBadge } from "@/components/PathologyBadge";
import { PATHOLOGY_LABELS_TR } from "@/lib/glossary";
import { cn } from "@/lib/utils";

function confidenceTone(c: number) {
  if (c >= 0.8) return { color: "var(--success)", label: "Yüksek güven", cls: "text-[var(--success)]" };
  if (c >= 0.5) return { color: "var(--warning)", label: "Orta güven", cls: "text-[var(--warning)]" };
  return { color: "var(--destructive)", label: "Düşük güven", cls: "text-[var(--destructive)]" };
}

export function PrimaryDiagnosisCard({ ctx }: { ctx: AnalysisContext }) {
  const tone = confidenceTone(ctx.primary.confidence);
  const pct = (ctx.primary.confidence * 100).toFixed(1);

  return (
    <Card className="relative overflow-hidden p-5">
      <motion.span
        aria-hidden
        className="absolute -right-10 -top-10 h-32 w-32 rounded-full"
        style={{ background: tone.color, opacity: 0.12 }}
        animate={{ scale: [1, 1.15, 1] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <HeartPulse className="h-4 w-4 text-primary" />
          Birincil Tanı
        </div>
        <span
          className={cn("rounded-full px-2.5 py-1 text-xs font-semibold", tone.cls)}
          style={{ background: `color-mix(in oklab, ${tone.color} 14%, transparent)` }}
        >
          {tone.label}
        </span>
      </div>

      <div className="mt-3 flex items-end gap-3">
        <span className="font-display text-4xl font-bold tracking-tight text-foreground">
          {ctx.primary.label}
        </span>
        <span className={cn("mb-1 font-mono text-2xl font-semibold", tone.cls)}>%{pct}</span>
      </div>
      <p className="mt-0.5 text-sm text-muted-foreground">
        {PATHOLOGY_LABELS_TR[ctx.primary.label] || ctx.primary.label}
      </p>
      <p className="mt-1 text-[11px] text-muted-foreground">
        Birincil güven = ensemble olasılığı (MI-first-then-priority kuralı)
      </p>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full"
          style={{ background: tone.color }}
          initial={{ width: 0 }}
          animate={{ width: `${ctx.primary.confidence * 100}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-1.5">
        {ctx.predictedLabels.map((l) => (
          <PathologyBadge key={l} label={l} />
        ))}
      </div>
      <p className="mt-3 font-mono text-[11px] text-muted-foreground">
        Kural: {ctx.primary.rule}
      </p>
    </Card>
  );
}
