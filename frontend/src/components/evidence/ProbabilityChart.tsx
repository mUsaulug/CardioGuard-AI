import { useState } from "react";
import { motion } from "framer-motion";
import { BarChart3, Info } from "lucide-react";
import type { AnalysisContext, PathologyKey } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { pathColorVar } from "@/components/PathologyBadge";
import { cn } from "@/lib/utils";

type Source = "ensemble" | "cnn" | "xgb";
const ORDER: PathologyKey[] = ["MI", "STTC", "CD", "HYP", "NORM"];

export function ProbabilityChart({ ctx }: { ctx: AnalysisContext }) {
  const [source, setSource] = useState<Source>("ensemble");
  const data = source === "ensemble" ? ctx.probabilities : ctx.sources[source];

  const primaryKey = ctx.primary.label;
  const primaryEnsemble = (ctx.probabilities as Record<string, number>)[primaryKey];
  const primarySource = (data as Record<string, number>)[primaryKey];
  const showDivergence =
    source !== "ensemble" &&
    primaryEnsemble !== undefined &&
    primarySource !== undefined &&
    Math.abs(primaryEnsemble - primarySource) >= 0.01;

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <BarChart3 className="h-4 w-4 text-primary" />
          Olasılık Dağılımı
        </div>
        <div className="flex gap-0.5 rounded-md bg-muted p-0.5">
          {(["ensemble", "cnn", "xgb"] as Source[]).map((s) => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={cn(
                "rounded px-2 py-1 text-[11px] font-medium uppercase transition-colors",
                source === s
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {showDivergence && (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-[var(--warning)]/12 px-3 py-2 text-[11px] text-[var(--warning)]">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            <span className="font-semibold uppercase">{source}</span> kaynağında {primaryKey} %
            {(primarySource * 100).toFixed(1)} — birincil kart ise ensemble (%
            {(primaryEnsemble * 100).toFixed(1)}) gösterir. Fark, kaynak modellerin farklı tahmin
            etmesindendir; karar ensemble'a dayanır.
          </span>
        </div>
      )}

      <div className="mt-4 space-y-3">
        {ORDER.map((key) => {
          const val = (data as Record<string, number>)[key] ?? 0;
          const thr = ctx.thresholds[key];
          const above = thr !== undefined && val >= thr;
          return (
            <div key={key}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-semibold text-foreground">{key}</span>
                <span className={cn("font-mono", above ? "text-foreground" : "text-muted-foreground")}>
                  %{(val * 100).toFixed(1)}
                </span>
              </div>
              <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: pathColorVar(key), opacity: above ? 1 : 0.45 }}
                  initial={{ width: 0 }}
                  animate={{ width: `${val * 100}%` }}
                  transition={{ duration: 0.7, ease: "easeOut" }}
                />
                {thr !== undefined && (
                  <span
                    className="absolute top-0 h-full w-[2px] bg-foreground/70"
                    style={{ left: `${thr * 100}%` }}
                    title={`Eşik: %${(thr * 100).toFixed(1)}`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[11px] text-muted-foreground">
        Kaynak: <span className="font-semibold uppercase text-foreground">{source}</span>.
        {source === "ensemble"
          ? " Birincil tanı kartındaki % değeri bu ensemble dağılımındandır."
          : " CNN/XGB sekmeleri kaynak model olasılıklarını gösterir; birincil karttan farklı olabilir."}
        {" "}Dikey çizgi sınıf eşiğidir.
      </p>
    </Card>
  );
}
