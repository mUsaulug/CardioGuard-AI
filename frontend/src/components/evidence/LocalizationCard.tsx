import { motion } from "framer-motion";
import { Crosshair } from "lucide-react";
import type { AnalysisContext } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const REGION_ORDER = ["AMI", "ASMI", "ALMI", "IMI", "LMI"];

export function LocalizationCard({ ctx }: { ctx: AnalysisContext }) {
  const loc = ctx.localization;
  if (!loc) return null;
  const topRegion = REGION_ORDER.reduce(
    (best, r) => ((loc.probabilities[r] ?? 0) > (loc.probabilities[best] ?? 0) ? r : best),
    REGION_ORDER[0],
  );

  return (
    <Card className="p-5">
      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
        <Crosshair className="h-4 w-4 text-primary" />
        MI Lokalizasyonu
      </div>

      <div className="mt-4 grid grid-cols-[auto_1fr] gap-4">
        {/* Heart diagram */}
        <div className="flex items-center justify-center">
          <svg viewBox="0 0 100 110" className="h-32 w-28" aria-label="Kalp bölge diyagramı">
            <defs>
              <clipPath id="heartClip">
                <path d="M50 100 C 10 70, 5 35, 28 22 C 40 15, 50 25, 50 35 C 50 25, 60 15, 72 22 C 95 35, 90 70, 50 100 Z" />
              </clipPath>
            </defs>
            <g clipPath="url(#heartClip)">
              <rect x="0" y="0" width="100" height="110" fill="var(--muted)" />
              {/* zones */}
              <rect x="30" y="10" width="40" height="35" fill="var(--path-mi)" opacity={zoneOpacity(loc.probabilities.ASMI ?? loc.probabilities.AMI)} />
              <rect x="0" y="10" width="30" height="50" fill="var(--path-mi)" opacity={zoneOpacity(loc.probabilities.ALMI)} />
              <rect x="70" y="10" width="30" height="50" fill="var(--path-mi)" opacity={zoneOpacity(loc.probabilities.LMI)} />
              <rect x="0" y="60" width="100" height="50" fill="var(--path-mi)" opacity={zoneOpacity(loc.probabilities.IMI)} />
            </g>
            <path
              d="M50 100 C 10 70, 5 35, 28 22 C 40 15, 50 25, 50 35 C 50 25, 60 15, 72 22 C 95 35, 90 70, 50 100 Z"
              fill="none"
              stroke="var(--border)"
              strokeWidth="1.5"
            />
          </svg>
        </div>

        {/* Region bars */}
        <div className="space-y-2">
          {REGION_ORDER.map((r) => {
            const v = loc.probabilities[r] ?? 0;
            const isTop = r === topRegion;
            return (
              <div key={r}>
                <div className="flex items-center justify-between text-xs">
                  <span className={cn("font-medium", isTop ? "text-foreground" : "text-muted-foreground")}>
                    {r} · {loc.labels_tr[r]}
                  </span>
                  <span className="font-mono text-muted-foreground">%{(v * 100).toFixed(0)}</span>
                </div>
                <div className="mt-0.5 h-2 w-full overflow-hidden rounded-full bg-muted">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: "var(--path-mi)", opacity: isTop ? 1 : 0.4 }}
                    initial={{ width: 0 }}
                    animate={{ width: `${v * 100}%` }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <p className="mt-3 text-[11px] text-muted-foreground">
        Öne çıkan bölge: <span className="font-medium text-foreground">{loc.labels_tr[topRegion]}</span>
      </p>
    </Card>
  );
}

function zoneOpacity(v?: number) {
  if (!v) return 0.05;
  return Math.min(0.85, 0.15 + v * 0.85);
}
