import { motion } from "framer-motion";
import { FileText, Clock } from "lucide-react";
import type { AnalysisContext } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EcgWaveformMock } from "./EcgWaveformMock";
import { PrimaryDiagnosisCard } from "./PrimaryDiagnosisCard";
import { ProbabilityChart } from "./ProbabilityChart";
import { ConsistencyGuardCard } from "./ConsistencyGuardCard";
import { LocalizationCard } from "./LocalizationCard";
import { XaiAccordion } from "./XaiAccordion";
import { TechnicalDetails } from "./TechnicalDetails";

const REGION_LEADS: Record<string, string[]> = {
  AMI: ["V3", "V4"],
  ASMI: ["V1", "V2", "V3", "V4"],
  ALMI: ["V3", "V4", "V5", "V6", "I", "aVL"],
  IMI: ["II", "III", "aVF"],
  LMI: ["I", "aVL", "V5", "V6"],
};

function highlightLeads(ctx: AnalysisContext): string[] {
  if (!ctx.localization) return [];
  const top = ctx.localization.regions[0];
  return top ? REGION_LEADS[top] ?? [] : [];
}

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0 },
};

export function EvidencePanel({ ctx, isDemo }: { ctx: AnalysisContext; isDemo: boolean }) {
  const date = new Date(ctx.timestamp);

  return (
    <motion.div
      className="space-y-4"
      initial="hidden"
      animate="show"
      variants={{ show: { transition: { staggerChildren: 0.08 } } }}
    >
      <motion.div variants={item}>
        <Card className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <FileText className="h-4 w-4 text-primary" />
              <span className="font-mono">{ctx.fileName}</span>
            </div>
            {isDemo ? (
              <Badge variant="outline" className="border-[var(--warning)]/40 text-[var(--warning)]">
                Simülasyon — örnek veri
              </Badge>
            ) : (
              <Badge variant="outline" className="border-[var(--success)]/40 text-[var(--success)]">
                Canlı analiz
                {ctx.latencyMs != null ? ` · backend ${(ctx.latencyMs / 1000).toFixed(2)}s` : ""}
              </Badge>
            )}
          </div>
          <div className="mt-1 flex items-center gap-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {date.toLocaleString("tr-TR")}
            </span>
            <span className="font-mono">hash: {ctx.sessionId.slice(0, 10)}</span>
          </div>
          <div className="mt-3">
            <EcgWaveformMock highlightLeads={highlightLeads(ctx)} />
          </div>
        </Card>
      </motion.div>

      <motion.div variants={item}>
        <PrimaryDiagnosisCard ctx={ctx} />
      </motion.div>
      <motion.div variants={item}>
        <ProbabilityChart ctx={ctx} />
      </motion.div>
      {ctx.consistency && (
        <motion.div variants={item}>
          <ConsistencyGuardCard ctx={ctx} />
        </motion.div>
      )}
      {ctx.localization && (
        <motion.div variants={item}>
          <LocalizationCard ctx={ctx} />
        </motion.div>
      )}
      {ctx.xai && (
        <motion.div variants={item}>
          <XaiAccordion ctx={ctx} />
        </motion.div>
      )}
      <motion.div variants={item}>
        <TechnicalDetails ctx={ctx} />
      </motion.div>
    </motion.div>
  );
}
