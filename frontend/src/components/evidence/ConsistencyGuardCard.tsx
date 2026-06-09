import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import type { AnalysisContext } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { TRIAGE_TR } from "@/lib/glossary";
import { cn } from "@/lib/utils";

function triageStyle(level: string) {
  switch (level) {
    case "HIGH":
      return "bg-[var(--success)]/14 text-[var(--success)]";
    case "MEDIUM":
      return "bg-[var(--warning)]/16 text-[var(--warning)]";
    case "REVIEW":
      return "bg-[var(--destructive)]/14 text-[var(--destructive)]";
    default:
      return "bg-muted text-muted-foreground";
  }
}

/** Human-readable meaning of the agreement code (NOT probability equality). */
function agreementMeaning(code: string): string {
  switch (code) {
    case "AGREE_MI":
      return "İki model de MI POZİTİF dedi — kararlar uyumlu (olasılıkların eşit olması gerekmez).";
    case "AGREE_NO_MI":
      return "İki model de MI NEGATİF dedi — kararlar uyumlu.";
    case "DISAGREE":
      return "Modeller farklı karar verdi — biri MI dedi, diğeri demedi. İnceleme önerilir.";
    default:
      return "İki bağımsız MI modelinin karar uyumu değerlendirilir; karşılaştırılan şey olasılık değil, MI kararıdır (pozitif/negatif).";
  }
}

export function ConsistencyGuardCard({ ctx }: { ctx: AnalysisContext }) {
  const c = ctx.consistency;
  if (!c) return null;
  const agree = c.agreement.startsWith("AGREE");
  const review = c.triage_level === "REVIEW";

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          {agree ? (
            <ShieldCheck className="h-4 w-4 text-[var(--success)]" />
          ) : (
            <ShieldAlert className="h-4 w-4 text-[var(--warning)]" />
          )}
          Consistency Guard
        </div>
        <span className={cn("rounded-full px-2.5 py-1 text-xs font-semibold", triageStyle(c.triage_level))}>
          {TRIAGE_TR[c.triage_level] || c.triage_level}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2.5 py-1 font-mono text-xs text-foreground">
          {c.agreement}
        </span>
        <span className="text-[11px] text-muted-foreground">{agreementMeaning(c.agreement)}</span>
      </div>

      <div className="mt-4 overflow-hidden rounded-lg border border-border text-xs">
        <div className="grid grid-cols-3 bg-muted/40 font-medium text-muted-foreground">
          <span className="px-2.5 py-1.5">Model</span>
          <span className="px-2.5 py-1.5">MI Kararı</span>
          <span className="px-2.5 py-1.5">Olasılık</span>
        </div>
        <div className="grid grid-cols-3 border-t border-border">
          <span className="px-2.5 py-1.5 text-foreground">Superclass</span>
          <span className="px-2.5 py-1.5">{c.superclass_mi_decision ? "Pozitif" : "Negatif"}</span>
          <span className="px-2.5 py-1.5 font-mono">%{(c.superclass_mi_prob * 100).toFixed(1)}</span>
        </div>
        <div className="grid grid-cols-3 border-t border-border">
          <span className="px-2.5 py-1.5 text-foreground">Binary</span>
          <span className="px-2.5 py-1.5">{c.binary_mi_decision ? "Pozitif" : "Negatif"}</span>
          <span className="px-2.5 py-1.5 font-mono">%{(c.binary_mi_prob * 100).toFixed(1)}</span>
        </div>
        <div
          className={cn(
            "grid grid-cols-1 border-t border-border px-2.5 py-1.5 font-medium",
            agree ? "text-[var(--success)]" : "text-[var(--warning)]",
          )}
        >
          Sonuç: {agree ? "Kararlar uyumlu" : "Kararlar farklı — inceleme önerilir"}
        </div>
      </div>

      {(review || c.warnings.length > 0) && (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-[var(--warning)]/12 p-3 text-xs text-[var(--warning)]">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            {review && <p>İnceleme gerekli — modeller arasında belirsizlik mevcut.</p>}
            {c.warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
          </div>
        </div>
      )}
      <p className="mt-3 text-[11px] text-muted-foreground">
        Superclass MI olasılığı ensemble çıktısıdır; Binary MI ayrı bir modeldir — sayılar farklı olabilir,
        önemli olan ikisinin de MI kararı (pozitif/negatif) uyumudur.
      </p>
      {agree && c.warnings.length === 0 && !review && (
        <p className="mt-2 text-xs text-[var(--success)]">
          ✓ İki bağımsız MI modeli uyumlu — yüksek güvenilirlik.
        </p>
      )}
    </Card>
  );
}
