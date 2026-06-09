import { useState } from "react";
import { Brain, Flame, FileText, CheckCircle2, XCircle, MinusCircle, ExternalLink } from "lucide-react";
import type { AnalysisContext, XaiArtifact } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Markdown } from "@/components/Markdown";

function StylizedPreview({ note }: { note: string }) {
  return (
    <div className="ecg-grid flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-border">
      <div className="flex flex-col items-center gap-2 text-center">
        <div
          className="h-16 w-40 rounded-md"
          style={{
            background:
              "radial-gradient(circle at 60% 50%, var(--path-mi) 0%, transparent 65%), linear-gradient(90deg, transparent, var(--warning) 60%, transparent)",
            opacity: 0.7,
          }}
        />
        <p className="px-4 text-xs text-muted-foreground">{note}</p>
      </div>
    </div>
  );
}

function ArtifactImage({ artifact, fallbackNote }: { artifact: XaiArtifact | null; fallbackNote: string }) {
  const [errored, setErrored] = useState(false);
  if (!artifact || errored) {
    return <StylizedPreview note={fallbackNote} />;
  }
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <img
        src={artifact.url}
        alt="XAI raporu (Grad-CAM + SHAP)"
        className="w-full"
        loading="lazy"
        onError={() => setErrored(true)}
      />
      <a
        href={artifact.url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center justify-end gap-1 border-t border-border px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground"
      >
        <ExternalLink className="h-3 w-3" /> Tam boyutta aç
      </a>
    </div>
  );
}

function CoherenceGauge({ value }: { value: number }) {
  const r = 26;
  const circ = 2 * Math.PI * r;
  const pct = Math.round(value * 100);
  const color = value >= 0.7 ? "var(--success)" : value >= 0.4 ? "var(--warning)" : "var(--destructive)";
  return (
    <div className="relative flex h-16 w-16 items-center justify-center">
      <svg viewBox="0 0 64 64" className="h-16 w-16 -rotate-90">
        <circle cx="32" cy="32" r={r} fill="none" stroke="var(--muted)" strokeWidth="6" />
        <circle
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - value)}
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
      </svg>
      <span className="absolute font-mono text-sm font-semibold text-foreground">%{pct}</span>
    </div>
  );
}

function SanityBadge({ passed }: { passed: boolean | null }) {
  if (passed === null)
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
        <MinusCircle className="h-3.5 w-3.5" /> Atlandı
      </span>
    );
  return passed ? (
    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--success)]/14 px-2.5 py-1 text-xs font-medium text-[var(--success)]">
      <CheckCircle2 className="h-3.5 w-3.5" /> Geçti
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-full bg-[var(--destructive)]/14 px-2.5 py-1 text-xs font-medium text-[var(--destructive)]">
      <XCircle className="h-3.5 w-3.5" /> Başarısız
    </span>
  );
}

export function XaiAccordion({ ctx }: { ctx: AnalysisContext }) {
  const xai = ctx.xai;
  if (!xai) return null;

  const reportArtifact =
    ctx.xaiArtifacts.find((a) => a.mime === "image/png" || a.type === "report_png") ?? null;

  return (
    <Card className="p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Brain className="h-4 w-4 text-primary" />
          Açıklanabilir AI (XAI)
        </div>
        <SanityBadge passed={xai.sanity_passed} />
      </div>

      <div className="mt-4 flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3">
        <CoherenceGauge value={xai.coherence_score} />
        <div>
          <p className="text-xs font-medium text-foreground">Tutarlılık (Coherence)</p>
          <p className="text-[11px] text-muted-foreground">
            Görsel (Grad-CAM) ve istatistiksel (SHAP) kanıtların uyum derecesi.
          </p>
        </div>
      </div>

      <Tabs defaultValue="narrative" className="mt-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="narrative" className="gap-1 text-xs">
            <FileText className="h-3.5 w-3.5" /> Anlatı
          </TabsTrigger>
          <TabsTrigger value="gradcam" className="gap-1 text-xs">
            <Flame className="h-3.5 w-3.5" /> Grad-CAM
          </TabsTrigger>
          <TabsTrigger value="shap" className="gap-1 text-xs">
            <Brain className="h-3.5 w-3.5" /> SHAP
          </TabsTrigger>
        </TabsList>

        <TabsContent value="narrative" className="mt-3">
          <Markdown content={xai.narrative} />
        </TabsContent>

        <TabsContent value="gradcam" className="mt-3">
          <ArtifactImage
            artifact={reportArtifact}
            fallbackNote={
              reportArtifact
                ? "Heatmap görseli yüklenemedi (backend erişilemiyor olabilir)."
                : "Stilize önizleme — bu oturumda backend XAI artifact PNG'si yok (explain=true ile canlı analiz gerekir)."
            }
          />
          <p className="mt-3 text-sm leading-relaxed text-foreground">{xai.gradcam_summary}</p>
        </TabsContent>

        <TabsContent value="shap" className="mt-3">
          {reportArtifact && (
            <ArtifactImage
              artifact={reportArtifact}
              fallbackNote="SHAP görseli yüklenemedi."
            />
          )}
          <p className="mt-3 text-sm leading-relaxed text-foreground">{xai.shap_summary}</p>
          <p className="mt-2 text-[11px] text-muted-foreground">
            SHAP, CNN embedding uzayından türetilen XGBoost özellik katkılarını özetler. Görsel,
            birleşik XAI raporundan alınmıştır.
          </p>
        </TabsContent>
      </Tabs>
    </Card>
  );
}
