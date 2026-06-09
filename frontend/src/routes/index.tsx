import { useState, useCallback, useRef } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  UploadCloud,
  Sparkles,
  Activity,
  ShieldCheck,
  Loader2,
  CheckCircle2,
  FileText,
  ClipboardList,
  MessageSquare,
  RefreshCw,
} from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { EvidencePanel } from "@/components/evidence/EvidencePanel";
import { ClinicalChatPanel } from "@/components/chat/ClinicalChatPanel";
import { useAnalysisSession, ANALYSIS_STEPS } from "@/hooks/useAnalysisSession";
import type { AnalyzeOptions } from "@/lib/types";
import { getApiKey, getDemoMode } from "@/lib/storage";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CardioGuard-AI — EKG Analizi & Klinik Asistan" },
      {
        name: "description",
        content:
          "12 derivasyonlu EKG yükleyin; CNN + XGBoost ensemble ile patoloji tespiti, MI lokalizasyonu, açıklanabilir AI ve Türkçe klinik asistan.",
      },
    ],
  }),
  component: HomePage,
});

function HomePage() {
  const session = useAnalysisSession();

  return (
    <AppShell>
      {session.appState === "welcome" && (
        <WelcomeView onAnalyze={session.runAnalysis} onDemo={session.loadDemo} />
      )}
      {session.appState === "analyzing" && <AnalyzingView stepIndex={session.stepIndex} />}
      {session.appState === "results" && session.context && (
        <ResultsView session={session} />
      )}
    </AppShell>
  );
}

function WelcomeView({
  onAnalyze,
  onDemo,
}: {
  onAnalyze: (file: File, fileName: string, o: AnalyzeOptions, demo: boolean) => void;
  onDemo: () => void;
}) {
  const [explain, setExplain] = useState(true);
  const [sanity, setSanity] = useState(true);
  const [ensemble, setEnsemble] = useState(0.85);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const validate = (f: File): boolean => {
    if (!/\.(npy|npz)$/i.test(f.name)) {
      toast.error("Yalnızca .npy veya .npz dosyaları kabul edilir");
      return false;
    }
    if (f.size > 10 * 1024 * 1024) {
      toast.error("Dosya boyutu 10MB'ı aşamaz");
      return false;
    }
    return true;
  };

  const handleFile = (f: File) => {
    if (validate(f)) {
      setFileName(f.name);
      setSelectedFile(f);
    }
  };

  const analyze = () => {
    if (!selectedFile || !fileName) {
      toast.error("Lütfen bir EKG dosyası seçin");
      return;
    }
    onAnalyze(
      selectedFile,
      fileName,
      { explain, sanityCheck: sanity, ensemble },
      getDemoMode(),
    );
  };

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-10 sm:px-6 sm:py-16">
      <div className="text-center">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
          <Activity className="h-3.5 w-3.5 text-primary" /> Açıklanabilir EKG Analiz Platformu
        </span>
        <h1 className="mt-4 font-display text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          CardioGuard-AI
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-base text-muted-foreground">
          12 derivasyonlu EKG → patoloji tespiti (MI, STTC, CD, HYP) → MI lokalizasyonu → açıklanabilir
          AI → anlaşılır Türkçe klinik özet.
        </p>
      </div>

      <div className="mt-10 grid gap-6 lg:grid-cols-[1fr_280px]">
        <Card className="p-6">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
            }}
            onClick={() => fileRef.current?.click()}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors",
              dragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/40",
            )}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".npy,.npz"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary">
              {fileName ? <FileText className="h-7 w-7" /> : <UploadCloud className="h-7 w-7" />}
            </div>
            {fileName ? (
              <p className="mt-3 font-mono text-sm text-foreground">{fileName}</p>
            ) : (
              <>
                <p className="mt-3 text-sm font-medium text-foreground">
                  EKG dosyanızı sürükleyin veya seçmek için tıklayın
                </p>
                <p className="mt-1 text-xs text-muted-foreground">.npy veya .npz — maks. 10MB</p>
              </>
            )}
          </div>

          <div className="mt-6 space-y-4">
            <label className="flex cursor-pointer items-center gap-2.5 text-sm text-foreground">
              <Checkbox checked={explain} onCheckedChange={(v) => setExplain(!!v)} />
              XAI Açıklama <span className="font-mono text-xs text-muted-foreground">(explain=true)</span>
            </label>
            <label className="flex cursor-pointer items-center gap-2.5 text-sm text-foreground">
              <Checkbox checked={sanity} onCheckedChange={(v) => setSanity(!!v)} />
              Kalite Kontrolü <span className="font-mono text-xs text-muted-foreground">(sanity check)</span>
            </label>
            <div>
              <div className="mb-2 flex items-center justify-between text-sm">
                <span className="text-foreground">Model ağırlığı</span>
                <span className="font-mono text-xs text-muted-foreground">
                  XGB %{Math.round(ensemble * 100)} · CNN %{Math.round((1 - ensemble) * 100)}
                </span>
              </div>
              <Slider value={[ensemble]} min={0} max={1} step={0.05} onValueChange={(v) => setEnsemble(v[0])} />
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                Sağa kaydır = daha güçlü XGBoost modeline ağırlık. Önerilen: XGB %85.
              </p>
            </div>
          </div>

          <div className="mt-6 flex flex-col gap-2 sm:flex-row">
            <Button onClick={analyze} className="flex-1 gap-2">
              <Activity className="h-4 w-4" /> EKG Analiz Et
            </Button>
            <Button onClick={onDemo} variant="outline" className="flex-1 gap-2">
              <Sparkles className="h-4 w-4" /> Demo Analizi Gör
            </Button>
          </div>
        </Card>

        <Card className="h-fit p-5">
          <p className="text-sm font-medium text-foreground">Sistem Durumu</p>
          <div className="mt-3 space-y-2.5 text-sm">
            <StatusRow ok label="Demo verisi hazır" />
            <StatusRow ok={!!getApiKey()} label={getApiKey() ? "LLM API bağlı" : "LLM API yok (offline)"} />
            <StatusRow ok label="Açıklama motoru aktif" />
          </div>
          <div className="mt-4 flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-xs text-muted-foreground">
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            Bu sistem tanı koymaz; klinik karar destek aracıdır.
          </div>
        </Card>
      </div>
    </div>
  );
}

function StatusRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={cn("h-2 w-2 rounded-full", ok ? "bg-[var(--success)]" : "bg-muted-foreground")} />
      <span className={ok ? "text-foreground" : "text-muted-foreground"}>{label}</span>
    </div>
  );
}

function AnalyzingView({ stepIndex }: { stepIndex: number }) {
  return (
    <div className="mx-auto w-full max-w-2xl px-4 py-16 sm:px-6">
      <div className="text-center">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
          <Loader2 className="h-7 w-7 animate-spin text-primary" />
        </div>
        <h2 className="mt-4 font-display text-xl font-semibold text-foreground">EKG analiz ediliyor</h2>
        <p className="mt-1 text-sm text-muted-foreground">Pipeline çalışıyor, lütfen bekleyin...</p>
      </div>

      <div className="mt-8 space-y-2">
        {ANALYSIS_STEPS.map((step, i) => {
          const done = i < stepIndex;
          const active = i === stepIndex;
          return (
            <motion.div
              key={step}
              initial={{ opacity: 0.4 }}
              animate={{ opacity: done || active ? 1 : 0.4 }}
              className={cn(
                "flex items-center gap-3 rounded-lg border p-3 text-sm",
                active ? "border-primary/40 bg-primary/5" : "border-border",
              )}
            >
              {done ? (
                <CheckCircle2 className="h-5 w-5 text-[var(--success)]" />
              ) : active ? (
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              ) : (
                <span className="h-5 w-5 rounded-full border-2 border-muted" />
              )}
              <span className={done || active ? "text-foreground" : "text-muted-foreground"}>{step}</span>
            </motion.div>
          );
        })}
      </div>

      <div className="mt-8 space-y-3">
        {[0, 1].map((i) => (
          <div key={i} className="h-24 animate-pulse rounded-xl bg-muted/60" />
        ))}
      </div>
    </div>
  );
}

function ResultsView({ session }: { session: ReturnType<typeof useAnalysisSession> }) {
  const ctx = session.context!;
  const [tab, setTab] = useState<"evidence" | "chat">("chat");

  return (
    <div className="mx-auto w-full max-w-[1500px] px-4 py-5 sm:px-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/30 px-3 py-2.5">
        <p className="text-xs text-muted-foreground">
          Başka bir EKG denemek için yeni analiz başlatın (oturum ve sohbet sıfırlanır).
        </p>
        <Button variant="outline" size="sm" className="gap-2 shrink-0" onClick={session.reset}>
          <RefreshCw className="h-4 w-4" />
          Yeni Analiz
        </Button>
      </div>

      {session.restored && (
        <div className="mb-4 flex items-center justify-between rounded-lg border border-border bg-accent/40 px-3 py-2 text-xs text-foreground">
          <span>↻ Önceki oturum geri yüklendi.</span>
          <button onClick={session.dismissRestored} className="text-muted-foreground hover:text-foreground">
            Kapat
          </button>
        </div>
      )}

      {/* Mobile tabs */}
      <div className="mb-4 flex gap-1 rounded-lg bg-muted p-1 lg:hidden">
        <button
          onClick={() => setTab("evidence")}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-md py-2 text-sm font-medium",
            tab === "evidence" ? "bg-card text-foreground shadow-sm" : "text-muted-foreground",
          )}
        >
          <ClipboardList className="h-4 w-4" /> Sonuçlar
        </button>
        <button
          onClick={() => setTab("chat")}
          className={cn(
            "flex flex-1 items-center justify-center gap-1.5 rounded-md py-2 text-sm font-medium",
            tab === "chat" ? "bg-card text-foreground shadow-sm" : "text-muted-foreground",
          )}
        >
          <MessageSquare className="h-4 w-4" /> Asistan
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,45fr)_minmax(0,55fr)]">
        <div className={cn("lg:block", tab === "evidence" ? "block" : "hidden")}>
          <div className="lg:sticky lg:top-20">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-foreground">
              <ClipboardList className="h-4 w-4 text-primary" /> Kanıt Paneli
            </h2>
            <EvidencePanel ctx={ctx} isDemo={session.isDemo} />
          </div>
        </div>

        <div className={cn("lg:block", tab === "chat" ? "block" : "hidden")}>
          <div className="h-[calc(100vh-9rem)] lg:sticky lg:top-20">
            <ClinicalChatPanel
              ctx={ctx}
              messages={session.messages}
              isDemo={session.isDemo}
              llmStatus={session.llmStatus}
              isResponding={session.isResponding}
              llmProgress={session.llmProgress}
              llmAvailable={session.llmAvailable}
              onSend={session.sendMessage}
              onElaborate={session.elaborateWithLlm}
              onReset={session.reset}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
