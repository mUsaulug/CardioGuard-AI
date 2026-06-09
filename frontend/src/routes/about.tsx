import { createFileRoute } from "@tanstack/react-router";
import { ShieldCheck, Brain, HeartPulse, Layers } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/card";
import { DISCLAIMER } from "@/lib/glossary";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "Hakkında — CardioGuard-AI" },
      {
        name: "description",
        content:
          "CardioGuard-AI tez projesi: CNN + XGBoost ensemble ile açıklanabilir 12 derivasyonlu EKG patoloji analizi ve klinik karar destek.",
      },
    ],
  }),
  component: AboutPage,
});

const FEATURES = [
  { icon: HeartPulse, title: "Çoklu-etiket Tespit", text: "MI, STTC, CD, HYP ve türetilmiş NORM sınıfları." },
  { icon: Layers, title: "Ensemble Model", text: "CNN + XGBoost birleşimi ve Consistency Guard güvenlik katmanı." },
  { icon: Brain, title: "Açıklanabilir AI", text: "Grad-CAM, SHAP ve birleşik anlatı ile şeffaf kararlar." },
  { icon: ShieldCheck, title: "Karar Destek", text: "Tanı koymaz; hekime anlaşılır Türkçe özet sunar." },
];

function AboutPage() {
  return (
    <AppShell>
      <div className="mx-auto w-full max-w-3xl px-4 py-10 sm:px-6">
        <h1 className="font-display text-2xl font-bold text-foreground">CardioGuard-AI Hakkında</h1>
        <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
          CardioGuard-AI, 12 derivasyonlu EKG sinyallerini analiz ederek kardiyak patolojileri tespit eden,
          sonuçları açıklanabilir AI yöntemleriyle gerekçelendiren ve anlaşılır Türkçe klinik dile çeviren bir
          lisans tezi projesidir. Sistem, makine öğrenmesi çıktısını bir LLM klinik anlatıcısı ve oturuma özel
          sohbet asistanı ile birleştirir.
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <Card key={f.title} className="p-5">
              <f.icon className="h-5 w-5 text-primary" />
              <p className="mt-2 text-sm font-semibold text-foreground">{f.title}</p>
              <p className="mt-1 text-xs text-muted-foreground">{f.text}</p>
            </Card>
          ))}
        </div>

        <Card className="mt-6 p-5">
          <h2 className="text-sm font-semibold text-foreground">Metodoloji</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            <li>Sinyal ön işleme ve özellik çıkarımı (CNN embedding).</li>
            <li>Çoklu-etiket sınıflandırma ve sınıfa özel eşik kalibrasyonu.</li>
            <li>MI lokalizasyonu (AMI, ASMI, ALMI, IMI, LMI).</li>
            <li>Consistency Guard: iki bağımsız MI modelinin uyum kontrolü.</li>
            <li>XAI: Grad-CAM + SHAP + tutarlılık (coherence) skoru.</li>
          </ul>
        </Card>

        <div className="mt-6 flex items-start gap-2 rounded-lg bg-[var(--warning)]/12 p-4 text-sm text-[var(--warning)]">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0" />
          {DISCLAIMER} Nihai değerlendirme her zaman hekim tarafından yapılmalıdır.
        </div>
      </div>
    </AppShell>
  );
}
