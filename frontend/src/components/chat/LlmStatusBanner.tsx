import { Wifi, WifiOff, AlertTriangle } from "lucide-react";
import type { LlmStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export function LlmStatusPill({ status }: { status: LlmStatus }) {
  const map = {
    active: { label: "Ücretsiz LLM", cls: "bg-[var(--success)]/14 text-[var(--success)]", icon: Wifi },
    offline: { label: "Offline Mod", cls: "bg-muted text-muted-foreground", icon: WifiOff },
    limit: { label: "Limit Aşıldı", cls: "bg-[var(--warning)]/16 text-[var(--warning)]", icon: AlertTriangle },
    error: { label: "LLM Hatası", cls: "bg-[var(--danger,#ef4444)]/16 text-[var(--danger,#ef4444)]", icon: AlertTriangle },
  }[status];
  const Icon = map.icon;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-medium", map.cls)}>
      <Icon className="h-3 w-3" /> {map.label}
    </span>
  );
}

export function LlmStatusBanner({ status, isDemo }: { status: LlmStatus; isDemo: boolean }) {
  if (status === "error") {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-[var(--danger,#ef4444)]/12 px-3 py-2 text-xs text-[var(--danger,#ef4444)]">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        LLM yanıtı alınamadı (anahtar/model hatası). Kural tabanlı yanıt gösteriliyor — Ayarlar'ı kontrol edin.
      </div>
    );
  }
  if (status === "limit") {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-[var(--warning)]/12 px-3 py-2 text-xs text-[var(--warning)]">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        OpenRouter ücretsiz kotası/rate limiti doldu (tüm modeller denendi). Ayarlar → model:
        openrouter/free veya birkaç dakika bekleyin.
      </div>
    );
  }
  if (isDemo) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
        <WifiOff className="h-4 w-4 shrink-0" />
        Simülasyon — gerçek API bağlantısı yok. Yanıtlar kural tabanlı üretilir.
      </div>
    );
  }
  if (status === "offline") {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
        <WifiOff className="h-4 w-4 shrink-0" />
        API anahtarı yok — kural tabanlı asistan modu aktif. Ayarlar'dan anahtar girebilirsiniz.
      </div>
    );
  }
  return null;
}
