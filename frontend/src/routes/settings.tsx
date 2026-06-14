import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Key, Sparkles, Moon, Server, Save, PlugZap, Bot, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { AppShell } from "@/components/AppShell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useTheme } from "@/hooks/useTheme";
import { testBackendConnection } from "@/lib/api/cardioguard";
import { testOpenRouterConnection, FREE_MODEL_CHAIN } from "@/lib/openrouter";
import {
  getApiKey,
  setApiKey,
  getBackendUrl,
  setBackendUrl,
  getDemoMode,
  setDemoMode,
} from "@/lib/storage";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Ayarlar — CardioGuard-AI" },
      { name: "description", content: "API anahtarı, demo modu, tema ve backend ayarları." },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const { theme, toggleTheme } = useTheme();
  const [key, setKey] = useState(getApiKey());
  const [demo, setDemo] = useState(getDemoMode());
  const [backend, setBackend] = useState(getBackendUrl());
  const [testing, setTesting] = useState(false);
  const [testingLlm, setTestingLlm] = useState(false);

  const dirty =
    key.trim() !== getApiKey() ||
    demo !== getDemoMode() ||
    backend.trim() !== getBackendUrl();

  const save = () => {
    setApiKey(key.trim());
    setDemoMode(demo);
    setBackendUrl(backend.trim());
    toast.success("Ayarlar kaydedildi");
  };

  const testConnection = async () => {
    setTesting(true);
    const ok = await testBackendConnection(backend.trim() || getBackendUrl());
    setTesting(false);
    if (ok) toast.success("Backend bağlantısı başarılı");
    else toast.error("Backend'e ulaşılamadı");
  };

  const testLlm = async () => {
    setTestingLlm(true);
    // Test with typed key (may differ from saved until Kaydet).
    const result = await testOpenRouterConnection(backend.trim() || getBackendUrl(), key.trim() || getApiKey());
    setTestingLlm(false);
    if (result.ok) {
      toast.success(`OpenRouter OK — model: ${result.model}`);
    } else {
      toast.error(result.detail || "OpenRouter bağlantısı başarısız");
    }
  };

  return (
    <AppShell>
      <div className="mx-auto w-full max-w-2xl px-4 py-10 sm:px-6">
        <h1 className="font-display text-2xl font-bold text-foreground">Ayarlar</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Prod ortamda OpenRouter anahtarı sunucuda (<code className="text-xs">OPENROUTER_API_KEY</code>)
          tutulur; tarayıcı doğrudan OpenRouter&apos;a bağlanmaz. Yerel geliştirmede isteğe bağlı
          tarayıcı anahtarı kullanılabilir.
        </p>

        <div className="mt-6 space-y-4">
          <Card className="p-5">
            <label className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Key className="h-4 w-4 text-primary" /> OpenRouter API Anahtarı
            </label>
            <Input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="sk-or-..."
              className="mt-2 font-mono"
            />
            <p className="mt-2 text-xs text-muted-foreground">
              Yerel geliştirme: anahtarı buraya yazın (backend proxy üzerinden iletilir). Prod: sunucu
              ortam değişkeni yeterli. Anahtarı yazdıktan sonra{" "}
              <span className="font-medium text-foreground">Kaydet</span> butonuna basın.
            </p>
          </Card>

          <Card className="p-5">
            <label className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Bot className="h-4 w-4 text-primary" /> LLM — yalnızca ücretsiz modeller
            </label>
            <p className="mt-2 text-xs text-muted-foreground">
              Tek OpenRouter API anahtarınız tüm ücretsiz modellere erişir. Sistem otomatik olarak
              sırayla dener; biri limitteyse veya yanıt vermezse sonrakine geçer — boş cevap
              bırakmaz.
            </p>
            <ul className="mt-3 space-y-1 rounded-lg border border-border bg-muted/30 p-3 font-mono text-[11px] text-foreground">
              {FREE_MODEL_CHAIN.map((m, i) => (
                <li key={m}>
                  {i + 1}. {m}
                </li>
              ))}
            </ul>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-3 gap-2"
              disabled={testingLlm}
              onClick={testLlm}
            >
              <PlugZap className="h-4 w-4" />
              {testingLlm ? "LLM test ediliyor..." : "Ücretsiz LLM Bağlantısını Test Et"}
            </Button>
          </Card>

          <Card className="flex items-center justify-between p-5">
            <div>
              <p className="flex items-center gap-2 text-sm font-medium text-foreground">
                <Sparkles className="h-4 w-4 text-primary" /> Demo Modu
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                EKG analizi canlı kalır; yalnızca sohbet yanıtları şablon/kural tabanlı olur.
              </p>
            </div>
            <Switch checked={demo} onCheckedChange={setDemo} />
          </Card>

          <Card className="flex items-center justify-between p-5">
            <div>
              <p className="flex items-center gap-2 text-sm font-medium text-foreground">
                <Moon className="h-4 w-4 text-primary" /> Koyu Tema
              </p>
              <p className="mt-1 text-xs text-muted-foreground">Açık / koyu görünüm arasında geçiş yap.</p>
            </div>
            <Switch checked={theme === "dark"} onCheckedChange={toggleTheme} />
          </Card>

          <Card className="p-5">
            <label className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Server className="h-4 w-4 text-primary" /> Backend URL
            </label>
            <Input
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
              placeholder="http://localhost:8000"
              className="mt-2 font-mono"
            />
            <p className="mt-2 text-xs text-muted-foreground">
              Canlı EKG analizi için CardioGuard-AI FastAPI backend adresi.
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-3 gap-2"
              disabled={testing}
              onClick={testConnection}
            >
              <PlugZap className="h-4 w-4" />
              {testing ? "Test ediliyor..." : "Bağlantıyı Test Et"}
            </Button>
          </Card>

          {dirty && (
            <div className="flex items-center gap-2 rounded-lg bg-[var(--warning)]/12 px-3 py-2 text-xs text-[var(--warning)]">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              Kaydedilmemiş değişiklikler var. Etkili olması için Kaydet'e basın.
            </div>
          )}
          <Button onClick={save} className="w-full gap-2">
            <Save className="h-4 w-4" /> Kaydet
          </Button>
        </div>
      </div>
    </AppShell>
  );
}
