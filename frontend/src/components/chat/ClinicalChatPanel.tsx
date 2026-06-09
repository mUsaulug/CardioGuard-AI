import { useEffect, useRef, useState } from "react";
import { Send, RefreshCw, Sparkles } from "lucide-react";
import type { AnalysisContext, ChatMessage as Msg, LlmStatus } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { ChatMessage } from "./ChatMessage";
import { QuickReplyChips } from "./QuickReplyChips";
import { LlmStatusBanner, LlmStatusPill } from "./LlmStatusBanner";
import logo from "@/assets/cardioguard-logo.png";

const MAX = 500;

export function ClinicalChatPanel({
  ctx,
  messages,
  isDemo,
  llmStatus,
  isResponding,
  llmProgress,
  llmAvailable,
  onSend,
  onElaborate,
  onReset,
}: {
  ctx: AnalysisContext;
  messages: Msg[];
  isDemo: boolean;
  llmStatus: LlmStatus;
  isResponding: boolean;
  llmProgress: string | null;
  llmAvailable: boolean;
  onSend: (text: string) => void;
  onElaborate: () => void;
  onReset: () => void;
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [ctx.sessionId]);

  const submit = () => {
    const t = input.trim();
    if (!t || isResponding) return;
    onSend(t);
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border p-3">
        <div className="flex items-center gap-2.5">
          <img src={logo} alt="" width={36} height={36} className="h-9 w-9" />
          <div className="leading-tight">
            <p className="text-sm font-semibold text-foreground">CardioGuard Klinik Asistan</p>
            <p className="text-[11px] text-muted-foreground">
              Bu oturumdaki EKG analizi hakkında soru sorun
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <LlmStatusPill status={isDemo ? "offline" : llmStatus} />
          <Button variant="ghost" size="icon" onClick={onReset} aria-label="Yeni analiz" title="Yeni Analiz">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
        <LlmStatusBanner status={llmStatus} isDemo={isDemo} />
        {isResponding && llmProgress && (
          <p className="animate-pulse text-xs text-muted-foreground">{llmProgress}</p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} />
        ))}
        {llmAvailable && !isResponding && !messages.some((m) => m.source === "llm") && (
          <Button
            variant="outline"
            size="sm"
            className="gap-2"
            onClick={onElaborate}
          >
            <Sparkles className="h-4 w-4 text-primary" /> LLM ile detaylandır
          </Button>
        )}
      </div>

      <div className="space-y-2 border-t border-border p-3">
        <QuickReplyChips onSelect={(t) => onSend(t)} disabled={isResponding} />
        <div className="flex items-end gap-2">
          <div className="relative flex-1">
            <textarea
              ref={inputRef}
              value={input}
              maxLength={MAX}
              rows={1}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="Bu EKG hakkında soru sorun..."
              className="max-h-32 w-full resize-none rounded-lg border border-input bg-background px-3 py-2.5 pr-14 text-sm text-foreground outline-none focus:border-primary"
              aria-label="Mesaj girişi"
            />
            <span className="pointer-events-none absolute bottom-2 right-3 font-mono text-[10px] text-muted-foreground">
              {input.length}/{MAX}
            </span>
          </div>
          <Button onClick={submit} disabled={!input.trim() || isResponding} size="icon" aria-label="Gönder">
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
