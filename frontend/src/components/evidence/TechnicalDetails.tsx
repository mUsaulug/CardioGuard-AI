import { useState } from "react";
import { ChevronDown, Copy, Check, Code2 } from "lucide-react";
import { toast } from "sonner";
import type { AnalysisContext } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function TechnicalDetails({ ctx }: { ctx: AnalysisContext }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(ctx, null, 2);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      toast.success("JSON kopyalandı");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Kopyalanamadı");
    }
  };

  return (
    <Card className="overflow-hidden p-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between p-4 text-sm font-medium text-foreground"
      >
        <span className="flex items-center gap-2">
          <Code2 className="h-4 w-4 text-primary" /> Teknik Detaylar
        </span>
        <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="border-t border-border p-4">
          <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground">
            <span className="font-mono">
              {ctx.versions
                ? `model_hash: ${ctx.versions.model_hash} · API ${ctx.versions.api_version} · thr#${ctx.versions.threshold_hash}`
                : `session: ${ctx.sessionId.slice(0, 12)}`}
            </span>
            <button onClick={copy} className="flex items-center gap-1 rounded px-2 py-1 hover:bg-muted">
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              Kopyala
            </button>
          </div>
          <pre className="max-h-72 overflow-auto rounded-lg bg-muted/50 p-3 font-mono text-[11px] leading-relaxed text-foreground">
            {json}
          </pre>
        </div>
      )}
    </Card>
  );
}
