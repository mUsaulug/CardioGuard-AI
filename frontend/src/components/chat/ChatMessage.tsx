import { motion } from "framer-motion";
import { Stethoscope, Sparkles, FileText, ScrollText } from "lucide-react";
import type { ChatMessage as ChatMessageType, MessageSource } from "@/lib/types";
import { Markdown } from "@/components/Markdown";
import { cn } from "@/lib/utils";

const SOURCE_META: Record<MessageSource, { label: string; cls: string; icon: typeof Sparkles }> = {
  llm: {
    label: "LLM",
    cls: "bg-[var(--success)]/14 text-[var(--success)]",
    icon: Sparkles,
  },
  auto: {
    label: "Otomatik özet",
    cls: "bg-primary/12 text-primary",
    icon: FileText,
  },
  template: {
    label: "Kural tabanlı",
    cls: "bg-muted text-muted-foreground",
    icon: ScrollText,
  },
  rule: {
    label: "Kural tabanlı",
    cls: "bg-amber-500/12 text-amber-700 dark:text-amber-400",
    icon: ScrollText,
  },
};

function SourceBadge({ source }: { source: MessageSource }) {
  const meta = SOURCE_META[source];
  const Icon = meta.icon;
  return (
    <span
      className={cn(
        "mb-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium",
        meta.cls,
      )}
    >
      <Icon className="h-2.5 w-2.5" /> {meta.label}
    </span>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-2 w-2 rounded-full bg-muted-foreground"
          animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.18 }}
        />
      ))}
    </div>
  );
}

export function ChatMessage({ message }: { message: ChatMessageType }) {
  const isUser = message.role === "user";
  const empty = message.pending && !message.content;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("flex gap-2.5", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {!isUser && (
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/12 text-primary">
          <Stethoscope className="h-4 w-4" />
        </div>
      )}
      <div className={cn("flex max-w-[85%] flex-col", isUser ? "items-end" : "items-start")}>
        {!isUser && !empty && message.source && <SourceBadge source={message.source} />}
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm",
            isUser
              ? "rounded-br-sm bg-primary text-primary-foreground"
              : "rounded-bl-sm border border-border bg-accent/40 text-foreground",
          )}
        >
          {empty ? (
            <TypingDots />
          ) : isUser ? (
            <span className="whitespace-pre-wrap">{message.content}</span>
          ) : (
            <Markdown content={message.content} />
          )}
        </div>
      </div>
    </motion.div>
  );
}
