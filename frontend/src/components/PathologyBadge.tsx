import { cn } from "@/lib/utils";
import { PATHOLOGY_LABELS_TR } from "@/lib/glossary";

const STYLES: Record<string, string> = {
  MI: "bg-[var(--path-mi)]/12 text-[var(--path-mi)] border-[var(--path-mi)]/30",
  STTC: "bg-[var(--path-sttc)]/14 text-[var(--path-sttc)] border-[var(--path-sttc)]/30",
  CD: "bg-[var(--path-cd)]/12 text-[var(--path-cd)] border-[var(--path-cd)]/30",
  HYP: "bg-[var(--path-hyp)]/12 text-[var(--path-hyp)] border-[var(--path-hyp)]/30",
  NORM: "bg-[var(--path-norm)]/12 text-[var(--path-norm)] border-[var(--path-norm)]/30",
};

export function PathologyBadge({
  label,
  className,
  showFull = false,
}: {
  label: string;
  className?: string;
  showFull?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-semibold",
        STYLES[label] ?? "bg-muted text-muted-foreground border-border",
        className,
      )}
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
      {showFull && PATHOLOGY_LABELS_TR[label] ? (
        <span className="font-normal opacity-80"> · {PATHOLOGY_LABELS_TR[label]}</span>
      ) : null}
    </span>
  );
}

export function pathColorVar(label: string): string {
  const map: Record<string, string> = {
    MI: "var(--path-mi)",
    STTC: "var(--path-sttc)",
    CD: "var(--path-cd)",
    HYP: "var(--path-hyp)",
    NORM: "var(--path-norm)",
  };
  return map[label] ?? "var(--muted-foreground)";
}
