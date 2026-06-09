const CHIPS = [
  "MI ne anlama geliyor?",
  "STTC neden tespit edildi?",
  "Lokalizasyonu açıkla",
  "Hasta dilinde özetle",
  "Risk seviyesi nedir?",
  "XAI ne gösteriyor?",
];

export function QuickReplyChips({
  onSelect,
  disabled,
}: {
  onSelect: (text: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {CHIPS.map((c) => (
        <button
          key={c}
          onClick={() => onSelect(c)}
          disabled={disabled}
          className="rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground disabled:opacity-50"
        >
          {c}
        </button>
      ))}
    </div>
  );
}
