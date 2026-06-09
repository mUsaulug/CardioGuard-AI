import { useMemo } from "react";

const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

/** Build a stylized ECG-like path. `elevated` adds an exaggerated ST segment (MI look). */
function buildPath(seed: number, elevated: boolean): string {
  const width = 220;
  const baseline = 24;
  const beats = 3;
  const beatW = width / beats;
  let d = `M 0 ${baseline}`;
  for (let b = 0; b < beats; b++) {
    const x0 = b * beatW;
    const wobble = ((seed * (b + 1)) % 5) - 2;
    // P wave
    d += ` L ${x0 + beatW * 0.12} ${baseline}`;
    d += ` q ${beatW * 0.04} -5 ${beatW * 0.08} 0`;
    d += ` L ${x0 + beatW * 0.32} ${baseline}`;
    // QRS complex
    d += ` L ${x0 + beatW * 0.36} ${baseline + 4}`;
    d += ` L ${x0 + beatW * 0.4} ${baseline - 18 + wobble}`;
    d += ` L ${x0 + beatW * 0.44} ${baseline + 7}`;
    // ST segment + T wave (elevated for MI)
    const st = elevated ? baseline - 7 : baseline;
    d += ` L ${x0 + beatW * 0.5} ${st}`;
    d += ` L ${x0 + beatW * 0.62} ${st}`;
    d += ` q ${beatW * 0.06} ${elevated ? -10 : -7} ${beatW * 0.12} 0`;
    d += ` L ${x0 + beatW} ${baseline}`;
  }
  return d;
}

export function EcgWaveformMock({ highlightLeads = [] }: { highlightLeads?: string[] }) {
  const paths = useMemo(
    () =>
      LEADS.map((lead, i) => ({
        lead,
        elevated: highlightLeads.includes(lead),
        d: buildPath(i + 1, highlightLeads.includes(lead)),
      })),
    [highlightLeads],
  );

  return (
    <div className="ecg-grid overflow-hidden rounded-lg border border-border bg-card/40 p-2">
      <div className="grid grid-cols-2 gap-px sm:grid-cols-3">
        {paths.map(({ lead, d, elevated }) => (
          <div key={lead} className="relative">
            <span
              className={`absolute left-1 top-0.5 z-10 font-mono text-[9px] font-medium ${
                elevated ? "text-[var(--path-mi)]" : "text-muted-foreground"
              }`}
            >
              {lead}
            </span>
            <svg viewBox="0 0 220 48" className="h-12 w-full" preserveAspectRatio="none" aria-hidden>
              <path
                d={d}
                fill="none"
                stroke={elevated ? "var(--path-mi)" : "var(--primary)"}
                strokeWidth={elevated ? 1.6 : 1.1}
                strokeLinejoin="round"
                strokeLinecap="round"
                style={{
                  strokeDasharray: 1400,
                  animation: `ecg-trace 2.2s ease-out forwards`,
                }}
              />
            </svg>
          </div>
        ))}
      </div>
      <p className="mt-1.5 px-1 text-[10px] text-muted-foreground">
        12 derivasyonlu EKG (stilize görselleştirme). Vurgulu derivasyonlar tespit edilen bölgeyi gösterir.
      </p>
    </div>
  );
}
