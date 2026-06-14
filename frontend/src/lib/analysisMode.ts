/**
 * Demo Modu (Settings) only affects LLM chat — not EKG inference.
 * Mock analysis runs only for explicit simulation (demo button) or missing file.
 */
export function shouldUseMockAnalysis(
  file: File | null,
  simulationRequest: boolean,
): boolean {
  return simulationRequest || file === null;
}
