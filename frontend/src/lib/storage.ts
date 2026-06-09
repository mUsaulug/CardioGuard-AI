import type { StoredSession } from "./types";

const SESSION_KEY = "cardioguard:session";
const APIKEY_KEY = "cardioguard:openrouter_key";
const BACKEND_KEY = "cardioguard:backend_url";
const DEMO_KEY = "cardioguard:demo_mode";
const THEME_KEY = "cardioguard:theme";
const TTL_MS = 24 * 60 * 60 * 1000;
/** Bump when AnalysisContext shape changes so stale sessions are discarded. */
const SESSION_SCHEMA = 2;

const DEFAULT_BACKEND =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) || "http://localhost:8000";

const isBrowser = typeof window !== "undefined";

export function loadSession(): StoredSession | null {
  if (!isBrowser) return null;
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredSession;
    // Discard sessions from an older context schema (avoids stale/broken data
    // and missing fields like xaiArtifacts crashing the UI).
    if (parsed.schema !== SESSION_SCHEMA) {
      localStorage.removeItem(SESSION_KEY);
      return null;
    }
    if (!parsed.timestamp || Date.now() - parsed.timestamp > TTL_MS) {
      localStorage.removeItem(SESSION_KEY);
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function saveSession(session: StoredSession): void {
  if (!isBrowser) return;
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify({ ...session, schema: SESSION_SCHEMA }));
  } catch {
    /* ignore quota */
  }
}

export function clearSession(): void {
  if (!isBrowser) return;
  localStorage.removeItem(SESSION_KEY);
}

export function getApiKey(): string {
  if (!isBrowser) return "";
  const stored = localStorage.getItem(APIKEY_KEY);
  if (stored) return stored;
  const envKey = import.meta.env.VITE_OPENROUTER_API_KEY as string | undefined;
  return envKey || "";
}

export function setApiKey(key: string): void {
  if (!isBrowser) return;
  if (key) localStorage.setItem(APIKEY_KEY, key);
  else localStorage.removeItem(APIKEY_KEY);
}

export function getDemoMode(): boolean {
  if (!isBrowser) return false;
  return localStorage.getItem(DEMO_KEY) === "true";
}

export function setDemoMode(v: boolean): void {
  if (!isBrowser) return;
  localStorage.setItem(DEMO_KEY, String(v));
}

export function getStoredTheme(): "light" | "dark" | null {
  if (!isBrowser) return null;
  const t = localStorage.getItem(THEME_KEY);
  return t === "light" || t === "dark" ? t : null;
}

export function setStoredTheme(t: "light" | "dark"): void {
  if (!isBrowser) return;
  localStorage.setItem(THEME_KEY, t);
}

export function getBackendUrl(): string {
  if (!isBrowser) return DEFAULT_BACKEND;
  return localStorage.getItem(BACKEND_KEY) || DEFAULT_BACKEND;
}

export function setBackendUrl(url: string): void {
  if (!isBrowser) return;
  const trimmed = url.trim();
  if (trimmed) localStorage.setItem(BACKEND_KEY, trimmed);
  else localStorage.removeItem(BACKEND_KEY);
}
