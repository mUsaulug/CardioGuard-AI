/**
 * Tarayıcı oturum olaylarını backend'e yazar — agent `GET /debug/client-log` ile okuyabilir.
 * API key asla loglanmaz.
 */
import { getBackendUrl } from "./storage";

export type DebugLevel = "info" | "warn" | "error";
export type DebugCategory = "llm" | "analysis" | "ui";

export interface DebugEvent {
  ts: string;
  level: DebugLevel;
  category: DebugCategory;
  message: string;
  meta?: Record<string, unknown>;
}

const LOCAL_KEY = "cardioguard:debug_log";
const MAX_LOCAL = 120;

function pushLocal(ev: DebugEvent): void {
  if (typeof window === "undefined") return;
  try {
    const prev = JSON.parse(localStorage.getItem(LOCAL_KEY) || "[]") as DebugEvent[];
    prev.push(ev);
    localStorage.setItem(LOCAL_KEY, JSON.stringify(prev.slice(-MAX_LOCAL)));
  } catch {
    /* ignore */
  }
}

function postToBackend(ev: DebugEvent): void {
  const base = getBackendUrl().replace(/\/+$/, "");
  fetch(`${base}/debug/client-log`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ev),
  }).catch(() => {
    /* backend kapalıysa sessiz */
  });
}

/** Oturum olayı kaydet (localStorage + backend dosyası). */
export function debugLog(
  category: DebugCategory,
  level: DebugLevel,
  message: string,
  meta?: Record<string, unknown>,
): void {
  const safeMeta = meta ? { ...meta } : undefined;
  if (safeMeta) {
    delete safeMeta.apiKey;
    delete safeMeta.key;
  }
  const ev: DebugEvent = {
    ts: new Date().toISOString(),
    level,
    category,
    message,
    meta: safeMeta,
  };
  pushLocal(ev);
  postToBackend(ev);
  if (import.meta.env.DEV) {
    const tag = level === "error" ? "error" : level === "warn" ? "warn" : "log";
    console[tag](`[CG:${category}]`, message, safeMeta ?? "");
  }
}

export function getLocalDebugLog(): DebugEvent[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(LOCAL_KEY) || "[]") as DebugEvent[];
  } catch {
    return [];
  }
}
