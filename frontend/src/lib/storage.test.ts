import { beforeEach, describe, it, expect, vi } from "vitest";
import type { StoredSession } from "./types";

const { store } = vi.hoisted(() => {
  const store: Record<string, string> = {};
  const mock = {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => {
      store[k] = v;
    },
    removeItem: (k: string) => {
      delete store[k];
    },
  };
  vi.stubGlobal("localStorage", mock);
  vi.stubGlobal("window", { localStorage: mock });
  return { store };
});

import {
  getBackendUrl,
  setBackendUrl,
  getDemoMode,
  setDemoMode,
  loadSession,
  saveSession,
  clearSession,
} from "./storage";

function makeSession(): StoredSession {
  return {
    schema: 2,
    timestamp: Date.now(),
    isDemo: false,
    context: {
      sessionId: "s1",
      fileName: "ecg.npz",
      timestamp: "2026-06-14T00:00:00Z",
      primary: { label: "MI", confidence: 0.4, rule: "r" },
      predictedLabels: ["MI"],
      probabilities: { MI: 0.4, STTC: 0, CD: 0, HYP: 0, NORM: 0.6 },
      thresholds: { MI: 0.16 },
      sources: { cnn: {}, xgb: {}, ensemble: {} },
      consistency: null,
      localization: null,
      xai: null,
      xaiArtifacts: [],
      runId: null,
      latencyMs: null,
      glossary: {},
    },
    messages: [],
  };
}

describe("storage", () => {
  beforeEach(() => {
    Object.keys(store).forEach((k) => delete store[k]);
    clearSession();
  });

  it("persists and loads backend URL", () => {
    setBackendUrl("http://127.0.0.1:8000");
    expect(getBackendUrl()).toBe("http://127.0.0.1:8000");
  });

  it("toggles demo mode", () => {
    expect(getDemoMode()).toBe(false);
    setDemoMode(true);
    expect(getDemoMode()).toBe(true);
  });

  it("discards session with stale schema", () => {
    store["cardioguard:session"] = JSON.stringify({ ...makeSession(), schema: 1 });
    expect(loadSession()).toBeNull();
  });

  it("loads valid session", () => {
    saveSession(makeSession());
    const loaded = loadSession();
    expect(loaded?.context.sessionId).toBe("s1");
  });
});
