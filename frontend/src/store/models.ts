// Per-route model preferences, persisted across page loads.
//
// A refresh cannot keep a Worker or an ONNX session alive — nothing persists an
// in-memory model. What it can keep is the *decisions*: which model the user
// picked, and whether they had asked for it to be loaded. Paired with the cache
// probe in `model/cache.ts`, that is enough to put the page back where it was
// without re-asking for consent the user already gave and without spending a
// byte (see `model/useModelSelection.ts` for the resume rule).
//
// UI state only, per CLAUDE.md — no server data lives in Zustand.

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

/**
 * `localStorage`, resolved per call and never fatal. Two reasons not to hand
 * zustand the global directly: it is unavailable (or throws on access) in
 * private-browsing modes, and it is *replaced* after module load in the test
 * environment — a reference captured at import time would be the wrong object
 * for the rest of the run.
 */
const safeStorage = {
  getItem: (key: string) => {
    try {
      return globalThis.localStorage?.getItem(key) ?? null;
    } catch {
      return null;
    }
  },
  setItem: (key: string, value: string) => {
    try {
      globalThis.localStorage?.setItem(key, value);
    } catch {
      /* preferences are a convenience, not a requirement */
    }
  },
  removeItem: (key: string) => {
    try {
      globalThis.localStorage?.removeItem(key);
    } catch {
      /* ditto */
    }
  },
};

interface ModelPrefsState {
  /** Route key → last selected model id, e.g. `asr` → `onnx-community/whisper-base`. */
  selected: Record<string, string>;
  /** Route key → the user asked for this model to be loaded. */
  autoResume: Record<string, boolean>;
  selectModel: (routeKey: string, modelId: string) => void;
  setAutoResume: (routeKey: string, on: boolean) => void;
  /** Drop a stored id that is no longer in the catalogue. */
  forget: (routeKey: string) => void;
}

export const useModelPrefs = create<ModelPrefsState>()(
  persist(
    immer((set) => ({
      selected: {},
      autoResume: {},
      selectModel: (routeKey, modelId) =>
        set((s) => {
          s.selected[routeKey] = modelId;
          // A different model is a different download: the old consent does not
          // carry over to it.
          s.autoResume[routeKey] = false;
        }),
      setAutoResume: (routeKey, on) =>
        set((s) => {
          s.autoResume[routeKey] = on;
        }),
      forget: (routeKey) =>
        set((s) => {
          delete s.selected[routeKey];
          delete s.autoResume[routeKey];
        }),
    })),
    { name: "model-prefs", storage: createJSONStorage(() => safeStorage) },
  ),
);
