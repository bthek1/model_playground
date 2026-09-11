// Per-route model preferences, persisted across page loads.
//
// A refresh cannot keep a Worker or an ONNX session alive — nothing persists an
// in-memory model. What it can keep is the one decision that is cheap to
// restore and costs nothing to be wrong about: **which model was selected**.
//
// It deliberately does *not* persist "and load it again". Loading a model is
// the LOAD button's job and nothing else's (model-page-pattern.md §1.2), so a
// revisited page puts the user's model back in the picker and then waits. The
// cache probe in `model/cache.ts` still runs — it makes the one click an
// informed one ("Cached · no download") rather than replacing it.
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
  selectModel: (routeKey: string, modelId: string) => void;
  /** Drop a stored id that is no longer in the catalogue. */
  forget: (routeKey: string) => void;
}

export const useModelPrefs = create<ModelPrefsState>()(
  persist(
    immer((set) => ({
      selected: {},
      selectModel: (routeKey, modelId) =>
        set((s) => {
          s.selected[routeKey] = modelId;
        }),
      forget: (routeKey) =>
        set((s) => {
          delete s.selected[routeKey];
        }),
    })),
    {
      name: "model-prefs",
      storage: createJSONStorage(() => safeStorage),
      // Only `selected` is persisted, and saying so explicitly is what stops an
      // old blob from outliving the feature that wrote it: earlier versions
      // stored an `autoResume` flag, and without `partialize` zustand merges
      // that dead key back into state on rehydration and writes it out again
      // on every save. Nothing reads it — but a persisted flag that says "load
      // this on sight" is not the thing to leave lying around.
      partialize: (s) => ({ selected: s.selected }),
    },
  ),
);
