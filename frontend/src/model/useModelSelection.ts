// Selection, the half of a model page that has to survive a refresh.
//
// Two things happen here, and they belong together because the second makes the
// first actionable:
//
//  1. **Selection persists.** The route's `useState(DEFAULT_MODEL)` threw the
//     user's choice away on every reload. The stored id is validated against the
//     catalogue first — a model we no longer ship falls back to the default and
//     the stale entry is dropped, rather than wedging the page on a dead id.
//  2. **The cache is probed.** `cached` drives the picker's badge and the LOAD
//     slot's copy, so "Load model" stops reading equally expensive for a 3 MB
//     warm model and a 1 GB cold one.
//
// What this hook deliberately does **not** do is start a load. Selecting a model
// is a statement of intent, not a command; the only thing that moves Machine A
// out of `idle` is the LOAD button (model-page-pattern.md §1.2). That holds on a
// first visit, on a refresh, and on a switch to a model whose weights are
// already cached — a cache hit makes the click cheap, it does not make the click
// unnecessary. Earlier revisions resumed a cached load on mount from a stored
// consent flag; a page that begins working before you have asked it to is
// indistinguishable from a page you cannot control, so the flag is gone.

import { useCallback, useEffect, useMemo, useState } from "react";

import { useModelPrefs } from "@/store/models";

import { cachedModels, evictModel } from "./cache";

export interface UseModelSelectionResult<T> {
  /** The selected catalogue entry. Never one that isn't in `models`. */
  model: T;
  setModel: (model: T) => void;
  /** Model ids whose weights are already in the browser cache. */
  cached: Set<string>;
  /** Shorthand for `cached.has(model.id)`. */
  isCached: boolean;
  /** Wrap the LOAD slot's action. Kept so the call sites stay uniform. */
  onLoad: (load: () => void) => () => void;
  /** Wrap the LOAD slot's cancel. */
  onCancel: (cancel: () => void) => () => void;
  /** Call when the model reaches `ready`, to refresh the cached set. */
  refreshCache: () => void;
  /** Delete this model's cached weights. */
  evict: (modelId: string) => Promise<void>;
}

export function useModelSelection<T extends { id: string }>({
  /** Stable per route — `"asr"`, `"tts"`, `"audio-classification"`. */
  routeKey,
  models,
  fallback,
}: {
  routeKey: string;
  models: readonly T[];
  fallback: T;
}): UseModelSelectionResult<T> {
  const storedId = useModelPrefs((s) => s.selected[routeKey]);
  const selectModel = useModelPrefs((s) => s.selectModel);
  const forget = useModelPrefs((s) => s.forget);

  const stored = models.find((m) => m.id === storedId);
  const model = stored ?? fallback;

  // A stored id that is no longer in the catalogue is dropped, silently: the
  // user gets the default rather than a page that cannot load anything.
  useEffect(() => {
    if (storedId && !stored) forget(routeKey);
  }, [storedId, stored, forget, routeKey]);

  const [cached, setCached] = useState<Set<string>>(() => new Set());
  const [probe, setProbe] = useState(0);

  useEffect(() => {
    let live = true;
    void cachedModels().then((set) => {
      if (live) setCached(set);
    });
    return () => {
      live = false;
    };
  }, [probe]);

  const setModel = useCallback(
    (next: T) => {
      if (next.id === model.id) return;
      selectModel(routeKey, next.id);
    },
    [model.id, routeKey, selectModel],
  );

  const onLoad = useCallback((load: () => void) => load, []);
  const onCancel = useCallback((cancel: () => void) => cancel, []);

  const refreshCache = useCallback(() => setProbe((n) => n + 1), []);

  const evict = useCallback(async (modelId: string) => {
    await evictModel(modelId);
    setProbe((n) => n + 1);
  }, []);

  return useMemo(
    () => ({
      model,
      setModel,
      cached,
      isCached: cached.has(model.id),
      onLoad,
      onCancel,
      refreshCache,
      evict,
    }),
    [model, setModel, cached, onLoad, onCancel, refreshCache, evict],
  );
}

/**
 * Re-probe the cache when a load finishes, so the "Cached" badge appears the
 * moment the download does — without waiting for the next page load.
 */
export function useCacheRefresh(
  session: Pick<UseModelSelectionResult<{ id: string }>, "refreshCache">,
  ready: boolean,
) {
  const { refreshCache } = session;
  useEffect(() => {
    if (ready) refreshCache();
  }, [ready, refreshCache]);
}
