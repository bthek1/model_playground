// Selection + resume, the half of a model page that has to survive a refresh.
//
// Three things happen here, and they belong together because the third depends
// on the first two:
//
//  1. **Selection persists.** The route's `useState(DEFAULT_MODEL)` threw the
//     user's choice away on every reload. The stored id is validated against the
//     catalogue first — a model we no longer ship falls back to the default and
//     the stale entry is dropped, rather than wedging the page on a dead id.
//  2. **The cache is probed.** `cached` drives the picker's badge and the LOAD
//     slot's copy, so "Load model" stops reading equally expensive for a 3 MB
//     warm model and a 1 GB cold one.
//  3. **A load resumes — only on a cache hit.** §1.2 of the page pattern says
//     nothing downloads without an action. A cached model costs no bandwidth, so
//     resuming it spends nothing the user hasn't already spent; an uncached one
//     stays `idle` and asks, with the fact that they were using it spelled out.
//
// The decision is resolved here rather than inside `useModelWorker` so the hook
// stays dumb: it is handed a plain `autoLoad` boolean, exactly as before.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  /**
   * Pass straight to the task hook. Stays `false` until the cache probe has
   * answered, so an undecided page never downloads on mount.
   */
  autoLoad: boolean;
  /** True while the load in flight was started by the resume path, not a click. */
  restoring: boolean;
  /** Wrap the LOAD slot's action: records the consent for next time. */
  onLoad: (load: () => void) => () => void;
  /** Wrap cancel: a cancelled load must not resume on the next refresh. */
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
  const wantsResume = useModelPrefs((s) => s.autoResume[routeKey] ?? false);
  const selectModel = useModelPrefs((s) => s.selectModel);
  const setAutoResume = useModelPrefs((s) => s.setAutoResume);
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
  // null = the probe hasn't answered yet, so we have not decided whether to
  // resume. `false` and `null` both mean "don't load", which is why the flip
  // from null to false is invisible to `useModelWorker` and costs no teardown.
  const [decision, setDecision] = useState<boolean | null>(null);
  const restoring = useRef(false);

  useEffect(() => {
    let live = true;
    void cachedModels().then((set) => {
      if (!live) return;
      setCached(set);
      setDecision((prev) => {
        // A decision already taken (the user clicked, or a previous probe
        // resolved) is never revisited — re-deciding mid-load would tear the
        // worker down under them.
        if (prev != null) return prev;
        const resume = wantsResume && set.has(model.id);
        restoring.current = resume;
        return resume;
      });
    });
    return () => {
      live = false;
    };
  }, [model.id, wantsResume, probe]);

  // Switching models re-opens the question for the new one.
  const setModel = useCallback(
    (next: T) => {
      if (next.id === model.id) return;
      restoring.current = false;
      setDecision(null);
      selectModel(routeKey, next.id);
    },
    [model.id, routeKey, selectModel],
  );

  const onLoad = useCallback(
    (load: () => void) => () => {
      restoring.current = false;
      setDecision(false);
      setAutoResume(routeKey, true);
      load();
    },
    [routeKey, setAutoResume],
  );

  const onCancel = useCallback(
    (cancel: () => void) => () => {
      restoring.current = false;
      setAutoResume(routeKey, false);
      cancel();
    },
    [routeKey, setAutoResume],
  );

  const refreshCache = useCallback(() => setProbe((n) => n + 1), []);

  const evict = useCallback(
    async (modelId: string) => {
      await evictModel(modelId);
      if (modelId === model.id) setAutoResume(routeKey, false);
      setProbe((n) => n + 1);
    },
    [model.id, routeKey, setAutoResume],
  );

  return useMemo(
    () => ({
      model,
      setModel,
      cached,
      isCached: cached.has(model.id),
      autoLoad: decision === true,
      restoring: restoring.current,
      onLoad,
      onCancel,
      refreshCache,
      evict,
    }),
    [model, setModel, cached, decision, onLoad, onCancel, refreshCache, evict],
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
