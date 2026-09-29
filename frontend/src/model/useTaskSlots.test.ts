import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseModelSelectionResult } from "./useModelSelection";
import { useTaskSlots, type TaskSlotSource } from "./useTaskSlots";

interface Entry {
  id: string;
  label: string;
  hint: string;
  params: number;
}

const MODELS: Entry[] = [
  { id: "a", label: "A", hint: "", params: 1 },
  { id: "b", label: "B", hint: "", params: 2 },
];

const refreshCache = vi.fn();
const evict = vi.fn().mockResolvedValue(undefined);
const setModel = vi.fn();

function session(): UseModelSelectionResult<Entry> {
  return {
    model: MODELS[1],
    setModel,
    cached: new Set(["b"]),
    isCached: true,
    refreshCache,
    evict,
  };
}

function task(extra: Partial<TaskSlotSource> = {}): TaskSlotSource {
  return {
    status: "idle",
    loading: false,
    ready: false,
    backend: null,
    loadProgress: null,
    loadedInMs: null,
    error: null,
    load: vi.fn(),
    retry: vi.fn(),
    cancel: vi.fn(),
    ...extra,
  };
}

const slotsFor = (t: TaskSlotSource, opts: { busy?: boolean } = {}) =>
  renderHook(() => useTaskSlots(session(), t, { models: MODELS, ...opts }))
    .result.current;

describe("useTaskSlots — errors go to the slot that produced them", () => {
  it("routes a load failure to LOAD only", () => {
    const s = slotsFor(task({ status: "error", error: "404" }));
    expect(s.loadError).toBe("404");
    expect(s.runError).toBeNull();
    expect(s.status.error).toBe("404");
  });

  it("routes a run failure to OUTPUT only, leaving LOAD clean", () => {
    const s = slotsFor(task({ status: "ready", ready: true, error: "bad input" }));
    expect(s.runError).toBe("bad input");
    expect(s.loadError).toBeNull();
    expect(s.status.error).toBeNull();
  });
});

describe("useTaskSlots — LOAD", () => {
  it("hands ModelStatus the hook's own actions, unwrapped", () => {
    const t = task({ backend: "webgpu", loadedInMs: 1200 });
    const s = slotsFor(t);
    expect(s.status).toMatchObject({
      status: "idle",
      backend: "webgpu",
      loadedInMs: 1200,
      cached: true,
      disabled: false,
    });
    expect(s.status.onLoad).toBe(t.load);
    expect(s.status.onCancel).toBe(t.cancel);
    expect(s.status.onRetry).toBe(t.retry);
  });

  it("shuts LOAD while something else is busy", () => {
    expect(slotsFor(task(), { busy: true }).status.disabled).toBe(true);
  });
});

describe("useTaskSlots — SELECT", () => {
  it("binds the picker to the selection and the cache", () => {
    const s = slotsFor(task());
    expect(s.picker.models).toBe(MODELS);
    expect(s.picker.value).toBe("b");
    expect(s.picker.onChange).toBe(setModel);
    expect(s.picker.cached.has("b")).toBe(true);
    s.picker.onEvict(MODELS[0]);
    expect(evict).toHaveBeenCalledWith("a");
  });

  it("locks the picker during a download and while busy, not otherwise", () => {
    expect(slotsFor(task()).picker.disabled).toBe(false);
    expect(slotsFor(task({ status: "loading", loading: true })).picker.disabled).toBe(true);
    expect(slotsFor(task(), { busy: true }).picker.disabled).toBe(true);
  });

  it("passes a backend probe only when the route asked for one", () => {
    expect("backend" in slotsFor(task()).picker).toBe(false);
    const probed = renderHook(() =>
      useTaskSlots(session(), task(), { models: MODELS, backend: null }),
    ).result.current;
    expect("backend" in probed.picker).toBe(true);
    expect(probed.picker.backend).toBeNull();
  });
});

describe("useTaskSlots — the cache", () => {
  beforeEach(() => refreshCache.mockClear());

  it("re-probes the cache once the model is ready, and not before", () => {
    const { rerender } = renderHook(
      ({ t }) => useTaskSlots(session(), t, { models: MODELS }),
      { initialProps: { t: task() } },
    );
    expect(refreshCache).not.toHaveBeenCalled();
    rerender({ t: task({ status: "ready", ready: true }) });
    expect(refreshCache).toHaveBeenCalledOnce();
  });
});
