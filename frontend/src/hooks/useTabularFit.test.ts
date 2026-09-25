import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { parseCsv } from "@/tabular/csv";
import { familyInfo } from "@/tabular/families";
import type { FitResult, TabularResponse } from "@/tabular/types";

import { useTabularFit } from "./useTabularFit";

class FakeWorker {
  onmessage: ((e: MessageEvent<TabularResponse>) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;
  postMessage(message: unknown) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: TabularResponse) {
    this.onmessage?.({ data } as MessageEvent<TabularResponse>);
  }
}

let lastWorker: FakeWorker;
let spawned = 0;
const spawn = () => {
  spawned += 1;
  lastWorker = new FakeWorker();
  return lastWorker as unknown as Worker;
};

const { dataset } = parseCsv("x,y\n1,a\n2,b\n3,a\n", { name: "tiny" });

const SPEC = {
  family: "forest" as const,
  objective: "classification" as const,
  targetIndex: 1,
  featureIndices: [0],
  hp: familyInfo("forest").defaults,
  seed: 1,
  testFraction: 0.25,
};

function fire<T>(p: Promise<T>): Promise<T> {
  p.catch(() => {});
  return p;
}

describe("useTabularFit", () => {
  beforeEach(() => {
    spawned = 0;
    vi.clearAllMocks();
  });

  it("spawns no worker and fits nothing on mount", () => {
    // `autoLoad` is never passed — nothing on this page may spend before the
    // user presses FIT, and a fit is the thing that costs seconds.
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    expect(spawned).toBe(0);
    expect(hook.result.current.idle).toBe(true);
    expect(hook.result.current.result).toBeNull();
  });

  it("fit() loads and runs in one action", () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    expect(spawned).toBe(1);
    expect(lastWorker.posted[0]).toMatchObject({ type: "load", dataset });
    expect(lastWorker.posted[1]).toMatchObject({ type: "run", id: 1, spec: SPEC });
  });

  it("does not re-load a worker that is already holding the data", async () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    act(() => lastWorker.emit({ type: "ready", model: "tiny", backend: "wasm" }));
    await waitFor(() => expect(hook.result.current.ready).toBe(true));
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    expect(spawned).toBe(1);
    expect(lastWorker.posted.filter((p) => (p as { type: string }).type === "load")).toHaveLength(1);
  });

  it("surfaces the fit's determinate counter on partial", async () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    act(() =>
      lastWorker.emit({
        type: "partial",
        id: 1,
        partial: { done: 7, total: 60, phase: "Fitting", loss: 0.5 },
      }),
    );
    await waitFor(() =>
      expect(hook.result.current.partial).toEqual({
        done: 7,
        total: 60,
        phase: "Fitting",
        loss: 0.5,
      }),
    );
    // Machine A is untouched by a partial: `ready` is about the weights (here,
    // the data), and a run in progress does not move it.
    expect(hook.result.current.status).toBe("loading");
    expect(hook.result.current.running).toBe(true);
  });

  it("keeps the fit result when a prediction lands afterwards", async () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    const fit = { spec: SPEC, trainRows: 2, testRows: 1, fitMs: 3, compute: "cpu", importance: [], curve: [] } as unknown as FitResult;
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    act(() => lastWorker.emit({ type: "result", id: 1, result: fit }));
    await waitFor(() => expect(hook.result.current.result).toBe(fit));

    act(() => {
      fire(hook.result.current.predict([1]));
    });
    act(() =>
      lastWorker.emit({
        type: "result",
        id: 2,
        result: { scores: [0.3, 0.7], labels: ["a", "b"], objective: "classification" },
      }),
    );
    // A prediction is a different question about the same model; it must not
    // replace the metric block on screen.
    await waitFor(() => expect(hook.result.current.result).toBe(fit));
  });

  it("sends stop out of band, without opening a request", () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    const before = lastWorker.posted.length;
    act(() => hook.result.current.stop());
    expect(lastWorker.posted[before]).toEqual({ type: "stop" });
    // Still one request in flight: stop sets a flag the loop polls, it does not
    // settle the run.
    expect(hook.result.current.running).toBe(true);
  });

  it("tears the worker down when the dataset changes", async () => {
    const hook = renderHook(({ d }) => useTabularFit(d, spawn), {
      initialProps: { d: dataset },
    });
    act(() => {
      fire(hook.result.current.fit(SPEC));
    });
    const first = lastWorker;
    const other = parseCsv("x,y\n9,a\n8,b\n", { name: "other" }).dataset;
    hook.rerender({ d: other });
    await waitFor(() => expect(first.terminated).toBe(true));
    expect(hook.result.current.idle).toBe(true);
  });

  it("rejects a run with no worker rather than silently doing nothing", async () => {
    const hook = renderHook(() => useTabularFit(dataset, spawn));
    await expect(hook.result.current.predict([1])).rejects.toThrow(/not ready/i);
  });
});
