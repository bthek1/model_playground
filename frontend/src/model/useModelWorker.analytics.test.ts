// The load/run analytics events (#60) are emitted once, here, rather than per
// route — so these cases are the whole of their coverage. They pin the shape:
// the catalogue id, the backend and the timing go out; the message never does.

import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const tracked = vi.hoisted(() => [] as Array<[string, Record<string, unknown>]>);
vi.mock("@/analytics", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/analytics")>()),
  track: (event: string, props: Record<string, unknown>) => tracked.push([event, props]),
}));

import type { ModelResponse } from "./types";
import { useModelWorker } from "./useModelWorker";

class FakeWorker {
  onmessage: ((e: MessageEvent<ModelResponse<string>>) => void) | null = null;
  postMessage() {}
  terminate() {}
  emit(data: ModelResponse<string>) {
    this.onmessage?.({ data } as MessageEvent<ModelResponse<string>>);
  }
}

function setup() {
  const workers: FakeWorker[] = [];
  const view = renderHook(() =>
    useModelWorker<string>({
      createWorker: () => {
        const w = new FakeWorker();
        workers.push(w);
        return w as unknown as Worker;
      },
      key: "k",
      loadMessage: { task: "text-classification", model: "Xenova/distilbert-sst-2" },
    }),
  );
  return { view, worker: () => workers[workers.length - 1] };
}

const MODEL = { modelId: "Xenova/distilbert-sst-2", task: "text-classification" };
const names = () => tracked.map(([e]) => e);

beforeEach(() => {
  tracked.length = 0;
});

describe("useModelWorker analytics", () => {
  it("emits nothing on mount — no load, no event", () => {
    setup();
    expect(tracked).toEqual([]);
  });

  it("a successful load and run: started → ready → run completed, with id, backend and timing", async () => {
    const { view, worker } = setup();
    act(() => view.result.current.load());
    act(() => worker().emit({ type: "ready", model: "m", backend: "webgpu" }));
    let pending!: Promise<string>;
    act(() => {
      pending = view.result.current.run({ text: "my private sentence" });
    });
    act(() => worker().emit({ type: "result", id: 1, result: "POSITIVE" }));
    await pending;

    expect(names()).toEqual(["model_load_started", "model_load_ready", "model_run_completed"]);
    expect(tracked[0][1]).toMatchObject({ ...MODEL, retry: false });
    expect(tracked[1][1]).toMatchObject({ ...MODEL, backend: "webgpu" });
    expect(tracked[1][1].loadedInMs).toEqual(expect.any(Number));
    expect(tracked[2][1]).toMatchObject(MODEL);
    expect(tracked[2][1].runMs).toEqual(expect.any(Number));
    // Neither the input nor the output went anywhere near an event.
    expect(JSON.stringify(tracked)).not.toMatch(/private sentence|POSITIVE/);
  });

  it("a failed load sends the category, never the message", () => {
    const { view, worker } = setup();
    act(() => view.result.current.load());
    act(() =>
      worker().emit({ type: "error", error: "Failed to fetch /home/me/secret-notes.txt" }),
    );
    expect(tracked[1]).toEqual(["model_load_failed", { ...MODEL, errorKind: "network" }]);
    expect(JSON.stringify(tracked)).not.toMatch(/secret-notes/);
  });

  it("a failed run sends the category, never the message", async () => {
    const { view, worker } = setup();
    act(() => view.result.current.load());
    act(() => worker().emit({ type: "ready", model: "m", backend: "wasm" }));
    let pending!: Promise<string>;
    act(() => {
      pending = view.result.current.run({});
    });
    act(() => worker().emit({ type: "error", id: 1, error: 'bad input "my words"' }));
    await expect(pending).rejects.toThrow();
    expect(tracked[tracked.length - 1]).toEqual(["model_run_failed", { ...MODEL, errorKind: "unknown" }]);
  });

  it("a retry is marked, with the backend it pinned", () => {
    const { view, worker } = setup();
    act(() => view.result.current.load());
    act(() => worker().emit({ type: "error", error: "WebGPU device was lost" }));
    act(() => view.result.current.retry({ backend: "wasm" }));
    expect(tracked[tracked.length - 1]).toEqual([
      "model_load_started",
      { ...MODEL, retry: true, backend: "wasm" },
    ]);
  });

  it("a cancel is its own event", () => {
    const { view } = setup();
    act(() => view.result.current.load());
    act(() => view.result.current.cancel());
    expect(tracked[tracked.length - 1]?.[0]).toBe("model_load_cancelled");
    expect(tracked[tracked.length - 1]?.[1]).toMatchObject(MODEL);
  });
});
