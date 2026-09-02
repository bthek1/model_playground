import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ModelResponse } from "./types";
import { useModelWorker } from "./useModelWorker";

// Stand-in for a real Web Worker: records posted messages, lets a test push
// responses back, and reports whether it was terminated.
class FakeWorker {
  onmessage: ((e: MessageEvent<ModelResponse<string>>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: ModelResponse<string>) {
    this.onmessage?.({ data } as MessageEvent<ModelResponse<string>>);
  }
}

/** Spawns FakeWorkers and keeps every one for inspection. */
function workerFactory() {
  const spawned: FakeWorker[] = [];
  const createWorker = () => {
    const w = new FakeWorker();
    spawned.push(w);
    return w as unknown as Worker;
  };
  return {
    createWorker,
    spawned,
    get last() {
      return spawned[spawned.length - 1];
    },
  };
}

function setup(options: { autoLoad?: boolean; key?: string } = {}) {
  const factory = workerFactory();
  const view = renderHook(
    ({ key }: { key: string }) =>
      useModelWorker<string>({
        createWorker: factory.createWorker,
        key,
        loadMessage: { model: key },
        autoLoad: options.autoLoad,
        notReadyMessage: "not ready",
      }),
    { initialProps: { key: options.key ?? "model-a" } },
  );
  return { ...view, factory };
}

describe("useModelWorker — Machine A (model lifecycle)", () => {
  it("auto-loads by default: loading on mount, then ready", async () => {
    const { result, factory } = setup();

    expect(result.current.status).toBe("loading");
    expect(factory.last.posted[0].message).toEqual({
      type: "load",
      model: "model-a",
    });

    act(() => factory.last.emit({ type: "ready", model: "m", backend: "wasm" }));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.backend).toBe("wasm");
  });

  it("starts idle and spawns no worker until load() when autoLoad is false", () => {
    const { result, factory } = setup({ autoLoad: false });

    expect(result.current.status).toBe("idle");
    expect(result.current.idle).toBe(true);
    // The whole point of `idle`: nothing is downloaded on navigation.
    expect(factory.spawned).toHaveLength(0);

    act(() => result.current.load());
    expect(factory.spawned).toHaveLength(1);
    expect(result.current.status).toBe("loading");
  });

  it("load() is a no-op unless idle", () => {
    const { result, factory } = setup();

    expect(factory.spawned).toHaveLength(1);
    act(() => result.current.load()); // already loading
    expect(factory.spawned).toHaveLength(1);
  });

  it("progress events never move status off loading", async () => {
    const { result, factory } = setup();

    act(() =>
      factory.last.emit({
        type: "progress",
        progress: { status: "download", file: "model.onnx", progress: 42 },
      }),
    );

    await waitFor(() => expect(result.current.progress?.progress).toBe(42));
    expect(result.current.status).toBe("loading");
  });

  it("an id-less error is a load failure and reaches the error state", async () => {
    const { result, factory } = setup();

    act(() => factory.last.emit({ type: "error", error: "download failed" }));

    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("download failed");
  });

  it("retry() restarts a failed load without a model change", async () => {
    const { result, factory } = setup();
    act(() => factory.last.emit({ type: "error", error: "boom" }));
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.retry());

    // A fresh worker, the old one torn down, and back to loading.
    expect(factory.spawned).toHaveLength(2);
    expect(factory.spawned[0].terminated).toBe(true);
    expect(result.current.status).toBe("loading");
    expect(factory.last.posted[0].message).toEqual({
      type: "load",
      model: "model-a",
    });

    act(() => factory.last.emit({ type: "ready", model: "m", backend: "wasm" }));
    await waitFor(() => expect(result.current.ready).toBe(true));
  });

  it("retry() is a no-op unless in error", () => {
    const { result, factory } = setup();
    act(() => result.current.retry());
    expect(factory.spawned).toHaveLength(1);
  });

  it("a model change tears the worker down and reloads", async () => {
    const { result, rerender, factory } = setup();
    act(() => factory.last.emit({ type: "ready", model: "m", backend: "wasm" }));
    await waitFor(() => expect(result.current.ready).toBe(true));

    rerender({ key: "model-b" });

    expect(factory.spawned[0].terminated).toBe(true);
    expect(factory.spawned).toHaveLength(2);
    expect(factory.last.posted[0].message).toEqual({
      type: "load",
      model: "model-b",
    });
    expect(result.current.status).toBe("loading");
    expect(result.current.backend).toBeNull();
  });
});

describe("useModelWorker — Machine B (inference)", () => {
  async function ready() {
    const view = setup();
    act(() =>
      view.factory.last.emit({ type: "ready", model: "m", backend: "wasm" }),
    );
    await waitFor(() => expect(view.result.current.ready).toBe(true));
    return view;
  }

  it("correlates a result by id and records it", async () => {
    const { result, factory } = await ready();

    let promise!: Promise<string>;
    act(() => {
      promise = result.current.run({ input: 1 });
    });
    const run = factory.last.posted[1];
    expect(run.message).toEqual({ type: "run", id: 1, input: 1 });

    act(() => factory.last.emit({ type: "result", id: 1, result: "out" }));
    await expect(promise).resolves.toBe("out");
    await waitFor(() => expect(result.current.result).toBe("out"));
  });

  it("keeps running true until BOTH overlapping requests resolve", async () => {
    // Regression test: `running` was a boolean, so the first response cleared it
    // while a second request was still in flight and the UI reported idle.
    const { result, factory } = await ready();

    let first!: Promise<string>;
    let second!: Promise<string>;
    act(() => {
      first = result.current.run({ input: 1 });
      second = result.current.run({ input: 2 });
    });
    await waitFor(() => expect(result.current.running).toBe(true));

    act(() => factory.last.emit({ type: "result", id: 1, result: "a" }));
    await expect(first).resolves.toBe("a");
    // Still one in flight — the flag must not have flipped.
    expect(result.current.running).toBe(true);

    act(() => factory.last.emit({ type: "result", id: 2, result: "b" }));
    await expect(second).resolves.toBe("b");
    await waitFor(() => expect(result.current.running).toBe(false));
  });

  it("an error carrying an id rejects that request and stays ready", async () => {
    const { result, factory } = await ready();

    let promise!: Promise<string>;
    act(() => {
      promise = result.current.run({ input: 1 });
    });
    act(() => factory.last.emit({ type: "error", id: 1, error: "bad input" }));

    await expect(promise).rejects.toThrow("bad input");
    // An inference failure must not take the model down.
    expect(result.current.status).toBe("ready");
    expect(result.current.error).toBe("bad input");
    await waitFor(() => expect(result.current.running).toBe(false));
  });

  it("forwards a transfer list only when one is given", async () => {
    const { result, factory } = await ready();
    const buffer = new Float32Array([1, 2]).buffer;

    act(() => void result.current.run({ audio: 1 }, [buffer]));
    expect(factory.last.posted[1].transfer).toEqual([buffer]);

    act(() => void result.current.run({ text: "hi" }));
    expect(factory.last.posted[2].transfer).toBeUndefined();
  });

  it("rejects run() when no worker is live", async () => {
    const { result } = setup({ autoLoad: false });
    await expect(result.current.run({ input: 1 })).rejects.toThrow("not ready");
  });

  it("rejects every pending request on teardown", async () => {
    const { result, unmount } = await ready();

    let promise!: Promise<string>;
    act(() => {
      promise = result.current.run({ input: 1 });
    });
    unmount();

    await expect(promise).rejects.toThrow("Worker terminated");
  });
});
