import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { AsrResponse } from "@/audio/types";

// A stand-in for the real Web Worker: records posted messages and lets a test
// push responses back via `emit`.
class FakeWorker {
  onmessage: ((e: MessageEvent<AsrResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: AsrResponse) {
    this.onmessage?.({ data } as MessageEvent<AsrResponse>);
  }
}

let lastWorker: FakeWorker;

// Counted, not just captured: proving that `autoLoad: false` spawns *nothing*
// needs to distinguish "no worker" from "a worker left over from a prior test".
let spawned = 0;

vi.mock("@/audio/asrClient", () => ({
  createAsrWorker: () => {
    spawned += 1;
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useAsr } = await import("./useAsr");

describe("useAsr", () => {
  beforeEach(() => {
    spawned = 0;
  });
  afterEach(() => vi.clearAllMocks());

  // The deferred-load contract (model-page-pattern.md §2). Every task route
  // passes `false`, so this passthrough is what stops a page from spending the
  // user's bandwidth just because they navigated to it.
  it("spawns no worker at all when autoLoad is false", () => {
    const { result } = renderHook(() => useAsr("m", false));
    expect(result.current.status).toBe("idle");
    expect(result.current.idle).toBe(true);
    expect(result.current.loading).toBe(false);
    expect(spawned).toBe(0);
  });

  it("load() starts the ASR worker once, and again is a no-op", () => {
    const { result } = renderHook(() => useAsr("m", false));

    act(() => result.current.load());
    expect(spawned).toBe(1);
    expect(result.current.status).toBe("loading");

    // Double-clicking Load must not open a second worker on the same model.
    act(() => result.current.load());
    expect(spawned).toBe(1);
  });

  it("retry() reloads after a failed load", async () => {
    const { result } = renderHook(() => useAsr("m", false));
    act(() => result.current.load());
    act(() => lastWorker.emit({ type: "error", error: "boom" }));
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.retry());
    expect(spawned).toBe(2);
    expect(result.current.status).toBe("loading");
  });

  it("posts a load message on mount and starts loading", () => {
    const { result } = renderHook(() => useAsr("onnx-community/whisper-base"));
    expect(result.current.loading).toBe(true);
    expect(result.current.ready).toBe(false);
    expect(lastWorker.posted[0].message).toEqual({
      type: "load",
      model: "onnx-community/whisper-base",
    });
  });

  it("flips to ready and records the backend on the ready message", async () => {
    const { result } = renderHook(() => useAsr());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "webgpu" }));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.backend).toBe("webgpu");
  });

  it("resolves transcribe() with the correlated result and transfers audio", async () => {
    const { result } = renderHook(() => useAsr());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    const audio = new Float32Array([0.1, 0.2, 0.3]);
    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.transcribe(audio);
    });

    // The run message carries an id and transfers the audio buffer.
    const run = lastWorker.posted[lastWorker.posted.length - 1];
    expect(run.message).toMatchObject({ type: "run", id: 1 });
    expect(run.transfer).toEqual([audio.buffer]);
    expect(result.current.running).toBe(true);

    act(() => lastWorker.emit({ type: "result", id: 1, result: { text: "hi" } }));
    await expect(promise).resolves.toEqual({ text: "hi" });
    await waitFor(() => expect(result.current.running).toBe(false));
    expect(result.current.result).toEqual({ text: "hi" });
  });

  it("rejects transcribe() on a matching run error", async () => {
    const { result } = renderHook(() => useAsr());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.transcribe(new Float32Array([0]));
    });
    act(() => lastWorker.emit({ type: "error", id: 1, error: "boom" }));

    await expect(promise).rejects.toThrow("boom");
  });

  it("marks status error on an id-less (load) error", async () => {
    const { result } = renderHook(() => useAsr());
    act(() => lastWorker.emit({ type: "error", error: "load failed" }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("load failed");
  });

  it("terminates the worker on unmount", () => {
    const { unmount } = renderHook(() => useAsr());
    const worker = lastWorker;
    unmount();
    expect(worker.terminated).toBe(true);
  });
});
