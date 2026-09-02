import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { EnhanceResponse } from "@/audio/enhance/types";

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<EnhanceResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: EnhanceResponse) {
    this.onmessage?.({ data } as MessageEvent<EnhanceResponse>);
  }
}

let lastWorker: FakeWorker;

vi.mock("@/audio/enhance/enhanceClient", () => ({
  createEnhanceWorker: () => {
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useEnhance } = await import("./useEnhance");

const ready = (model = "soniqo/DeepFilterNet3-ONNX") =>
  lastWorker.emit({ type: "ready", model, backend: "wasm" });

describe("useEnhance", () => {
  afterEach(() => vi.clearAllMocks());

  it("downloads nothing until asked — the weights are the user's bandwidth", () => {
    const { result } = renderHook(() => useEnhance());
    expect(result.current.idle).toBe(true);
    expect(result.current.loading).toBe(false);
    expect(lastWorker).toBeUndefined();
  });

  it("posts a load message with the model when load() is called", () => {
    const { result } = renderHook(() => useEnhance());
    act(() => result.current.load());
    expect(result.current.loading).toBe(true);
    expect(lastWorker.posted[0].message).toEqual({
      type: "load",
      model: "soniqo/DeepFilterNet3-ONNX",
    });
  });

  it("flips to ready and records the backend", async () => {
    const { result } = renderHook(() => useEnhance());
    act(() => result.current.load());
    act(() => ready());
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.backend).toBe("wasm");
  });

  it("resolves run() with the correlated result and transfers the input", async () => {
    const { result } = renderHook(() => useEnhance());
    act(() => result.current.load());
    act(() => ready());
    await waitFor(() => expect(result.current.ready).toBe(true));

    const audio = new Float32Array([0.1, 0.2, 0.3]);
    let resolved: unknown;
    act(() => {
      void result.current.run(audio).then((r) => (resolved = r));
    });

    const run = lastWorker.posted[lastWorker.posted.length - 1]!;
    expect(run.message).toMatchObject({ type: "run", id: 1 });
    expect(run.transfer).toEqual([audio.buffer]);
    await waitFor(() => expect(result.current.running).toBe(true));

    const out = { audio: new Float32Array([0.5]), sampleRate: 48000 };
    act(() => lastWorker.emit({ type: "result", id: 1, result: out }));
    await waitFor(() => expect(resolved).toEqual(out));
    expect(result.current.result).toEqual(out);
    expect(result.current.running).toBe(false);
  });

  it("keeps the model loaded when a single run fails", async () => {
    const { result } = renderHook(() => useEnhance());
    act(() => result.current.load());
    act(() => ready());
    await waitFor(() => expect(result.current.ready).toBe(true));

    let rejection: Error | null = null;
    act(() => {
      void result.current.run(new Float32Array(4)).catch((e) => (rejection = e));
    });
    act(() => lastWorker.emit({ type: "error", id: 1, error: "session run failed" }));

    await waitFor(() => expect(rejection).toBeInstanceOf(Error));
    expect(result.current.ready).toBe(true);
    expect(result.current.error).toBe("session run failed");
  });

  it("moves to error on a load failure and retries from there", async () => {
    const { result } = renderHook(() => useEnhance());
    act(() => result.current.load());
    act(() => lastWorker.emit({ type: "error", error: "404 on deepfilter.onnx" }));
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.retry());
    await waitFor(() => expect(result.current.loading).toBe(true));
    expect(lastWorker.posted[0].message).toMatchObject({ type: "load" });
  });
});
