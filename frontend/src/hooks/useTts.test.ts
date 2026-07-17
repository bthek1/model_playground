import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TtsResponse } from "@/audio/tts";

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<TtsResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: TtsResponse) {
    this.onmessage?.({ data } as MessageEvent<TtsResponse>);
  }
}

let lastWorker: FakeWorker;

vi.mock("@/audio/ttsClient", () => ({
  createTtsWorker: () => {
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useTts } = await import("./useTts");

describe("useTts", () => {
  afterEach(() => vi.clearAllMocks());

  it("posts a load message with the model on mount", () => {
    const { result } = renderHook(() => useTts("onnx-community/Kokoro-82M-v1.0-ONNX"));
    expect(result.current.loading).toBe(true);
    expect(lastWorker.posted[0].message).toEqual({
      type: "load",
      model: "onnx-community/Kokoro-82M-v1.0-ONNX",
    });
  });

  it("flips to ready and records the backend", async () => {
    const { result } = renderHook(() => useTts());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "webgpu" }));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.backend).toBe("webgpu");
  });

  it("resolves synthesize() with the correlated audio and stores it", async () => {
    const { result } = renderHook(() => useTts());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.synthesize("hello", { voice: "af_heart" });
    });

    expect(lastWorker.posted[lastWorker.posted.length - 1].message).toEqual({
      type: "run",
      id: 1,
      text: "hello",
      opts: { voice: "af_heart" },
    });
    expect(result.current.running).toBe(true);

    const payload = { audio: new Float32Array([0.5]), sampleRate: 24000 };
    act(() => lastWorker.emit({ type: "result", id: 1, result: payload }));
    await expect(promise).resolves.toEqual(payload);
    await waitFor(() => expect(result.current.running).toBe(false));
    expect(result.current.result).toEqual(payload);
  });

  it("rejects synthesize() on a matching error", async () => {
    const { result } = renderHook(() => useTts());
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.synthesize("hi");
    });
    act(() => lastWorker.emit({ type: "error", id: 1, error: "boom" }));

    await expect(promise).rejects.toThrow("boom");
  });

  it("marks status error on an id-less load error", async () => {
    const { result } = renderHook(() => useTts());
    act(() => lastWorker.emit({ type: "error", error: "load failed" }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("load failed");
  });

  it("terminates the worker on unmount", () => {
    const { unmount } = renderHook(() => useTts());
    const worker = lastWorker;
    unmount();
    expect(worker.terminated).toBe(true);
  });
});
