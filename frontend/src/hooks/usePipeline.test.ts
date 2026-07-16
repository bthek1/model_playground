import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PipelineResponse } from "@/audio/pipelineTypes";

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<PipelineResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: PipelineResponse) {
    this.onmessage?.({ data } as MessageEvent<PipelineResponse>);
  }
}

let lastWorker: FakeWorker;

vi.mock("@/audio/pipelineClient", () => ({
  createPipelineWorker: () => {
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { usePipeline } = await import("./usePipeline");

describe("usePipeline", () => {
  afterEach(() => vi.clearAllMocks());

  it("posts a load message with the task + model on mount", () => {
    const { result } = renderHook(() =>
      usePipeline("audio-classification", "onnx-community/ast"),
    );
    expect(result.current.loading).toBe(true);
    expect(lastWorker.posted[0].message).toEqual({
      type: "load",
      task: "audio-classification",
      model: "onnx-community/ast",
    });
  });

  it("flips to ready and records the backend", async () => {
    const { result } = renderHook(() => usePipeline("audio-classification", "m"));
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "webgpu" }));
    await waitFor(() => expect(result.current.ready).toBe(true));
    expect(result.current.backend).toBe("webgpu");
  });

  it("resolves run() with the correlated result and transfers the input", async () => {
    const { result } = renderHook(() => usePipeline("audio-classification", "m"));
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    const input = new Float32Array([0.1, 0.2, 0.3]);
    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.run(input, [{ top_k: 6 }]);
    });

    const run = lastWorker.posted[lastWorker.posted.length - 1];
    expect(run.message).toMatchObject({ type: "run", id: 1, args: [{ top_k: 6 }] });
    expect(run.transfer).toEqual([input.buffer]);
    expect(result.current.running).toBe(true);

    const payload = [{ label: "Speech", score: 0.8 }];
    act(() => lastWorker.emit({ type: "result", id: 1, result: payload }));
    await expect(promise).resolves.toEqual(payload);
    await waitFor(() => expect(result.current.running).toBe(false));
  });

  it("rejects run() on a matching error", async () => {
    const { result } = renderHook(() => usePipeline("audio-classification", "m"));
    act(() => lastWorker.emit({ type: "ready", model: "m", backend: "wasm" }));

    let promise!: Promise<unknown>;
    act(() => {
      promise = result.current.run(new Float32Array([0]));
    });
    act(() => lastWorker.emit({ type: "error", id: 1, error: "boom" }));

    await expect(promise).rejects.toThrow("boom");
  });

  it("marks status error on an id-less load error", async () => {
    const { result } = renderHook(() => usePipeline("audio-classification", "m"));
    act(() => lastWorker.emit({ type: "error", error: "load failed" }));
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.error).toBe("load failed");
  });

  it("terminates the worker on unmount", () => {
    const { unmount } = renderHook(() => usePipeline("audio-classification", "m"));
    const worker = lastWorker;
    unmount();
    expect(worker.terminated).toBe(true);
  });
});
