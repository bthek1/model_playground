import { afterEach, describe, expect, it, vi } from "vitest";

import { createAsrHandler, type AsrPipeline } from "./asrEngine";
import type { AsrResponse } from "./types";

// pickBackend (used when a `load` message omits opts) touches navigator.gpu; keep
// it absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

describe("createAsrHandler", () => {
  afterEach(() => clearGpu());

  it("loads a model and posts progress then ready", async () => {
    clearGpu();
    const posted: AsrResponse[] = [];
    const pipe = vi.fn() as unknown as AsrPipeline;
    const factory = vi.fn(async (_model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "model.onnx", progress: 42 });
      return pipe;
    });

    const handle = createAsrHandler((m) => posted.push(m), factory);
    await handle({ type: "load", model: "onnx-community/whisper-base" });

    expect(factory).toHaveBeenCalledWith(
      "onnx-community/whisper-base",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(posted).toEqual([
      { type: "progress", progress: { status: "download", file: "model.onnx", progress: 42 } },
      { type: "ready", model: "onnx-community/whisper-base", backend: "wasm" },
    ]);
  });

  it("transcribes after loading and returns the first result", async () => {
    clearGpu();
    const posted: AsrResponse[] = [];
    const pipe = vi
      .fn()
      .mockResolvedValue({ text: "hello world", chunks: [] }) as unknown as AsrPipeline;
    const handle = createAsrHandler((m) => posted.push(m), async () => pipe);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });
    const audio = new Float32Array([0.1, 0.2]);
    await handle({ type: "run", id: 7, audio });

    // Default run args (timestamps + chunking) are applied.
    expect(pipe).toHaveBeenCalledWith(
      audio,
      expect.objectContaining({ return_timestamps: true, chunk_length_s: 30 }),
    );
    expect(posted[posted.length - 1]).toEqual({
      type: "result",
      id: 7,
      result: { text: "hello world", chunks: [] },
    });
  });

  it("unwraps an array result to its first element", async () => {
    const posted: AsrResponse[] = [];
    const pipe = vi.fn().mockResolvedValue([{ text: "first" }]) as unknown as AsrPipeline;
    const handle = createAsrHandler((m) => posted.push(m), async () => pipe);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 1, audio: new Float32Array([0]) });

    expect(posted[posted.length - 1]).toEqual({ type: "result", id: 1, result: { text: "first" } });
  });

  it("errors a run when no model is loaded", async () => {
    const posted: AsrResponse[] = [];
    const handle = createAsrHandler((m) => posted.push(m), async () => vi.fn() as never);

    await handle({ type: "run", id: 3, audio: new Float32Array([0]) });

    expect(posted[posted.length - 1]).toEqual({
      type: "error",
      id: 3,
      error: "No ASR model loaded",
    });
  });

  it("reports a load failure as an id-less error", async () => {
    clearGpu();
    const posted: AsrResponse[] = [];
    const handle = createAsrHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("download failed");
      },
    );

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    expect(posted[posted.length - 1]).toEqual({ type: "error", error: "download failed" });
  });

  it("disposes the previous model before loading a new one", async () => {
    const dispose = vi.fn().mockResolvedValue(undefined);
    const first = Object.assign(vi.fn(), { dispose }) as unknown as AsrPipeline;
    const second = vi.fn() as unknown as AsrPipeline;
    const factory = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second);
    const handle = createAsrHandler(() => {}, factory);

    await handle({ type: "load", model: "a", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "load", model: "b", opts: { device: "wasm", dtype: "q8" } });

    expect(dispose).toHaveBeenCalledTimes(1);
  });
});
