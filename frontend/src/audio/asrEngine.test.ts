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

    const handle = createAsrHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle({ type: "load", model: "onnx-community/whisper-base" });

    // On WASM the decoder stays fp32 — the quantized one can't open a session
    // in the bundled ONNX Runtime (see `asrLoadOpts`).
    expect(factory).toHaveBeenCalledWith(
      "onnx-community/whisper-base",
      expect.objectContaining({
        device: "wasm",
        dtype: { encoder_model: "q8", decoder_model_merged: "fp32" },
      }),
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

  it("warms the model up with silence before reporting ready", async () => {
    clearGpu();
    const posted: AsrResponse[] = [];
    const pipe = vi.fn().mockResolvedValue({ text: "" }) as unknown as AsrPipeline;
    const handle = createAsrHandler((m) => posted.push(m), async () => pipe);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    // One throwaway inference on silence, before `ready`.
    expect(pipe).toHaveBeenCalledTimes(1);
    const [audio] = (pipe as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(audio).toBeInstanceOf(Float32Array);
    expect(audio.every((v: number) => v === 0)).toBe(true);
    expect(posted).toEqual([
      { type: "progress", progress: { status: "warmup" } },
      { type: "ready", model: "m", backend: "wasm" },
    ]);
  });

  it("still reports ready when the warm-up inference throws", async () => {
    clearGpu();
    const posted: AsrResponse[] = [];
    const pipe = vi
      .fn()
      .mockRejectedValue(new Error("shader compile failed")) as unknown as AsrPipeline;
    const handle = createAsrHandler((m) => posted.push(m), async () => pipe);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    expect(posted[posted.length - 1]).toEqual({
      type: "ready",
      model: "m",
      backend: "wasm",
    });
  });

  it("disposes the previous model before loading a new one", async () => {
    const dispose = vi.fn().mockResolvedValue(undefined);
    const first = Object.assign(vi.fn(), { dispose }) as unknown as AsrPipeline;
    const second = vi.fn() as unknown as AsrPipeline;
    const factory = vi
      .fn()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second);
    const handle = createAsrHandler(() => {}, factory, { warmup: false });

    await handle({ type: "load", model: "a", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "load", model: "b", opts: { device: "wasm", dtype: "q8" } });

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(second).not.toBe(first); // the new model is the live one
  });

  it("loads the next model even when disposing the previous one fails", async () => {
    const dispose = vi.fn().mockRejectedValue(new Error("device lost"));
    const first = Object.assign(vi.fn(), { dispose }) as unknown as AsrPipeline;
    const second = vi.fn().mockResolvedValue({ text: "second" }) as unknown as AsrPipeline;
    const posted: AsrResponse[] = [];
    const factory = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const handle = createAsrHandler((m) => posted.push(m), factory, { warmup: false });

    await handle({ type: "load", model: "a", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "load", model: "b", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 1, audio: new Float32Array([0]) });

    // A failed dispose must not abort the load or leave the old model live.
    expect(posted).toContainEqual({ type: "ready", model: "b", backend: "wasm" });
    expect(second).toHaveBeenCalled();
    expect(posted[posted.length - 1]).toEqual({
      type: "result",
      id: 1,
      result: { text: "second" },
    });
  });
});
