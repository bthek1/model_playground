import { afterEach, describe, expect, it, vi } from "vitest";

import { createPipelineHandler, type CallablePipeline } from "./pipelineEngine";
import type { PipelineResponse } from "./pipelineTypes";

// pickBackend (used when a `load` omits opts) touches navigator.gpu; keep it
// absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

describe("createPipelineHandler", () => {
  afterEach(() => clearGpu());

  it("loads a model for the requested task and posts progress then ready", async () => {
    clearGpu();
    const posted: PipelineResponse[] = [];
    const pipe = vi.fn() as unknown as CallablePipeline;
    const factory = vi.fn(async (_task: string, _model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "model.onnx", progress: 30 });
      return pipe;
    });

    const handle = createPipelineHandler((m) => posted.push(m), factory);
    await handle({
      type: "load",
      task: "audio-classification",
      model: "onnx-community/ast",
    });

    expect(factory).toHaveBeenCalledWith(
      "audio-classification",
      "onnx-community/ast",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(posted).toEqual([
      { type: "progress", progress: { status: "download", file: "model.onnx", progress: 30 } },
      { type: "ready", model: "onnx-community/ast", backend: "wasm" },
    ]);
  });

  it("spreads run args positionally (fixed-label top_k)", async () => {
    const posted: PipelineResponse[] = [];
    const pipe = vi
      .fn()
      .mockResolvedValue([{ label: "Speech", score: 0.9 }]) as unknown as CallablePipeline;
    const handle = createPipelineHandler((m) => posted.push(m), async () => pipe);

    await handle({
      type: "load",
      task: "audio-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    const audio = new Float32Array([0.1, 0.2]);
    await handle({ type: "run", id: 5, input: audio, args: [{ top_k: 6 }] });

    expect(pipe).toHaveBeenCalledWith(audio, { top_k: 6 });
    expect(posted[posted.length - 1]).toEqual({
      type: "result",
      id: 5,
      result: [{ label: "Speech", score: 0.9 }],
    });
  });

  it("spreads multiple positional args (zero-shot candidate labels)", async () => {
    const posted: PipelineResponse[] = [];
    const pipe = vi.fn().mockResolvedValue([]) as unknown as CallablePipeline;
    const handle = createPipelineHandler((m) => posted.push(m), async () => pipe);

    await handle({
      type: "load",
      task: "zero-shot-audio-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    const audio = new Float32Array([0]);
    await handle({ type: "run", id: 1, input: audio, args: [["a dog", "rain"]] });

    expect(pipe).toHaveBeenCalledWith(audio, ["a dog", "rain"]);
  });

  it("errors a run when no model is loaded", async () => {
    const posted: PipelineResponse[] = [];
    const handle = createPipelineHandler((m) => posted.push(m), async () => vi.fn() as never);

    await handle({ type: "run", id: 3, input: new Float32Array([0]) });

    expect(posted[posted.length - 1]).toEqual({
      type: "error",
      id: 3,
      error: "No model loaded",
    });
  });

  it("reports a load failure as an id-less error", async () => {
    clearGpu();
    const posted: PipelineResponse[] = [];
    const handle = createPipelineHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("download failed");
      },
    );

    await handle({
      type: "load",
      task: "audio-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });

    expect(posted[posted.length - 1]).toEqual({ type: "error", error: "download failed" });
  });

  it("disposes the previous model before loading a new one", async () => {
    const dispose = vi.fn().mockResolvedValue(undefined);
    const first = Object.assign(vi.fn(), { dispose }) as unknown as CallablePipeline;
    const second = vi.fn() as unknown as CallablePipeline;
    const factory = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const handle = createPipelineHandler(() => {}, factory);

    const load = { type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } } as const;
    await handle({ ...load, task: "audio-classification" });
    await handle({ ...load, task: "zero-shot-audio-classification" });

    expect(dispose).toHaveBeenCalledTimes(1);
  });
});
