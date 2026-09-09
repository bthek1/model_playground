import { afterEach, describe, expect, it, vi } from "vitest";

import { createVisionHandler, type CallableVisionPipeline } from "./engine";
import type { VisionResponse } from "./types";
import type { ImagePayload as Payload } from "./image";

// pickBackend (used when a `load` omits opts) touches navigator.gpu; keep it
// absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

function image(side = 2): Payload {
  return {
    data: new Uint8ClampedArray(side * side * 3),
    width: side,
    height: side,
    channels: 3,
  };
}

describe("createVisionHandler", () => {
  afterEach(() => clearGpu());

  it("loads a model for the requested task and posts progress then ready", async () => {
    clearGpu();
    const posted: VisionResponse[] = [];
    const pipe = vi.fn() as unknown as CallableVisionPipeline;
    const factory = vi.fn(async (_task: string, _model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "model.onnx", progress: 30 });
      return pipe;
    });

    const handle = createVisionHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle({
      type: "load",
      task: "image-classification",
      model: "Xenova/vit-base-patch16-224",
    });

    expect(factory).toHaveBeenCalledWith(
      "image-classification",
      "Xenova/vit-base-patch16-224",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(posted).toEqual([
      {
        type: "progress",
        progress: { status: "download", file: "model.onnx", progress: 30 },
      },
      {
        type: "ready",
        model: "Xenova/vit-base-patch16-224",
        backend: "wasm",
      },
    ]);
  });

  it("warms the model up before ready, and survives a warm-up that throws", async () => {
    const posted: VisionResponse[] = [];
    const pipe = vi
      .fn()
      .mockRejectedValueOnce(new Error("shader compile blew up"))
      .mockResolvedValue([]) as unknown as CallableVisionPipeline;
    const handle = createVisionHandler((m) => posted.push(m), async () => pipe);

    await handle({
      type: "load",
      task: "image-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });

    // The warm-up ran, it failed, and the load still reached `ready`.
    expect(pipe).toHaveBeenCalledOnce();
    expect(posted.map((p) => p.type)).toEqual(["progress", "ready"]);
    expect(posted[0]).toEqual({ type: "progress", progress: { status: "warmup" } });
  });

  it("passes a valid warm-up argument for each task", async () => {
    const calls: unknown[][] = [];
    const pipe = vi.fn(async (...args: unknown[]) => {
      calls.push(args);
      return [];
    }) as unknown as CallableVisionPipeline;
    const handle = createVisionHandler(() => {}, async () => pipe);

    await handle({
      type: "load",
      task: "zero-shot-image-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });

    // Zero-shot cannot be called with no candidate labels — a bare warm-up
    // would throw every time and silently buy nothing.
    expect(calls[0][1]).toEqual(["a photo"]);
  });

  it("spreads run args positionally and returns the pipeline output", async () => {
    const posted: VisionResponse[] = [];
    const pipe = vi
      .fn()
      .mockResolvedValue([{ label: "tabby", score: 0.9 }]) as unknown as CallableVisionPipeline;
    const handle = createVisionHandler((m) => posted.push(m), async () => pipe, {
      warmup: false,
    });

    await handle({
      type: "load",
      task: "image-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    const img = image();
    await handle({ type: "run", id: 7, image: img, args: [{ top_k: 5 }] });

    expect(pipe).toHaveBeenCalledWith(img, { top_k: 5 });
    expect(posted[posted.length - 1]).toEqual({
      type: "result",
      id: 7,
      result: [{ label: "tabby", score: 0.9 }],
    });
  });

  it("holds one model live at a time, disposing the previous one first", async () => {
    const order: string[] = [];
    const first = Object.assign(vi.fn(), {
      dispose: vi.fn(async () => {
        order.push("dispose:first");
      }),
    }) as unknown as CallableVisionPipeline;
    const second = vi.fn() as unknown as CallableVisionPipeline;

    let n = 0;
    const handle = createVisionHandler(() => {}, async () => {
      order.push(`load:${++n}`);
      return n === 1 ? first : second;
    }, { warmup: false });

    const load = { type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } } as const;
    await handle({ ...load, task: "image-classification" });
    await handle({ ...load, task: "image-classification" });

    expect(order).toEqual(["load:1", "dispose:first", "load:2"]);
  });

  it("does not leave a stale model live when disposal throws", async () => {
    const doomed = Object.assign(vi.fn(), {
      dispose: vi.fn(async () => {
        throw new Error("backend teardown failed");
      }),
    }) as unknown as CallableVisionPipeline;
    const replacement = vi.fn().mockResolvedValue("second result") as unknown as CallableVisionPipeline;
    const posted: VisionResponse[] = [];

    let n = 0;
    const handle = createVisionHandler((m) => posted.push(m), async () => (++n === 1 ? doomed : replacement), { warmup: false });

    const load = { type: "load", task: "image-classification", model: "m", opts: { device: "wasm", dtype: "q8" } } as const;
    await handle(load);
    await handle(load);
    await handle({ type: "run", id: 1, image: image() });

    // The failed dispose did not wedge the engine, and the run reached the new
    // model rather than the one that refused to die.
    expect(replacement).toHaveBeenCalled();
    expect(doomed).not.toHaveBeenCalled();
    expect(posted[posted.length - 1]).toEqual({ type: "result", id: 1, result: "second result" });
  });

  it("reports a load failure with no id and a run failure with one", async () => {
    const posted: VisionResponse[] = [];
    const failing = createVisionHandler((m) => posted.push(m), async () => {
      throw new Error("404 model not found");
    });
    await failing({
      type: "load",
      task: "image-classification",
      model: "nope/nope",
      opts: { device: "wasm", dtype: "q8" },
    });
    // No id: Machine A. The page moves to `error`.
    expect(posted[posted.length - 1]).toEqual({ type: "error", error: "404 model not found" });

    const posted2: VisionResponse[] = [];
    const pipe = vi.fn().mockRejectedValue(new Error("bad input")) as unknown as CallableVisionPipeline;
    const handle = createVisionHandler((m) => posted2.push(m), async () => pipe, { warmup: false });
    await handle({ type: "load", task: "image-classification", model: "m", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 3, image: image() });
    // With an id: Machine B. The model stays loaded.
    expect(posted2[posted2.length - 1]).toEqual({ type: "error", id: 3, error: "bad input" });
  });

  it("refuses to run before a model is loaded", async () => {
    const posted: VisionResponse[] = [];
    const handle = createVisionHandler((m) => posted.push(m), async () => vi.fn() as unknown as CallableVisionPipeline);
    await handle({ type: "run", id: 1, image: image() });
    expect(posted).toEqual([{ type: "error", id: 1, error: "No model loaded" }]);
  });
});

describe("precision overrides", () => {
  it("pins the dtype for the backend it actually landed on", async () => {
    // A catalogue entry can override `loadOpts()` per backend — the escape
    // hatch for an export that is only correct at one precision.
    const factory = vi.fn(async () => vi.fn() as unknown as CallableVisionPipeline);
    const handle = createVisionHandler(() => {}, factory, { warmup: false });

    await handle({
      type: "load",
      task: "image-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
      dtypes: { wasm: "fp32" },
    });

    expect(factory).toHaveBeenCalledWith(
      "image-classification",
      "m",
      expect.objectContaining({ device: "wasm", dtype: "fp32" }),
    );
  });

  it("leaves the default alone for a backend it says nothing about", async () => {
    const factory = vi.fn(async () => vi.fn() as unknown as CallableVisionPipeline);
    const handle = createVisionHandler(() => {}, factory, { warmup: false });

    await handle({
      type: "load",
      task: "image-classification",
      model: "m",
      opts: { device: "webgpu", dtype: "fp16" },
      dtypes: { wasm: "fp32" },
    });

    expect(factory).toHaveBeenCalledWith(
      "image-classification",
      "m",
      expect.objectContaining({ device: "webgpu", dtype: "fp16" }),
    );
  });
});
