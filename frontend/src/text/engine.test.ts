import { afterEach, describe, expect, it, vi } from "vitest";

import { createTextHandler, type CallableTextPipeline } from "./engine";
import type { TextResponse } from "./types";

/** The most recent message. `Array.prototype.at` is past this project's lib. */
function last<T>(xs: T[]): T | undefined {
  return xs[xs.length - 1];
}

// pickBackend (used when a `load` omits opts) touches navigator.gpu; keep it
// absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

describe("createTextHandler", () => {
  afterEach(() => clearGpu());

  it("loads a model for the requested task and posts progress then ready", async () => {
    clearGpu();
    const posted: TextResponse[] = [];
    const pipe = vi.fn() as unknown as CallableTextPipeline;
    const factory = vi.fn(async (_task: string, _model: string, opts) => {
      opts.progress_callback?.({
        status: "download",
        file: "model.onnx",
        progress: 30,
      });
      return pipe;
    });

    const handle = createTextHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle({
      type: "load",
      task: "text-classification",
      model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    });

    expect(factory).toHaveBeenCalledWith(
      "text-classification",
      "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(posted).toEqual([
      {
        type: "progress",
        progress: { status: "download", file: "model.onnx", progress: 30 },
      },
      {
        type: "ready",
        model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
        backend: "wasm",
      },
    ]);
  });

  it("warms the model up before ready, and survives a warm-up that throws", async () => {
    const posted: TextResponse[] = [];
    const pipe = vi
      .fn()
      .mockRejectedValueOnce(new Error("wasm JIT blew up"))
      .mockResolvedValue([]) as unknown as CallableTextPipeline;
    const handle = createTextHandler((m) => posted.push(m), async () => pipe);

    await handle({
      type: "load",
      task: "text-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });

    // The warm-up ran, it failed, and the load still reached `ready` — a page
    // that refuses to load because a throwaway inference threw is strictly
    // worse than one that pays the compile cost on the first real request.
    expect(pipe).toHaveBeenCalledOnce();
    expect(posted).toEqual([
      { type: "progress", progress: { status: "warmup" } },
      { type: "ready", model: "m", backend: "wasm" },
    ]);
  });

  it("applies a per-backend dtype override from the catalogue entry", async () => {
    const factory = vi.fn(async () => vi.fn() as unknown as CallableTextPipeline);
    const handle = createTextHandler(() => {}, factory, { warmup: false });

    await handle({
      type: "load",
      task: "text-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
      dtypes: { wasm: "fp32" },
    });

    expect(factory).toHaveBeenCalledWith(
      "text-classification",
      "m",
      expect.objectContaining({ device: "wasm", dtype: "fp32" }),
    );
  });

  it("disposes the previous model before the new one can go stale", async () => {
    const order: string[] = [];
    const first = Object.assign(vi.fn(), {
      dispose: vi.fn(async () => {
        order.push("dispose");
      }),
    }) as unknown as CallableTextPipeline;
    const second = vi.fn() as unknown as CallableTextPipeline;

    let call = 0;
    const handle = createTextHandler(() => {}, async () => {
      order.push("factory");
      return call++ === 0 ? first : second;
    }, { warmup: false });

    const load = { type: "load" as const, task: "text-classification" as const, opts: { device: "wasm" as const, dtype: "q8" as const } };
    await handle({ ...load, model: "a" });
    await handle({ ...load, model: "b" });

    // The old model is freed before the new one is even requested — one model
    // live at a time, and the reference is nulled first so a teardown that
    // throws cannot leave a stale model reachable.
    expect(order).toEqual(["factory", "dispose", "factory"]);
  });

  it("reaches ready even when the previous model's teardown throws", async () => {
    const posted: TextResponse[] = [];
    const bad = Object.assign(vi.fn(), {
      dispose: vi.fn().mockRejectedValue(new Error("backend gone")),
    }) as unknown as CallableTextPipeline;

    const handle = createTextHandler(
      (m) => posted.push(m),
      async () => bad,
      { warmup: false },
    );
    const load = { type: "load" as const, task: "text-classification" as const, opts: { device: "wasm" as const, dtype: "q8" as const } };
    await handle({ ...load, model: "a" });
    await handle({ ...load, model: "b" });

    expect(posted.filter((m) => m.type === "ready")).toHaveLength(2);
    expect(posted.some((m) => m.type === "error")).toBe(false);
  });

  it("spreads run args positionally after the input", async () => {
    const pipe = vi.fn().mockResolvedValue([{ label: "POSITIVE", score: 1 }]);
    const posted: TextResponse[] = [];
    const handle = createTextHandler(
      (m) => posted.push(m),
      async () => pipe as unknown as CallableTextPipeline,
      { warmup: false },
    );
    await handle({
      type: "load",
      task: "text-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    await handle({ type: "run", id: 7, input: "hi", args: [{ top_k: 6 }] });

    expect(pipe).toHaveBeenCalledWith("hi", { top_k: 6 });
    expect(last(posted)).toEqual({
      type: "result",
      id: 7,
      result: [{ label: "POSITIVE", score: 1 }],
    });
  });

  it("carries the request id on a run error and omits it on a load error", async () => {
    const posted: TextResponse[] = [];
    const handle = createTextHandler(
      (m) => posted.push(m),
      async (_t, model) => {
        if (model === "broken") throw new Error("404 not found");
        return vi.fn().mockRejectedValue(new Error("bad input")) as unknown as CallableTextPipeline;
      },
      { warmup: false },
    );

    await handle({ type: "load", task: "text-classification", model: "broken", opts: { device: "wasm", dtype: "q8" } });
    // A load failure is Machine A: no id, so the hook moves `status` to error.
    expect(last(posted)).toEqual({ type: "error", error: "404 not found" });

    await handle({ type: "load", task: "text-classification", model: "ok", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 3, input: "x" });
    // A run failure is Machine B: it carries the id, so the model stays loaded.
    expect(last(posted)).toEqual({ type: "error", id: 3, error: "bad input" });
  });

  it("rejects a run with no model loaded", async () => {
    const posted: TextResponse[] = [];
    const handle = createTextHandler((m) => posted.push(m), async () => vi.fn() as unknown as CallableTextPipeline);
    await handle({ type: "run", id: 1, input: "x" });
    expect(posted).toEqual([{ type: "error", id: 1, error: "No model loaded" }]);
  });
});
