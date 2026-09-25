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

  // `aggregation_strategy: "simple"` is pinned in the engine rather than passed
  // by the hook, because forgetting it is a **rendering bug** rather than an
  // error: without it the pipeline returns one result per subword token, so
  // "Wellington" comes back as three spans and the page paints three
  // highlights across one word. Pinning it means there is one call site to get
  // right instead of every future one.
  it("pins aggregation_strategy for token classification", async () => {
    const pipe = vi.fn().mockResolvedValue([]);
    const handle = createTextHandler(
      () => {},
      async () => pipe as unknown as CallableTextPipeline,
      { warmup: false },
    );
    await handle({
      type: "load",
      task: "token-classification",
      model: "Xenova/bert-base-NER",
      opts: { device: "wasm", dtype: "q8" },
    });
    await handle({ type: "run", id: 1, input: "Priya flew to Berlin" });

    expect(pipe).toHaveBeenCalledWith("Priya flew to Berlin", {
      aggregation_strategy: "simple",
    });
  });

  it("merges the pinned option into a caller's own options", async () => {
    const pipe = vi.fn().mockResolvedValue([]);
    const handle = createTextHandler(
      () => {},
      async () => pipe as unknown as CallableTextPipeline,
      { warmup: false },
    );
    await handle({
      type: "load",
      task: "token-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    await handle({ type: "run", id: 1, input: "x", args: [{ ignore_labels: [] }] });

    expect(pipe).toHaveBeenCalledWith("x", {
      ignore_labels: [],
      aggregation_strategy: "simple",
    });
  });

  it("leaves other tasks' args untouched", async () => {
    const pipe = vi.fn().mockResolvedValue([]);
    const handle = createTextHandler(
      () => {},
      async () => pipe as unknown as CallableTextPipeline,
      { warmup: false },
    );
    await handle({
      type: "load",
      task: "text-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    await handle({ type: "run", id: 1, input: "x", args: [{ top_k: 6 }] });

    expect(pipe).toHaveBeenCalledWith("x", { top_k: 6 });
  });
});

// --- fill-mask ---------------------------------------------------------------
//
// The one task whose *input* depends on the loaded tokenizer and whose result
// carries a fact about it back. Everything asserted here was measured against
// the real pipeline first: a wrong mask literal throws, and the warm-up string
// needs a mask of its own.

/** A fake fill-mask pipeline that records what it was actually called with. */
function maskPipe(maskToken: string | null) {
  const calls: unknown[][] = [];
  const pipe = Object.assign(
    vi.fn(async (...args: unknown[]) => {
      calls.push(args);
      return [{ token_str: " Paris", score: 0.67 }];
    }),
    maskToken === null ? {} : { tokenizer: { mask_token: maskToken } },
  ) as unknown as CallableTextPipeline;
  return { pipe, calls };
}

describe("createTextHandler — fill-mask", () => {
  afterEach(() => clearGpu());

  async function loaded(maskToken: string | null, warmup = false) {
    const posted: TextResponse[] = [];
    const { pipe, calls } = maskPipe(maskToken);
    const handle = createTextHandler((m) => posted.push(m), async () => pipe, {
      warmup,
    });
    await handle({
      type: "load",
      task: "fill-mask",
      model: "Xenova/roberta-base",
      opts: { device: "wasm", dtype: "q8" },
    });
    return { handle, posted, calls };
  }

  it("rewrites the caller's mask literal to the tokenizer's own", async () => {
    // The page inserted `[MASK]` — from a catalogue entry, or from text the
    // user pasted — and the loaded checkpoint is RoBERTa. Without this the
    // pipeline raises "Mask token (<mask>) not found in text."
    const { handle, calls } = await loaded("<mask>");
    await handle({
      type: "run",
      id: 1,
      input: "The capital of France is [MASK].",
      mask: "[MASK]",
      args: [{ top_k: 5 }],
    });

    expect(calls[0][0]).toBe("The capital of France is <mask>.");
  });

  it("leaves the input alone when the two already agree", async () => {
    const { handle, calls } = await loaded("[MASK]");
    await handle({
      type: "run",
      id: 1,
      input: "a [MASK] b",
      mask: "[MASK]",
    });
    expect(calls[0][0]).toBe("a [MASK] b");
  });

  it("rewrites every prompt in a batch, not only the first", async () => {
    // The bias probe sends six prompts in one call; a rewrite that stopped at
    // the first would fail five of them.
    const { handle, calls } = await loaded("<mask>");
    await handle({
      type: "run",
      id: 1,
      input: ["a [MASK]", "b [MASK]"],
      mask: "[MASK]",
    });
    expect(calls[0][0]).toEqual(["a <mask>", "b <mask>"]);
  });

  it("returns the resolved mask beside the fillings", async () => {
    const { handle, posted } = await loaded("<mask>");
    await handle({ type: "run", id: 7, input: "a [MASK]", mask: "[MASK]" });

    expect(last(posted)).toEqual({
      type: "result",
      id: 7,
      result: {
        mask: "<mask>",
        fills: [{ token_str: " Paris", score: 0.67 }],
      },
    });
  });

  it("reports a null mask rather than inventing one", async () => {
    // A checkpoint with no mask token cannot be rewritten to anything; the page
    // needs to be told that rather than shown the literal it sent.
    const { handle, posted } = await loaded(null);
    await handle({ type: "run", id: 1, input: "a [MASK]", mask: "[MASK]" });

    expect(last(posted)).toMatchObject({ result: { mask: null } });
    // And the input passed through untouched — there was nothing to swap to.
    expect((last(posted) as { result: { fills: unknown } }).result.fills).toEqual([
      { token_str: " Paris", score: 0.67 },
    ]);
  });

  it("warms up on a string that actually contains a mask", async () => {
    // `FillMaskPipeline` throws on an input with no mask — after the forward
    // pass, so a maskless warm-up looks harmless and is one dependency
    // refactor away from warming nothing at all.
    const { calls } = await loaded("<mask>", true);
    expect(calls).toHaveLength(1);
    expect(String(calls[0][0])).toContain("<mask>");
  });

  it("does not touch another task's input", async () => {
    const posted: TextResponse[] = [];
    const { pipe, calls } = maskPipe("<mask>");
    const handle = createTextHandler((m) => posted.push(m), async () => pipe, {
      warmup: false,
    });
    await handle({
      type: "load",
      task: "text-classification",
      model: "m",
      opts: { device: "wasm", dtype: "q8" },
    });
    await handle({ type: "run", id: 1, input: "a [MASK] b", mask: "[MASK]" });

    expect(calls[0][0]).toBe("a [MASK] b");
    // …and the result is the pipeline's own shape, not a fill-mask envelope.
    expect(last(posted)).toEqual({
      type: "result",
      id: 1,
      result: [{ token_str: " Paris", score: 0.67 }],
    });
  });
});

describe("createTextHandler — feature extraction", () => {
  afterEach(() => clearGpu());

  /** Load a feature-extraction pipeline whose call returns `output`. */
  async function loaded(output: unknown) {
    clearGpu();
    const posted: TextResponse[] = [];
    const call = vi.fn(async () => output);
    const pipe = call as unknown as CallableTextPipeline;
    const handle = createTextHandler(
      (m) => posted.push(m),
      async () => pipe,
      { warmup: false },
    );
    await handle({
      type: "load",
      task: "feature-extraction",
      model: "Xenova/all-MiniLM-L6-v2",
    });
    posted.length = 0;
    return { handle, posted, call };
  }

  // Normalising is pinned in the engine because every consumer compares by
  // cosine; the **pooling** deliberately is not, because it is a property of
  // the checkpoint and the hook is the single call site for it.
  it("pins normalize and passes the caller's pooling through", async () => {
    const { handle, call } = await loaded({ data: [1, 0], dims: [1, 2] });
    await handle({
      type: "run",
      id: 1,
      input: "hello",
      args: [{ pooling: "cls" }],
    });

    expect(call).toHaveBeenCalledWith("hello", {
      pooling: "cls",
      normalize: true,
    });
  });

  it("cannot be talked out of normalising", async () => {
    const { handle, call } = await loaded({ data: [1, 0], dims: [1, 2] });
    await handle({
      type: "run",
      id: 1,
      input: "hello",
      args: [{ pooling: "mean", normalize: false }],
    });

    expect(call).toHaveBeenCalledWith("hello", {
      pooling: "mean",
      normalize: true,
    });
  });

  // The transport problem this arm introduced. A `Tensor` keeps `data`/`dims`
  // as prototype getters, so `postMessage` refuses it outright — the engine
  // flattens every result through `model/serialize.ts`, and a plain object
  // standing in for a Tensor here is what makes that assertable without the
  // runtime.
  it("flattens the tensor it posts back, and copies the buffer", async () => {
    const data = new Float32Array([0.5, -0.5]);
    const tensor = {
      dims: [1, 2],
      get data() {
        return data;
      },
      type: "float32",
    };
    const { handle, posted } = await loaded(tensor);
    await handle({ type: "run", id: 7, input: "hello", args: [{ pooling: "mean" }] });

    const message = last(posted);
    expect(message?.type).toBe("result");
    const result = (message as { result: { data: Float32Array; dims: number[] } })
      .result;
    // Own properties, not getters — which is the whole difference.
    expect(Object.prototype.hasOwnProperty.call(result, "data")).toBe(true);
    expect([...result.data]).toEqual([0.5, -0.5]);
    expect(result.dims).toEqual([1, 2]);
    // A copy: the posted buffer must not alias a view the runtime may reuse.
    expect(result.data).not.toBe(data);
  });

  it("warms up with a pooling, so the warm-up call is valid", async () => {
    clearGpu();
    const posted: TextResponse[] = [];
    const call = vi.fn(async (...args: unknown[]) => {
      void args;
      return { data: [1], dims: [1, 1] };
    });
    const pipe = call as unknown as CallableTextPipeline;
    const handle = createTextHandler((m) => posted.push(m), async () => pipe);
    await handle({
      type: "load",
      task: "feature-extraction",
      model: "Xenova/all-MiniLM-L6-v2",
    });

    expect(call).toHaveBeenCalledTimes(1);
    expect(call.mock.calls[0][1]).toMatchObject({ normalize: true });
    expect(
      posted.some(
        (m) => m.type === "progress" && m.progress.status === "warmup",
      ),
    ).toBe(true);
  });
});
