import { afterEach, describe, expect, it, vi } from "vitest";

import { createTextGenHandler, type Generator } from "./textgenEngine";
import type { TextGenResponse } from "./textgenTypes";

function last<T>(xs: T[]): T | undefined {
  return xs[xs.length - 1];
}

/** No `navigator.gpu`, so the backend pick is deterministically wasm. */
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

const DECODING = {
  doSample: false,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  repetitionPenalty: 1,
  maxNewTokens: 16,
};

function fakeGenerator(overrides: Partial<Generator> = {}): Generator {
  return {
    generate: vi.fn(async () => ({ text: "Paris.", tokens: 2, ms: 10 })),
    dispose: vi.fn(async () => {}),
    ...overrides,
  };
}

describe("createTextGenHandler", () => {
  afterEach(() => clearGpu());

  it("loads, warms up, and reports ready with the resolved backend", async () => {
    clearGpu();
    const posted: TextGenResponse[] = [];
    const gen = fakeGenerator();
    const factory = vi.fn(async (_model: string, opts) => {
      opts.progress_callback?.({ status: "download", progress: 40 });
      return gen;
    });

    const handle = createTextGenHandler((m) => posted.push(m), factory);
    await handle({ type: "load", model: "HuggingFaceTB/SmolLM2-360M-Instruct" });

    // `vlmLoadOpts` on wasm is plain q4 — the 4-bit format without the f16 half,
    // because fp16 activations are a GPU format.
    expect(factory).toHaveBeenCalledWith(
      "HuggingFaceTB/SmolLM2-360M-Instruct",
      expect.objectContaining({ device: "wasm", dtype: "q4" }),
    );
    // Warm-up happened, was announced, and was capped rather than generating a
    // whole answer on the critical path to `ready`.
    expect(gen.generate).toHaveBeenCalledTimes(1);
    expect(
      (gen.generate as ReturnType<typeof vi.fn>).mock.calls[0][1],
    ).toMatchObject({ maxNewTokens: 2, doSample: false });
    expect(
      posted.some((m) => m.type === "progress" && m.progress.status === "warmup"),
    ).toBe(true);
    expect(last(posted)).toEqual({
      type: "ready",
      model: "HuggingFaceTB/SmolLM2-360M-Instruct",
      backend: "wasm",
    });
  });

  it("applies a per-backend dtype pin over the family default", async () => {
    clearGpu();
    const factory = vi.fn(async () => fakeGenerator());
    const handle = createTextGenHandler(() => {}, factory, { warmup: false });
    await handle({
      type: "load",
      model: "HuggingFaceTB/SmolLM2-360M-Instruct",
      // The measured pin: q8 is the WASM build that was verified to load and
      // generate; q4 on WASM was not.
      dtypes: { wasm: "q8" },
    });

    expect(factory).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ dtype: "q8" }),
    );
  });

  it("passes a legacy graph name through to the factory", async () => {
    clearGpu();
    const factory = vi.fn(async () => fakeGenerator());
    const handle = createTextGenHandler(() => {}, factory, { warmup: false });
    await handle({
      type: "load",
      model: "Xenova/gpt2",
      modelFile: "decoder_model_merged",
    });

    expect(factory).toHaveBeenCalledWith(
      "Xenova/gpt2",
      expect.objectContaining({ modelFile: "decoder_model_merged" }),
    );
  });

  it("never fails a load because the warm-up threw", async () => {
    clearGpu();
    const posted: TextGenResponse[] = [];
    const gen = fakeGenerator({
      generate: vi.fn(async () => {
        throw new Error("shader compile failed");
      }),
    });
    const handle = createTextGenHandler(
      (m) => posted.push(m),
      async () => gen,
    );
    await handle({ type: "load", model: "m" });

    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  // One model live at a time, and the *order* is the assertion: the reference is
  // nulled before the dispose, so a teardown that throws cannot leave a stale
  // model live. These are among the larger downloads in the app.
  it("disposes the previous model, and survives a dispose that throws", async () => {
    clearGpu();
    const first = fakeGenerator({
      dispose: vi.fn(async () => {
        throw new Error("teardown exploded");
      }),
    });
    const second = fakeGenerator();
    const posted: TextGenResponse[] = [];
    let call = 0;
    const handle = createTextGenHandler(
      (m) => posted.push(m),
      async () => (call++ === 0 ? first : second),
      { warmup: false },
    );

    await handle({ type: "load", model: "a" });
    await handle({ type: "load", model: "b" });

    expect(first.dispose).toHaveBeenCalled();
    expect(last(posted)).toMatchObject({ type: "ready", model: "b" });

    // And the *new* model is the one that runs.
    await handle({ type: "run", id: 1, prompt: "hi", decoding: DECODING, chat: false });
    expect(second.generate).toHaveBeenCalled();
    expect(first.generate).not.toHaveBeenCalled();
  });

  describe("a run", () => {
    async function loaded() {
      clearGpu();
      const posted: TextGenResponse[] = [];
      const gen: Generator = {
        generate: vi.fn(async (_prompt, _decoding, _chat, onPartial) => {
          onPartial({ text: "Par", tokens: 1 });
          onPartial({ text: "Paris", tokens: 2 });
          return { text: "Paris.", tokens: 2, ms: 42 };
        }),
      };
      const handle = createTextGenHandler(
        (m) => posted.push(m),
        async () => gen,
        { warmup: false },
      );
      await handle({ type: "load", model: "m" });
      posted.length = 0;
      return { handle, posted, gen };
    }

    it("streams partials correlated to the request id, then the result", async () => {
      const { handle, posted } = await loaded();
      await handle({
        type: "run",
        id: 7,
        prompt: "The capital of France is",
        decoding: DECODING,
        chat: true,
      });

      // Every partial carries the id of the request that will carry the result,
      // which is what lets the main thread drop a late chunk from a superseded
      // run instead of repainting a finished answer.
      const partials = posted.filter((m) => m.type === "partial");
      expect(partials).toHaveLength(2);
      for (const p of partials) expect(p).toMatchObject({ id: 7 });
      expect(partials[1]).toMatchObject({ partial: { text: "Paris", tokens: 2 } });

      expect(last(posted)).toEqual({
        type: "result",
        id: 7,
        result: { text: "Paris.", tokens: 2, ms: 42 },
      });
      // `partial` is not a status and not a running flag: no `ready` or
      // `progress` message was posted by a run.
      expect(posted.some((m) => m.type === "ready")).toBe(false);
      expect(posted.some((m) => m.type === "progress")).toBe(false);
    });

    it("hands the decoding parameters and the chat flag straight through", async () => {
      const { handle, gen } = await loaded();
      const decoding = { ...DECODING, doSample: true, temperature: 1.4 };
      await handle({ type: "run", id: 1, prompt: "p", decoding, chat: true });

      const call = (gen.generate as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(call[1]).toEqual(decoding);
      expect(call[2]).toBe(true);
    });

    it("reports a run failure against its id, so Machine A stays ready", async () => {
      clearGpu();
      const posted: TextGenResponse[] = [];
      const handle = createTextGenHandler(
        (m) => posted.push(m),
        async () => ({
          generate: vi.fn(async () => {
            throw new Error("out of memory");
          }),
        }),
        { warmup: false },
      );
      await handle({ type: "load", model: "m" });
      await handle({ type: "run", id: 3, prompt: "p", decoding: DECODING, chat: false });

      expect(last(posted)).toEqual({
        type: "error",
        id: 3,
        error: "out of memory",
      });
    });

    it("refuses to run with no model loaded, against the request id", async () => {
      const posted: TextGenResponse[] = [];
      const handle = createTextGenHandler(
        (m) => posted.push(m),
        async () => fakeGenerator(),
      );
      await handle({ type: "run", id: 9, prompt: "p", decoding: DECODING, chat: false });

      expect(last(posted)).toEqual({
        type: "error",
        id: 9,
        error: "No model loaded",
      });
    });
  });

  it("reports a load failure with no id, which is Machine A's discriminator", async () => {
    clearGpu();
    const posted: TextGenResponse[] = [];
    const handle = createTextGenHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("404 not found");
      },
    );
    await handle({ type: "load", model: "nope" });

    expect(last(posted)).toEqual({ type: "error", error: "404 not found" });
  });
});
