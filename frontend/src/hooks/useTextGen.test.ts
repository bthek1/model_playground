// `useTextGen` — the hook whose point is how little it contains.
//
// #30 put the `partial` variant in the **shared** `ModelResponse` envelope with
// the claim that "NLP text-generation will need exactly the same thing". This
// is the page that cashes that claim, so these tests assert the wrapper adds
// nothing: the stream comes straight off `useModelWorker`, `running` stays the
// hook's inflight count, and the only thing `useTextGen` decides is whether the
// prompt is a continuation or a chat turn — a fact about the **checkpoint**,
// not a page setting.
//
// The rules about *dropping* a stale partial and clearing the previous run's
// stream live in `useModelWorker.test.ts`, where the behaviour lives. Asserting
// them again here would pin the mock rather than the code.

import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const post = vi.fn();
const load = vi.fn();
const worker = vi.fn();
let partial: unknown = null;

vi.mock("@/model/useModelWorker", () => ({
  useModelWorker: (opts: unknown) => {
    worker(opts);
    return {
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      progress: null,
      loadProgress: null,
      loadedInMs: 1234,
      backend: "webgpu",
      running: false,
      error: null,
      partial,
      run: post,
      load,
      retry: vi.fn(),
      cancel: vi.fn(),
    };
  },
}));
vi.mock("@/text/client", () => ({ createTextGenWorker: vi.fn() }));

const { useTextGen } = await import("./useTextGen");
const { TEXTGEN_MODELS, DEFAULT_TEXTGEN_MODEL } = await import(
  "@/text/catalogue"
);
const { DEFAULT_DECODING } = await import("@/text/textgenTypes");

const SMOL = "HuggingFaceTB/SmolLM2-360M-Instruct";
const GPT2 = "Xenova/gpt2";

describe("useTextGen", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    partial = null;
    post.mockResolvedValue({ text: "a continuation", tokens: 3, ms: 90 });
  });

  it("keys the worker per checkpoint, so a model change is a new worker", () => {
    renderHook(() => useTextGen(SMOL));
    expect(worker).toHaveBeenCalledWith(
      expect.objectContaining({ key: `text-generation:${SMOL}` }),
    );

    worker.mockClear();
    renderHook(() => useTextGen(GPT2));
    expect(worker).toHaveBeenCalledWith(
      expect.objectContaining({ key: `text-generation:${GPT2}` }),
    );
  });

  it("sends the decoding block through untouched", async () => {
    const hook = renderHook(() => useTextGen(SMOL));

    await act(async () => {
      await hook.result.current.run("Once upon", DEFAULT_DECODING);
    });

    // Untouched is the assertion: the hook must not normalise, clamp or drop a
    // knob. Under greedy the route omits the sampling knobs *at the route*, so
    // a hook that quietly re-added defaults would undo that.
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: "Once upon",
        decoding: DEFAULT_DECODING,
      }),
    );
  });

  it("defaults to greedy, so the first thing a visitor sees is reproducible", () => {
    // The property `just fe-e2e-textgen` pins in a real browser — greedy twice
    // must be byte-identical — is only meaningful if greedy is the default.
    expect(DEFAULT_DECODING.doSample).toBe(false);
  });

  it("asks an instruct model a question and a base model for a continuation", async () => {
    const instruct = TEXTGEN_MODELS.find((m) => m.id === SMOL)!;
    const base = TEXTGEN_MODELS.find((m) => m.id === GPT2)!;
    expect(instruct.instruct).toBe(true);
    expect(base.instruct).toBeFalsy();

    const a = renderHook(() => useTextGen(SMOL));
    await act(async () => {
      await a.result.current.run("hi", DEFAULT_DECODING);
    });
    expect(post).toHaveBeenLastCalledWith(
      expect.objectContaining({ chat: true }),
    );

    const b = renderHook(() => useTextGen(GPT2));
    await act(async () => {
      await b.result.current.run("hi", DEFAULT_DECODING);
    });
    // Sending a raw prompt to an instruct model produces a fluent non-answer
    // with nothing failing, so this is a fact about the checkpoint rather than
    // a setting the page offers.
    expect(post).toHaveBeenLastCalledWith(
      expect.objectContaining({ chat: false }),
    );
  });

  it("passes the entry's precision, graph name and f16 requirement to the load", () => {
    const gpt2 = TEXTGEN_MODELS.find((m) => m.id === GPT2)!;
    renderHook(() => useTextGen(GPT2));

    const { loadMessage } = worker.mock.calls[0][0] as {
      loadMessage: Record<string, unknown>;
    };
    expect(loadMessage.model).toBe(GPT2);
    if (gpt2.dtypes) expect(loadMessage.dtypes).toEqual(gpt2.dtypes);
    if (gpt2.modelFile) expect(loadMessage.modelFile).toBe(gpt2.modelFile);
    // Absent keys stay absent rather than arriving as `undefined`: the worker
    // spreads this into the load options, and an explicit `undefined` dtype is
    // not the same request as no dtype at all.
    if (!gpt2.modelFile) expect("modelFile" in loadMessage).toBe(false);
    if (!gpt2.requireShaderF16)
      expect("requireShaderF16" in loadMessage).toBe(false);
  });

  it("marks every f16 entry as needing shader-f16", () => {
    for (const m of TEXTGEN_MODELS) {
      const spec = m.dtypes?.webgpu;
      const f16 = typeof spec === "string" && spec.includes("f16");
      if (f16) {
        // An adapter without the feature loads the weights, reports ready, and
        // then fails on the first operator of every run — after the user has
        // paid for the download. The picker can only disable the row if the
        // entry declares it.
        expect(m.requireShaderF16).toBe(true);
      }
    }
  });

  it("surfaces the worker's stream as-is", () => {
    partial = { text: "a cont" };
    const hook = renderHook(() => useTextGen(SMOL));
    // No copy, no re-wrap: the hook's whole streaming implementation is this
    // pass-through, which is the claim #30's shared envelope made.
    expect(hook.result.current.partial).toEqual({ text: "a cont" });
  });

  it("holds the finished result and returns it", async () => {
    const hook = renderHook(() => useTextGen(SMOL));
    expect(hook.result.current.result).toBeNull();

    let out: unknown;
    await act(async () => {
      out = await hook.result.current.run("Once upon", DEFAULT_DECODING);
    });

    expect(out).toEqual({ text: "a continuation", tokens: 3, ms: 90 });
    expect(hook.result.current.result).toEqual(out);
  });

  it("falls back to the first entry for an unknown id", () => {
    const hook = renderHook(() => useTextGen("nobody/nothing"));
    expect(hook.result.current.meta.id).toBe(DEFAULT_TEXTGEN_MODEL);
  });

  it("does not load on mount unless asked", () => {
    renderHook(() => useTextGen(SMOL));
    expect(load).not.toHaveBeenCalled();
    expect(worker).toHaveBeenCalledWith(
      expect.objectContaining({ autoLoad: false }),
    );

    worker.mockClear();
    renderHook(() => useTextGen(SMOL, true));
    expect(worker).toHaveBeenCalledWith(
      expect.objectContaining({ autoLoad: true }),
    );
  });
});
