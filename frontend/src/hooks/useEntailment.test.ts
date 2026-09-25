// `useEntailment` — the faithfulness half of `/summarization`, and three of its
// four behaviours are ones that fail silently.
//
//   1. **One pass per sentence, never one call with N labels.** Passing the
//      summary's sentences as one label list softmaxes them against each other,
//      so the scores sum to 1 and answer "which sentence is most entailed" —
//      not "does the article support this sentence". Plausible numbers, wrong
//      question, nothing throwing.
//   2. **`hypothesis_template: "{}"`.** Without it the pipeline wraps each
//      sentence in "This example is {}." and scores a sentence nobody wrote.
//      Third page in the repo to need this.
//   3. **Sequential, not `Promise.all`.** Two overlapping calls into one ONNX
//      session is not a guarantee worth relying on.

import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const post = vi.fn();
const pipeline = vi.fn();
const load = vi.fn();

vi.mock("@/hooks/useTextPipeline", () => ({
  useTextPipeline: (...args: unknown[]) => {
    pipeline(...args);
    return {
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      progress: null,
      loadProgress: null,
      loadedInMs: null,
      backend: "wasm",
      running: false,
      error: null,
      run: post,
      load,
      retry: vi.fn(),
      cancel: vi.fn(),
    };
  },
}));

const { useEntailment } = await import("./useEntailment");
const { FAITHFULNESS_MODEL, ZERO_SHOT_TEXT_MODELS } = await import(
  "@/text/catalogue"
);
const { BARE_HYPOTHESIS_TEMPLATE } = await import("@/text/zeroShot");

const ARTICLE =
  "WebGPU shipped in Chrome 113. It exposes compute shaders to the web.";
const SENTENCES = [
  "WebGPU shipped in Chrome.",
  "WebGPU was invented in 1997.",
];

describe("useEntailment", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    post.mockImplementation(async (_premise, args: unknown[]) => {
      const [labels] = args as [string[]];
      return { labels, scores: [0.9] };
    });
  });

  it("runs one forward pass per sentence, each with one label", async () => {
    const hook = renderHook(() => useEntailment());

    await act(async () => {
      await hook.result.current.run(ARTICLE, SENTENCES);
    });

    expect(post).toHaveBeenCalledTimes(SENTENCES.length);
    for (const [i, sentence] of SENTENCES.entries()) {
      const [premise, args] = post.mock.calls[i];
      expect(premise).toBe(ARTICLE);
      // Exactly one label. A two-label call would softmax the sentences
      // against each other and answer a different question entirely.
      expect((args as [string[]])[0]).toEqual([sentence]);
    }
  });

  it("passes the bare hypothesis template on every call", async () => {
    const hook = renderHook(() => useEntailment());

    await act(async () => {
      await hook.result.current.run(ARTICLE, SENTENCES);
    });

    expect(BARE_HYPOTHESIS_TEMPLATE).toBe("{}");
    for (const [, args] of post.mock.calls) {
      expect((args as [string[], Record<string, unknown>])[1]).toEqual({
        hypothesis_template: BARE_HYPOTHESIS_TEMPLATE,
      });
    }
  });

  it("scores sentences one at a time, never overlapping", async () => {
    let inflight = 0;
    let peak = 0;
    post.mockImplementation(async () => {
      peak = Math.max(peak, ++inflight);
      await Promise.resolve();
      inflight--;
      return { labels: ["x"], scores: [0.5] };
    });

    const hook = renderHook(() => useEntailment());
    await act(async () => {
      await hook.result.current.run(ARTICLE, [
        "one",
        "two",
        "three",
        "four",
      ]);
    });

    expect(peak).toBe(1);
  });

  it("keeps each sentence beside its own score, in the order given", async () => {
    post
      .mockResolvedValueOnce({ labels: [SENTENCES[0]], scores: [0.97] })
      .mockResolvedValueOnce({ labels: [SENTENCES[1]], scores: [0.02] });

    const hook = renderHook(() => useEntailment());
    let out: { sentence: string; score: number }[] = [];
    await act(async () => {
      out = await hook.result.current.run(ARTICLE, SENTENCES);
    });

    // A per-sentence score is the whole point: an aggregate over the summary
    // hides the single fabricated clause.
    expect(out).toEqual([
      { sentence: SENTENCES[0], score: 0.97 },
      { sentence: SENTENCES[1], score: 0.02 },
    ]);
    expect(hook.result.current.result).toEqual(out);
  });

  it("scores a missing number as 0 rather than NaN", async () => {
    post.mockResolvedValueOnce({ labels: ["a"] });
    const hook = renderHook(() => useEntailment());

    let out: { score: number }[] = [];
    await act(async () => {
      out = await hook.result.current.run(ARTICLE, ["a"]);
    });

    // NaN renders as "NaN%" in the score strip and sorts unpredictably.
    expect(out[0].score).toBe(0);
    expect(Number.isNaN(out[0].score)).toBe(false);
  });

  it("does nothing, and asks nothing, for no sentences", async () => {
    const hook = renderHook(() => useEntailment());

    await act(async () => {
      expect(await hook.result.current.run(ARTICLE, [])).toEqual([]);
    });
    expect(post).not.toHaveBeenCalled();
  });

  it("borrows the zero-shot page's cheapest entry instead of adding a model", () => {
    renderHook(() => useEntailment());

    const meta = ZERO_SHOT_TEXT_MODELS.find((m) => m.id === FAITHFULNESS_MODEL)!;
    expect(meta).toBeDefined();
    expect(pipeline).toHaveBeenCalledWith(
      meta.task,
      FAITHFULNESS_MODEL,
      false,
      meta.dtypes,
    );
    // It is a *shared* catalogue entry, not a private id: the page quotes its
    // size from the same place `/zero-shot-classification` does.
    expect(meta.task).toBe("zero-shot-classification");
  });

  it("exposes the model id so the route can quote the second download", () => {
    const hook = renderHook(() => useEntailment());
    expect(hook.result.current.modelId).toBe(FAITHFULNESS_MODEL);
  });

  it("downloads nothing until the route's own opt-in", () => {
    renderHook(() => useEntailment());
    expect(load).not.toHaveBeenCalled();

    pipeline.mockClear();
    renderHook(() => useEntailment(true));
    const meta = ZERO_SHOT_TEXT_MODELS.find((m) => m.id === FAITHFULNESS_MODEL)!;
    expect(pipeline).toHaveBeenCalledWith(
      meta.task,
      FAITHFULNESS_MODEL,
      true,
      meta.dtypes,
    );
  });

  it("carries no precision pin, and that is the catalogue's answer not a gap", () => {
    const meta = ZERO_SHOT_TEXT_MODELS.find((m) => m.id === FAITHFULNESS_MODEL)!;
    // MobileBERT takes `loadOpts()`'s defaults on both backends — fp16 on
    // WebGPU, q8 on WASM — so `dtypes` is genuinely absent rather than
    // forgotten. Asserted so that a pin arriving later is a deliberate edit to
    // this expectation, with a measurement behind it, rather than a silent
    // change of what the page downloads.
    expect(meta.dtypes).toBeUndefined();
  });
});
