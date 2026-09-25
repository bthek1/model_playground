// `useSummarize` — a thin wrapper, and the tests are about the two run
// parameters that look like view settings and are not.
//
// `max_new_tokens` and `min_length` change the generation, not a view of it:
// there is nothing in a finished summary from which a longer one could be
// derived. So they are free to edit, they must never imply a download, and the
// next GENERATE is a real second inference. `min_length` in particular has to
// be **sent explicitly and be non-zero** — DistilBART's own config carries one,
// and a pipeline call that omits it gets a model happy to emit a single clause
// and call it a summary.

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
      backend: "webgpu",
      running: false,
      error: null,
      run: post,
      load,
      retry: vi.fn(),
      cancel: vi.fn(),
    };
  },
}));

const { useSummarize } = await import("./useSummarize");
const {
  SUMMARIZER_MODELS,
  DEFAULT_SUMMARIZER,
  SUMMARY_MAX_TOKENS,
  SUMMARY_MIN_TOKENS,
} = await import("@/text/catalogue");

const T5 = "Xenova/t5-small";
const DISTILBART = "Xenova/distilbart-cnn-6-6";
const ARTICLE = "WebGPU reached the browser. It runs compute shaders.";

describe("useSummarize", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    post.mockResolvedValue([{ summary_text: " WebGPU runs compute shaders. " }]);
  });

  it("sends both length bounds, and min_length is not zero", async () => {
    const hook = renderHook(() => useSummarize(T5));

    await act(async () => {
      await hook.result.current.run({ text: ARTICLE });
    });

    // Explicit and non-zero: a summarizer told nothing about a floor will
    // happily return one clause, which reads as a broken page rather than a
    // missing parameter.
    expect(post).toHaveBeenCalledWith(ARTICLE, [
      { max_new_tokens: SUMMARY_MAX_TOKENS, min_length: SUMMARY_MIN_TOKENS },
    ]);
    expect(SUMMARY_MIN_TOKENS).toBeGreaterThan(0);
    expect(SUMMARY_MIN_TOKENS).toBeLessThan(SUMMARY_MAX_TOKENS);
  });

  it("lets a run override either bound", async () => {
    const hook = renderHook(() => useSummarize(T5));

    await act(async () => {
      await hook.result.current.run({
        text: ARTICLE,
        maxNewTokens: 40,
        minLength: 10,
      });
    });

    expect(post).toHaveBeenCalledWith(ARTICLE, [
      { max_new_tokens: 40, min_length: 10 },
    ]);
  });

  it("changing a length bound does not load the model", async () => {
    const hook = renderHook(() => useSummarize(T5));

    await act(async () => {
      await hook.result.current.run({ text: ARTICLE, maxNewTokens: 40 });
    });
    await act(async () => {
      await hook.result.current.run({ text: ARTICLE, maxNewTokens: 90 });
    });

    // Two real inferences, one model. The controls are INPUT.
    expect(load).not.toHaveBeenCalled();
    expect(post).toHaveBeenCalledTimes(2);
  });

  it("trims the summary, because the pipeline pads it with a leading space", async () => {
    const hook = renderHook(() => useSummarize(T5));

    let out = "";
    await act(async () => {
      out = await hook.result.current.run({ text: ARTICLE });
    });

    // Untrimmed, the OUTPUT panel indents the first line and the summary looks
    // misaligned beside the lead-3 baseline it is measured against.
    expect(out).toBe("WebGPU runs compute shaders.");
    expect(hook.result.current.result).toBe("WebGPU runs compute shaders.");
  });

  it("reads a bare object as well as an array, and survives a malformed one", async () => {
    const hook = renderHook(() => useSummarize(T5));

    post.mockResolvedValueOnce({ summary_text: "Bare." });
    await act(async () => {
      expect(await hook.result.current.run({ text: ARTICLE })).toBe("Bare.");
    });

    post.mockResolvedValueOnce([{}]);
    await act(async () => {
      expect(await hook.result.current.run({ text: ARTICLE })).toBe("");
    });
  });

  it("keeps the WebGPU q8 pin that is the only reason the page fits", () => {
    renderHook(() => useSummarize(DISTILBART));

    const meta = SUMMARIZER_MODELS.find((m) => m.id === DISTILBART)!;
    // The pin is a **measurement**, not a preference: fp16 is 563.6 MB and over
    // the feasibility bar, so un-pinning it removes the page silently. Asserted
    // here as well as in the catalogue test because this is the call site that
    // would carry a dropped `dtypes` to the worker.
    expect(meta.dtypes?.webgpu).toBe("q8");
    expect(pipeline).toHaveBeenCalledWith(
      meta.task,
      DISTILBART,
      false,
      meta.dtypes,
    );
  });

  it("opens on a model with a CPU path, so the page has a floor", () => {
    const first = SUMMARIZER_MODELS.find((m) => m.id === DEFAULT_SUMMARIZER)!;
    // DistilBART is `backends: ["webgpu"]`, so it cannot be the default without
    // leaving every CPU-only machine with a disabled page.
    expect(first.backends ?? ["webgpu", "wasm"]).toContain("wasm");
  });

  it("falls back to the first entry for an unknown id", () => {
    const hook = renderHook(() => useSummarize("Xenova/not-a-summarizer"));
    expect(hook.result.current.meta.id).toBe(DEFAULT_SUMMARIZER);
  });

  it("does not load on mount", () => {
    renderHook(() => useSummarize(T5));
    expect(load).not.toHaveBeenCalled();
  });
});
