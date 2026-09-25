// `useTranslate` — the hook whose whole correctness surface is what it does
// **not** accept.
//
// A Marian checkpoint *is* its language pair, so there is no language argument
// anywhere in the task: `tr(text)` takes the text and nothing else. The failure
// this guards is a hook that grew a `{ from, to }` and dropped it — the model
// would keep translating in the direction it was built for, so every assertion
// about the output would still pass and the page would look like it worked
// while the direction control did nothing. The only thing that catches it is
// asserting **which model id reaches the pipeline**, which is what these tests
// do on every direction the catalogue ships.

import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const post = vi.fn();
/** Every `useTextPipeline(task, model, autoLoad, dtypes)` call, in order. */
const pipeline = vi.fn();
// Stable across renders, so "was the model asked to load" is answerable. A
// fresh `vi.fn()` per render would make every re-render look like a new hook.
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

const { useTranslate, MAX_NEW_TOKENS } = await import("./useTranslate");
const { TRANSLATION_MODELS, DEFAULT_TRANSLATION_MODEL } = await import(
  "@/text/catalogue"
);

const EN_DE = "Xenova/opus-mt-en-de";
const DE_EN = "Xenova/opus-mt-de-en";

describe("useTranslate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    post.mockResolvedValue([{ translation_text: "Guten Tag" }]);
  });

  it("takes no language arguments — the pair is the model", () => {
    const hook = renderHook(() => useTranslate(EN_DE));

    // The contract, asserted as a contract: `run` is unary plus an options bag
    // that holds only `maxNewTokens`. A `{ from, to }` would have to arrive
    // here, and cannot.
    expect(hook.result.current.run.length).toBeLessThanOrEqual(2);
    expect(hook.result.current.meta.id).toBe(EN_DE);
  });

  it("sends the selected pair's own id to the pipeline", () => {
    renderHook(() => useTranslate(EN_DE));

    const meta = TRANSLATION_MODELS.find((m) => m.id === EN_DE)!;
    expect(pipeline).toHaveBeenCalledWith(meta.task, EN_DE, false, meta.dtypes);
  });

  it("changes the checkpoint when the direction changes, not just a label", () => {
    const hook = renderHook(({ id }) => useTranslate(id), {
      initialProps: { id: EN_DE },
    });
    expect(hook.result.current.meta.id).toBe(EN_DE);

    hook.rerender({ id: DE_EN });

    // The reverse pair is a **different checkpoint**, which is the assertion
    // that catches a direction control wired to a label instead of a model.
    expect(hook.result.current.meta.id).toBe(DE_EN);
    const ids = pipeline.mock.calls.map((c) => c[1]);
    expect(ids).toContain(DE_EN);
    expect(hook.result.current.meta.id).not.toBe(EN_DE);
  });

  it("every catalogue pair reaches the pipeline as its own id", () => {
    for (const m of TRANSLATION_MODELS) {
      pipeline.mockClear();
      renderHook(() => useTranslate(m.id));
      expect(pipeline).toHaveBeenCalledWith(m.task, m.id, false, m.dtypes);
    }
  });

  it("caps generation by default and lets a run override it", async () => {
    const hook = renderHook(() => useTranslate(EN_DE));

    await act(async () => {
      await hook.result.current.run("Good day");
    });
    // A seq2seq that never emits EOS decodes until something stops it, so the
    // ceiling is not optional.
    expect(post).toHaveBeenCalledWith("Good day", [
      { max_new_tokens: MAX_NEW_TOKENS },
    ]);

    await act(async () => {
      await hook.result.current.run("Good day", { maxNewTokens: 8 });
    });
    expect(post).toHaveBeenLastCalledWith("Good day", [
      { max_new_tokens: 8 },
    ]);
  });

  it("asking for a longer output does not touch the model", async () => {
    const hook = renderHook(() => useTranslate(EN_DE));

    await act(async () => {
      await hook.result.current.run("Good day", { maxNewTokens: 64 });
    });

    // `max_new_tokens` is a **run** parameter: a different question about the
    // same weights, never another download. Asserted on `load` and on the id
    // the pipeline was keyed to — *not* on how many times the hook re-rendered,
    // which a `setResult` legitimately changes.
    expect(load).not.toHaveBeenCalled();
    expect(pipeline.mock.calls.every((c) => c[1] === EN_DE)).toBe(true);
  });

  it("does not load on a direction change either — only LOAD loads", () => {
    const hook = renderHook(({ id }) => useTranslate(id), {
      initialProps: { id: EN_DE },
    });

    hook.rerender({ id: DE_EN });

    // The rule this page is most likely to break: SELECT is a free, reversible
    // choice, and switching to a ~209 MB checkpoint must not begin fetching it.
    expect(load).not.toHaveBeenCalled();
  });

  it("holds the latest translation and returns it", async () => {
    const hook = renderHook(() => useTranslate(EN_DE));
    expect(hook.result.current.result).toBeNull();

    let returned = "";
    await act(async () => {
      returned = await hook.result.current.run("Good day");
    });

    expect(returned).toBe("Guten Tag");
    expect(hook.result.current.result).toBe("Guten Tag");
  });

  it("reads a bare object as well as a one-element array", async () => {
    post.mockResolvedValueOnce({ translation_text: "Bonjour" });
    const hook = renderHook(() => useTranslate(EN_DE));

    await act(async () => {
      expect(await hook.result.current.run("Hello")).toBe("Bonjour");
    });
  });

  it("returns an empty string rather than 'undefined' on a malformed result", async () => {
    post.mockResolvedValueOnce([{}]);
    const hook = renderHook(() => useTranslate(EN_DE));

    await act(async () => {
      expect(await hook.result.current.run("Hello")).toBe("");
    });
    // The failure mode being avoided: `String(undefined)` rendering the word
    // "undefined" into the output panel as though it were a translation.
    expect(hook.result.current.result).toBe("");
  });

  it("falls back to the first pair for an id the catalogue does not have", () => {
    const hook = renderHook(() => useTranslate("Xenova/opus-mt-en-klingon"));
    expect(hook.result.current.meta.id).toBe(DEFAULT_TRANSLATION_MODEL);
  });

  it("does not load on mount unless asked", () => {
    renderHook(() => useTranslate(EN_DE));
    expect(pipeline).toHaveBeenCalledWith(
      expect.anything(),
      EN_DE,
      false,
      expect.anything(),
    );

    pipeline.mockClear();
    renderHook(() => useTranslate(EN_DE, true));
    expect(pipeline).toHaveBeenCalledWith(
      expect.anything(),
      EN_DE,
      true,
      expect.anything(),
    );
  });
});
