// `useRanking` — the hook that holds `/text-ranking`'s cost model, and the one
// place in the repo where two models are live in **two** workers.
//
// What these tests are actually defending:
//
//   * **`ready` means both.** A page that answered a query with half its
//     pipeline loaded would render three empty columns and one full one, which
//     reads as a broken retriever rather than an unloaded reranker.
//   * **One bar, from `combineProgress`.** Two workers keep two progress
//     tables, so showing either fills to 100% and restarts — the "100% halfway
//     through" failure `/pose` documents, reached from the other direction.
//   * **The query and the documents are embedded under different prefixes.**
//     The same sentence as a query and as a document is two different vectors,
//     which is why the cache is keyed on the composed string. Embedding the
//     query with the document prefix still returns a rankable vector.
//   * **Reranking is one pass per candidate, sequentially, over a shortlist.**
//     A cross-encoder scores a *pair*, so nothing can be precomputed; running
//     it over the corpus instead of the shortlist is the mistake the whole
//     page exists to make visible.

import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const embedRun = vi.fn();
const embedLoad = vi.fn();
const embedRetry = vi.fn();
const embedCancel = vi.fn();
const clearCache = vi.fn();
const crossRun = vi.fn();
const crossLoad = vi.fn();
const crossRetry = vi.fn();
const crossCancel = vi.fn();

let embedState: Record<string, unknown> = {};
let crossState: Record<string, unknown> = {};

function half(over: Record<string, unknown>) {
  return {
    status: "idle",
    idle: true,
    loading: false,
    ready: false,
    progress: null,
    loadProgress: null,
    loadedInMs: null,
    backend: null,
    running: false,
    error: null,
    ...over,
  };
}

vi.mock("@/hooks/useTextEmbed", () => ({
  useTextEmbed: () => ({
    ...half(embedState),
    run: embedRun,
    load: embedLoad,
    retry: embedRetry,
    cancel: embedCancel,
    cached: (embedState.cached as number) ?? 0,
    clearCache,
    compose: (t: string) => t,
    meta: { id: "Xenova/all-MiniLM-L6-v2", dim: 384 },
  }),
}));

vi.mock("@/hooks/useTextPipeline", () => ({
  useTextPipeline: () => ({
    ...half(crossState),
    run: crossRun,
    load: crossLoad,
    retry: crossRetry,
    cancel: crossCancel,
  }),
}));

vi.mock("@/model/backend", async (orig) => ({
  ...(await orig<Record<string, unknown>>()),
  pickBackend: vi.fn(async () => "wasm"),
}));

const { useRanking } = await import("./useRanking");
const { RANKING_PAIRS, DEFAULT_RANKING_PAIR, RERANK_TOP_K } = await import(
  "@/text/catalogue"
);

/** A unit vector pointing along axis `i` of a 4-d space. */
function axis(i: number): Float32Array {
  const v = new Float32Array(4);
  v[i] = 1;
  return v;
}

const CORPUS = ["alpha doc", "beta doc", "gamma doc"];

function ready() {
  embedState = { status: "ready", idle: false, ready: true, backend: "wasm" };
  crossState = { status: "ready", idle: false, ready: true, backend: "wasm" };
}

describe("useRanking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    embedState = {};
    crossState = {};
    embedRun.mockResolvedValue([axis(0)]);
    crossRun.mockResolvedValue([{ label: "LABEL_1", score: 0.5 }]);
  });

  describe("the pair is ready only when both halves are", () => {
    it("is idle when neither has loaded", () => {
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.status).toBe("idle");
      expect(hook.result.current.ready).toBe(false);
    });

    it("is still loading when only the embedder is ready", () => {
      embedState = { status: "ready", idle: false, ready: true };
      crossState = { status: "loading", idle: false, loading: true };
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.status).toBe("loading");
      expect(hook.result.current.ready).toBe(false);
    });

    it("is not ready when one half is done and the other has not started", () => {
      embedState = { status: "ready", idle: false, ready: true };
      const hook = renderHook(() => useRanking());
      // Neither `ready` nor `loading` — and specifically not `ready`, which is
      // what gates the search trigger.
      expect(hook.result.current.ready).toBe(false);
    });

    it("is ready only once both are", () => {
      ready();
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.status).toBe("ready");
      expect(hook.result.current.ready).toBe(true);
    });

    it("reports an error from either half, even while the other loads", () => {
      embedState = { status: "loading", idle: false, loading: true };
      crossState = { status: "error", idle: false, error: "reranker 404" };
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.status).toBe("error");
      expect(hook.result.current.error).toBe("reranker 404");
    });
  });

  describe("loading the pair", () => {
    it("loads both halves from one press", () => {
      const hook = renderHook(() => useRanking());
      act(() => hook.result.current.load());
      expect(embedLoad).toHaveBeenCalledTimes(1);
      expect(crossLoad).toHaveBeenCalledTimes(1);
    });

    it("retries only the half that failed", () => {
      embedState = { status: "ready", idle: false, ready: true };
      crossState = { status: "error", idle: false, error: "boom" };
      const hook = renderHook(() => useRanking());

      act(() => hook.result.current.retry());

      // Retrying the healthy half would re-download a model that is already in
      // memory, which on this page is up to 218 MB.
      expect(crossRetry).toHaveBeenCalledTimes(1);
      expect(embedRetry).not.toHaveBeenCalled();
    });

    it("cancels both", () => {
      const hook = renderHook(() => useRanking());
      act(() => hook.result.current.cancel());
      expect(embedCancel).toHaveBeenCalledTimes(1);
      expect(crossCancel).toHaveBeenCalledTimes(1);
    });

    it("does not load on mount", () => {
      renderHook(() => useRanking());
      expect(embedLoad).not.toHaveBeenCalled();
      expect(crossLoad).not.toHaveBeenCalled();
    });
  });

  describe("one bar across two workers", () => {
    it("does not report a half-downloaded pair as finished", async () => {
      const pair = RANKING_PAIRS[0];
      // The embedder is done; the reranker has not started. Reporting the
      // embedder's own table here would show 100%.
      embedState = {
        status: "loading",
        idle: false,
        loading: true,
        loadProgress: {
          percent: 100,
          loaded: pair.embedder.bytes.wasm,
          total: pair.embedder.bytes.wasm,
          files: { done: 2, count: 2 },
          phase: "downloading",
          elapsedMs: 900,
        },
      };
      crossState = { status: "loading", idle: false, loading: true };

      const hook = renderHook(() => useRanking(pair.id));
      await act(async () => {});

      const bar = hook.result.current.loadProgress;
      expect(bar).not.toBeNull();
      expect(bar!.percent).toBeLessThan(100);
      // Roughly the embedder's share of the pair, which for the default pair is
      // about half.
      expect(bar!.percent).toBeGreaterThan(0);
    });

    it("has no bar before either half starts", () => {
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.loadProgress).toBeNull();
    });

    it("reports loadedInMs only when both halves have finished", () => {
      embedState = { status: "ready", idle: false, ready: true, loadedInMs: 800 };
      crossState = { status: "ready", idle: false, ready: true };
      let hook = renderHook(() => useRanking());
      expect(hook.result.current.loadedInMs).toBeNull();

      crossState = {
        status: "ready",
        idle: false,
        ready: true,
        loadedInMs: 1500,
      };
      hook = renderHook(() => useRanking());
      // The slower half is the pair's load time; the faster one finished while
      // the page was still unusable.
      expect(hook.result.current.loadedInMs).toBe(1500);
    });
  });

  describe("embedding the corpus", () => {
    it("embeds every document under the document prefix", async () => {
      ready();
      embedRun.mockResolvedValue(CORPUS.map((_, i) => axis(i)));
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.embedCorpus(CORPUS);
      });

      expect(embedRun).toHaveBeenCalledWith(CORPUS, "document");
    });

    it("skips blank lines rather than embedding empty strings", async () => {
      ready();
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.embedCorpus(["a", "   ", "", "b"]);
      });

      expect(embedRun).toHaveBeenCalledWith(["a", "b"], "document");
    });

    it("asks for nothing at all for an all-blank corpus", async () => {
      ready();
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.embedCorpus(["", "  "]);
      });

      expect(embedRun).not.toHaveBeenCalled();
    });

    it("chunks a large corpus instead of sending one giant batch", async () => {
      ready();
      const big = Array.from({ length: 40 }, (_, i) => `doc ${i}`);
      embedRun.mockImplementation(async (texts: string[]) =>
        texts.map(() => axis(0)),
      );
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.embedCorpus(big);
      });

      // Several calls, and every chunk within the cap: padding to the longest
      // line in a 40-line batch costs more than the batching saves.
      expect(embedRun.mock.calls.length).toBeGreaterThan(1);
      for (const [texts] of embedRun.mock.calls) {
        expect((texts as string[]).length).toBeLessThanOrEqual(16);
      }
      // Every document, exactly once, in order.
      expect(embedRun.mock.calls.flatMap(([t]) => t as string[])).toEqual(big);
    });

    it("clears the progress readout when it finishes, and when it throws", async () => {
      ready();
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.embedCorpus(CORPUS);
      });
      expect(hook.result.current.embedding).toBeNull();

      embedRun.mockRejectedValueOnce(new Error("session gone"));
      await act(async () => {
        await expect(
          hook.result.current.embedCorpus(CORPUS),
        ).rejects.toThrow("session gone");
      });
      // A stuck "embedding 16 of 40" is indistinguishable from a hang.
      expect(hook.result.current.embedding).toBeNull();
    });
  });

  describe("dense search", () => {
    it("embeds the query as a query and scores it against the corpus", async () => {
      ready();
      embedRun
        .mockResolvedValueOnce([axis(1)]) // the query
        .mockResolvedValueOnce([axis(0), axis(1), axis(2)]); // the corpus
      const hook = renderHook(() => useRanking());

      let out: { doc: number; score: number }[] = [];
      await act(async () => {
        out = await hook.result.current.denseSearch("beta", CORPUS);
      });

      expect(embedRun).toHaveBeenNthCalledWith(1, "beta", "query");
      expect(embedRun).toHaveBeenNthCalledWith(2, CORPUS, "document");
      // Document 1 is the one the query points at.
      expect(out[0]).toEqual({ doc: 1, score: 1 });
      expect(out.map((r) => r.doc)).toHaveLength(CORPUS.length);
    });

    it("sorts by score descending and breaks ties by document order", async () => {
      ready();
      embedRun
        .mockResolvedValueOnce([axis(0)])
        .mockResolvedValueOnce([axis(0), axis(1), axis(0)]);
      const hook = renderHook(() => useRanking());

      let out: { doc: number; score: number }[] = [];
      await act(async () => {
        out = await hook.result.current.denseSearch("alpha", CORPUS);
      });

      // Docs 0 and 2 both score 1; the lower index comes first, so a re-score
      // does not reshuffle a list the user is reading.
      expect(out.map((r) => r.doc)).toEqual([0, 2, 1]);
    });
  });

  describe("reranking", () => {
    it("scores the query and the document together, as a pair", async () => {
      ready();
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.rerank("q", CORPUS, [2, 0]);
      });

      // `{ text, text_pair }` — not a concatenation, which is what makes a
      // cross-encoder impossible to precompute and the reason it reranks a
      // shortlist.
      expect(crossRun).toHaveBeenCalledWith({ text: "q", text_pair: CORPUS[2] });
      expect(crossRun).toHaveBeenCalledWith({ text: "q", text_pair: CORPUS[0] });
      expect(crossRun).toHaveBeenCalledTimes(2);
    });

    it("runs one pass per candidate, never overlapping", async () => {
      ready();
      let inflight = 0;
      let peak = 0;
      crossRun.mockImplementation(async () => {
        peak = Math.max(peak, ++inflight);
        await Promise.resolve();
        inflight--;
        return [{ score: 0.5 }];
      });
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.rerank("q", CORPUS, [0, 1, 2]);
      });

      expect(peak).toBe(1);
    });

    it("caps the shortlist, so reranking can never become a corpus scan", async () => {
      ready();
      const big = Array.from({ length: 60 }, (_, i) => `doc ${i}`);
      const hook = renderHook(() => useRanking());

      await act(async () => {
        await hook.result.current.rerank(
          "q",
          big,
          big.map((_, i) => i),
        );
      });

      // The cap is the page's whole argument: one pass per candidate is a cost
      // a browser feels, so it is applied to a shortlist.
      expect(crossRun).toHaveBeenCalledTimes(RERANK_TOP_K);
      expect(RERANK_TOP_K).toBeLessThan(big.length);
    });

    it("re-sorts by the cross-encoder's score, not the shortlist's order", async () => {
      ready();
      crossRun
        .mockResolvedValueOnce([{ score: 0.1 }])
        .mockResolvedValueOnce([{ score: 0.9 }]);
      const hook = renderHook(() => useRanking());

      let out: { doc: number; score: number }[] = [];
      await act(async () => {
        out = await hook.result.current.rerank("q", CORPUS, [0, 1]);
      });

      // If the rerank returned the shortlist unchanged the page would render
      // the dense column twice, which is its most plausible bug.
      expect(out).toEqual([
        { doc: 1, score: 0.9 },
        { doc: 0, score: 0.1 },
      ]);
    });

    it("reads a bare object, and scores a malformed result 0 rather than NaN", async () => {
      ready();
      crossRun.mockResolvedValueOnce({ score: 0.7 }).mockResolvedValueOnce([{}]);
      const hook = renderHook(() => useRanking());

      let out: { doc: number; score: number }[] = [];
      await act(async () => {
        out = await hook.result.current.rerank("q", CORPUS, [0, 1]);
      });

      expect(out).toEqual([
        { doc: 0, score: 0.7 },
        { doc: 1, score: 0 },
      ]);
      expect(out.every((r) => !Number.isNaN(r.score))).toBe(true);
    });
  });

  describe("the pair and its cache", () => {
    it("drops every vector on request, which is the only answer to a pair change", () => {
      const hook = renderHook(() => useRanking());
      act(() => hook.result.current.clear());
      // Vectors from another embedder are a different space: comparing them by
      // cosine returns a plausible number that means nothing.
      expect(clearCache).toHaveBeenCalledTimes(1);
    });

    it("exposes how many documents currently have a vector", () => {
      embedState = { status: "ready", idle: false, ready: true, cached: 12 };
      crossState = { status: "ready", idle: false, ready: true };
      const hook = renderHook(() => useRanking());
      expect(hook.result.current.embedded).toBe(12);
    });

    it("falls back to the first pair for an unknown id", () => {
      const hook = renderHook(() => useRanking("minilm+nonesuch"));
      expect(hook.result.current.pair.id).toBe(DEFAULT_RANKING_PAIR);
    });

    it("measures both backends for the pair and for each half", () => {
      for (const pair of RANKING_PAIRS) {
        for (const backend of ["webgpu", "wasm"] as const) {
          // `MeasuredBytes` makes both fields optional, so an entry that
          // quotes only one backend type-checks. The bar's denominator would
          // then be 0 on the other one and the percent indeterminate for the
          // whole load.
          expect(typeof pair.bytes[backend]).toBe("number");
          expect(typeof pair.embedder.bytes[backend]).toBe("number");
          expect(typeof pair.reranker.bytes[backend]).toBe("number");
        }
      }
    });

    it("quotes a combined size larger than either half alone", () => {
      for (const pair of RANKING_PAIRS) {
        for (const backend of ["webgpu", "wasm"] as const) {
          const combined = pair.bytes[backend] ?? 0;
          // The denominator of the one bar. Quoting a single half's size makes
          // the bar stop at ~50% or overshoot, depending which half.
          expect(combined).toBeGreaterThan(pair.embedder.bytes[backend] ?? 0);
          expect(combined).toBeGreaterThan(pair.reranker.bytes[backend] ?? 0);
        }
      }
    });
  });
});
