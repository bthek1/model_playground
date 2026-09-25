import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const post = vi.fn();
vi.mock("@/hooks/useTextPipeline", () => ({
  useTextPipeline: () => ({
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
    load: vi.fn(),
    retry: vi.fn(),
    cancel: vi.fn(),
  }),
}));

const { useTextEmbed } = await import("./useTextEmbed");

const MINILM = "Xenova/all-MiniLM-L6-v2"; // 384-d, mean-pooled
const BGE = "Xenova/bge-base-en-v1.5"; // 768-d, CLS-pooled
const NOMIC = "nomic-ai/nomic-embed-text-v1.5"; // needs a task prefix

/** A `[n, dim]` tensor whose row i is filled with i + 1. */
function batch(n: number, dim: number) {
  const data = new Float32Array(n * dim);
  for (let r = 0; r < n; r++) data.fill(r + 1, r * dim, (r + 1) * dim);
  return { data, dims: [n, dim] };
}

describe("useTextEmbed", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends the checkpoint's own pooling, not a constant", async () => {
    post.mockResolvedValueOnce(batch(1, 768));
    const hook = renderHook(() => useTextEmbed(BGE));

    await act(async () => {
      await hook.result.current.run("hello");
    });

    // CLS, because that is how BGE was trained. Mean-pooling it returns a
    // 768-wide vector that ranks plausibly and is wrong.
    expect(post).toHaveBeenCalledWith(["hello"], [{ pooling: "cls" }]);
  });

  it("batches a list into one call and keeps the caller's order", async () => {
    post.mockResolvedValueOnce(batch(3, 4));
    const hook = renderHook(() => useTextEmbed(MINILM));

    let out: Float32Array[] = [];
    await act(async () => {
      out = await hook.result.current.run(["a", "b", "c"]);
    });

    expect(post).toHaveBeenCalledTimes(1);
    expect(out.map((v) => v[0])).toEqual([1, 2, 3]);
  });

  it("re-embeds only the side that changed", async () => {
    post.mockResolvedValueOnce(batch(2, 4));
    const hook = renderHook(() => useTextEmbed(MINILM));

    await act(async () => {
      await hook.result.current.run(["a", "b"]);
    });
    expect(post).toHaveBeenCalledTimes(1);

    // "a" is cached; only "c" goes to the worker.
    post.mockResolvedValueOnce(batch(1, 4));
    await act(async () => {
      await hook.result.current.run(["a", "c"]);
    });
    expect(post).toHaveBeenCalledTimes(2);
    expect(post.mock.calls[1][0]).toEqual(["c"]);
  });

  it("sends nothing at all when every text is already cached", async () => {
    post.mockResolvedValueOnce(batch(1, 4));
    const hook = renderHook(() => useTextEmbed(MINILM));

    await act(async () => {
      await hook.result.current.run("a");
    });
    await act(async () => {
      await hook.result.current.run("a");
    });
    expect(post).toHaveBeenCalledTimes(1);
    expect(hook.result.current.cached).toBe(1);
  });

  it("de-duplicates repeats inside one batch", async () => {
    post.mockResolvedValueOnce(batch(2, 4));
    const hook = renderHook(() => useTextEmbed(MINILM));

    let out: Float32Array[] = [];
    await act(async () => {
      out = await hook.result.current.run(["a", "b", "a"]);
    });

    expect(post.mock.calls[0][0]).toEqual(["a", "b"]);
    // Still three vectors back, in the order asked for.
    expect(out).toHaveLength(3);
    expect(out[0]).toBe(out[2]);
  });

  // The failure this guards is a wrong *attribution*: n vectors handed back for
  // m texts silently pairs each score with the wrong sentence.
  it("throws when the batch comes back the wrong length", async () => {
    post.mockResolvedValueOnce(batch(2, 4));
    const hook = renderHook(() => useTextEmbed(MINILM));

    await expect(
      act(async () => {
        await hook.result.current.run(["a", "b", "c"]);
      }),
    ).rejects.toThrow(/asked for 3 embeddings, got 2/i);
  });

  it("clears the cache when the checkpoint changes", async () => {
    post.mockResolvedValueOnce(batch(1, 384));
    const hook = renderHook(({ id }) => useTextEmbed(id), {
      initialProps: { id: MINILM },
    });

    await act(async () => {
      await hook.result.current.run("a");
    });
    expect(hook.result.current.cached).toBe(1);

    hook.rerender({ id: BGE });
    expect(hook.result.current.cached).toBe(0);

    // And the next run genuinely re-embeds rather than serving a 384-d vector
    // for a 768-d model.
    post.mockResolvedValueOnce(batch(1, 768));
    await act(async () => {
      await hook.result.current.run("a");
    });
    expect(post).toHaveBeenCalledTimes(2);
  });

  describe("task prefixes", () => {
    it("prepends the instruction the checkpoint expects", async () => {
      post.mockResolvedValueOnce(batch(1, 768));
      const hook = renderHook(() => useTextEmbed(NOMIC));

      await act(async () => {
        await hook.result.current.run("a cat", "query");
      });
      expect(post).toHaveBeenCalledWith(
        ["search_query: a cat"],
        [{ pooling: "mean" }],
      );
    });

    // The same sentence as a query and as a document is two different vectors,
    // so sharing one cache entry between them is a wrong answer that looks like
    // a working cache.
    it("keys the cache on the composed string, not the raw text", async () => {
      post.mockResolvedValueOnce(batch(1, 768));
      const hook = renderHook(() => useTextEmbed(NOMIC));

      await act(async () => {
        await hook.result.current.run("a cat", "query");
      });
      post.mockResolvedValueOnce(batch(1, 768));
      await act(async () => {
        await hook.result.current.run("a cat", "document");
      });

      expect(post).toHaveBeenCalledTimes(2);
      expect(post.mock.calls[1][0]).toEqual(["search_document: a cat"]);
    });

    it("composes nothing for a checkpoint with no prefixes", () => {
      const hook = renderHook(() => useTextEmbed(MINILM));
      expect(hook.result.current.compose("a cat")).toBe("a cat");
    });

    it("exposes the exact string it would send", () => {
      const hook = renderHook(() => useTextEmbed(NOMIC));
      expect(hook.result.current.compose("a cat")).toBe("clustering: a cat");
      expect(hook.result.current.compose("a cat", "document")).toBe(
        "search_document: a cat",
      );
    });
  });
});
