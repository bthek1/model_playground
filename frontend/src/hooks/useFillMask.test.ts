import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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

const { useFillMask } = await import("./useFillMask");
const { FILL_TOP_K } = await import("@/text/catalogue");

const BERT = "Xenova/bert-base-uncased";
const ROBERTA = "Xenova/roberta-base";

describe("useFillMask", () => {
  it("sends the selected model's mask literal beside the prompt", async () => {
    post.mockResolvedValueOnce({ mask: "<mask>", fills: [] });
    const hook = renderHook(() => useFillMask(ROBERTA));

    await act(async () => {
      await hook.result.current.run("The capital of France is <mask>.");
    });

    // The third argument is the mask, and it is RoBERTa's — not a literal
    // `[MASK]` and not inside the pipeline options, which never see it.
    expect(post).toHaveBeenCalledWith(
      "The capital of France is <mask>.",
      [{ top_k: FILL_TOP_K }],
      "<mask>",
    );
    expect(hook.result.current.maskToken).toBe("<mask>");
  });

  it("exposes the token the tokenizer actually used, not the one it sent", async () => {
    // The catalogue declares `[MASK]` for BERT; a repo that had changed its
    // tokenizer would come back saying otherwise, and the page must be able to
    // tell the user which one the run really used.
    post.mockResolvedValueOnce({ mask: "<mask>", fills: [] });
    const hook = renderHook(() => useFillMask(BERT));

    let mask: string | null = null;
    await act(async () => {
      mask = (await hook.result.current.run("a [MASK] b")).mask;
    });

    expect(hook.result.current.maskToken).toBe("[MASK]");
    expect(mask).toBe("<mask>");
  });

  it("strips the leading space a byte-level decode leaves on", async () => {
    post.mockResolvedValueOnce({
      mask: "<mask>",
      fills: [
        { token_str: " Paris", score: 0.67 },
        { token_str: " Lyon", score: 0.02 },
      ],
    });
    const hook = renderHook(() => useFillMask(ROBERTA));

    await act(async () => {
      await hook.result.current.run("The capital of France is <mask>.");
    });

    expect(hook.result.current.result?.fills).toEqual([
      [
        { label: "Paris", score: 0.67 },
        { label: "Lyon", score: 0.02 },
      ],
    ]);
  });

  it("normalises a single prompt to one list, not a bare list", async () => {
    post.mockResolvedValueOnce({
      mask: "[MASK]",
      fills: [{ token_str: "paris", score: 0.3 }],
    });
    const hook = renderHook(() => useFillMask(BERT));

    let out: { fills: unknown[] } = { fills: [] };
    await act(async () => {
      out = await hook.result.current.run("a [MASK] b");
    });

    // One list per prompt, whatever the pipeline's own shape was — a caller
    // that batches and one that does not must read the same thing.
    expect(out.fills).toHaveLength(1);
    expect(out.fills[0]).toEqual([{ label: "paris", score: 0.3 }]);
  });

  it("keeps a batch's lists in prompt order", async () => {
    post.mockResolvedValueOnce({
      mask: "[MASK]",
      fills: [
        [{ token_str: "lawyer", score: 0.1 }],
        [{ token_str: "nurse", score: 0.2 }],
      ],
    });
    const hook = renderHook(() => useFillMask(BERT));

    let out: { fills: { label: string }[][] } = { fills: [] };
    await act(async () => {
      out = await hook.result.current.run([
        "The man worked as a [MASK].",
        "The woman worked as a [MASK].",
      ]);
    });

    expect(out.fills.map((l) => l[0].label)).toEqual(["lawyer", "nurse"]);
  });

  it("refuses to pair a flat answer with a batched question", async () => {
    // The bias probe renders its lists side by side under the prompt that
    // produced them. A flat list where a nested one was asked for would label
    // prompt 1's answer as both columns — plausible, and wrong.
    post.mockResolvedValueOnce({
      mask: "[MASK]",
      fills: [{ token_str: "lawyer", score: 0.1 }],
    });
    const hook = renderHook(() => useFillMask(BERT));

    let out: { fills: unknown[] } = { fills: [] };
    await act(async () => {
      out = await hook.result.current.run(["a [MASK]", "b [MASK]"]);
    });

    expect(out.fills).toEqual([]);
  });

  it("drops a candidate that trims onto one already listed", async () => {
    // Two token ids can decode to the same word once the leading space goes.
    // `ScoreList` keys its rows on the label, so a duplicate would clash.
    post.mockResolvedValueOnce({
      mask: "<mask>",
      fills: [
        { token_str: " Paris", score: 0.6 },
        { token_str: "Paris", score: 0.1 },
      ],
    });
    const hook = renderHook(() => useFillMask(ROBERTA));

    await act(async () => {
      await hook.result.current.run("a <mask> b");
    });

    // The higher-scoring one survives, because the list arrives ranked.
    expect(hook.result.current.result?.fills[0]).toEqual([
      { label: "Paris", score: 0.6 },
    ]);
  });

  it("survives a worker that returns nothing usable", async () => {
    post.mockResolvedValueOnce(undefined);
    const hook = renderHook(() => useFillMask(BERT));

    let out: { mask: string | null; fills: unknown[] } = { mask: "", fills: [] };
    await act(async () => {
      out = await hook.result.current.run("a [MASK] b");
    });

    expect(out).toEqual({ mask: null, fills: [] });
  });
});
