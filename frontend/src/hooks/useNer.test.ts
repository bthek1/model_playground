import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { EntitySpan } from "@/text/highlight";

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

const { useNer } = await import("./useNer");

describe("useNer", () => {
  it("normalises an aggregated entity, keeping the model's own offsets", async () => {
    post.mockResolvedValueOnce([
      { entity_group: "PER", score: 0.99, start: 0, end: 11, word: "Priya Raman" },
    ]);
    const hook = renderHook(() => useNer());

    let out: EntitySpan[] = [];
    await act(async () => {
      out = await hook.result.current.run("Priya Raman flew");
    });

    // The offsets are taken as given, never recomputed from `word` — that
    // recomputation is the token-rebuild bug in a different disguise.
    expect(out).toEqual([{ start: 0, end: 11, label: "PER", score: 0.99 }]);
  });

  it("strips a BIO prefix, so an unaggregated result names the same types", async () => {
    // With `aggregation_strategy` pinned this shape should not arrive — but if
    // it ever does, the page's colour slots and redact choices must still key
    // on `PER`, not on `B-PER` and `I-PER` as two separate types.
    post.mockResolvedValueOnce([
      { entity: "B-PER", score: 0.9, start: 0, end: 5 },
      { entity: "I-PER", score: 0.8, start: 6, end: 11 },
    ]);
    const hook = renderHook(() => useNer());

    let out: EntitySpan[] = [];
    await act(async () => {
      out = await hook.result.current.run("Priya Raman");
    });

    expect(out.map((s) => s.label)).toEqual([
      "PER",
      "PER",
    ]);
  });

  it("survives a model that returns nothing", async () => {
    post.mockResolvedValueOnce([]);
    const hook = renderHook(() => useNer());
    await act(async () => {
      await hook.result.current.run("nothing to see");
    });
    await waitFor(() => expect(hook.result.current.result).toEqual([]));
  });
});
