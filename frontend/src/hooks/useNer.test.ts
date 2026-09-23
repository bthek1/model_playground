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
  // **The shape the runtime actually returns**, recorded from a real 4.2.0 run:
  // no `start`, no `end`, anywhere. The pipeline ships with the work unwritten
  // and declares both fields optional, so a hook reading `e.start` type-checks
  // and receives `undefined` — every span is then dropped as invalid and the
  // page renders the user's text with nothing marked. That shipped past a green
  // unit suite once, because the mock supplied offsets the real one never does.
  it("recovers offsets the pipeline does not return", async () => {
    post.mockResolvedValueOnce([
      { entity_group: "PER", score: 0.998, word: "P" },
      { entity_group: "PER", score: 0.984, word: "##riya Raman" },
      { entity_group: "ORG", score: 0.991, word: "Siemens" },
    ]);
    const hook = renderHook(() => useNer());

    const text = "Priya Raman works at Siemens.";
    let out: EntitySpan[] = [];
    await act(async () => {
      out = await hook.result.current.run(text);
    });

    expect(out.map((s) => text.slice(s.start, s.end))).toEqual([
      "Priya Raman",
      "Siemens",
    ]);
    expect(hook.result.current.unplaced).toEqual([]);
  });

  it("takes the model's own offsets when a future runtime provides them", async () => {
    post.mockResolvedValueOnce([
      { entity_group: "PER", score: 0.99, start: 0, end: 11, word: "Priya Raman" },
    ]);
    const hook = renderHook(() => useNer());

    let out: EntitySpan[] = [];
    await act(async () => {
      out = await hook.result.current.run("Priya Raman flew");
    });

    // Taken as given, never recomputed — the recovery path is a fallback for a
    // runtime gap, not the preferred source of truth.
    expect(out).toEqual([{ start: 0, end: 11, label: "PER", score: 0.99 }]);
  });

  it("reports an entity it could not place rather than shortening the list silently", async () => {
    post.mockResolvedValueOnce([
      { entity_group: "ORG", score: 0.9, word: "Siemens" },
    ]);
    const hook = renderHook(() => useNer());

    await act(async () => {
      await hook.result.current.run("no company named here");
    });

    expect(hook.result.current.result).toEqual([]);
    expect(hook.result.current.unplaced).toHaveLength(1);
  });

  it("strips a BIO prefix, so an unaggregated result names the same types", async () => {
    // With `aggregation_strategy` pinned this shape should not arrive — but if
    // it ever does, the page's colour slots and redact choices must still key
    // on `PER`, not on `B-PER` and `I-PER` as two separate types.
    post.mockResolvedValueOnce([
      { entity: "B-PER", score: 0.9, word: "Priya" },
      { entity: "I-PER", score: 0.8, word: "Raman" },
    ]);
    const hook = renderHook(() => useNer());

    let out: EntitySpan[] = [];
    await act(async () => {
      out = await hook.result.current.run("Priya Raman");
    });

    // Two spans, not one: these are separated by a space, so they do not
    // touch. Merging across whitespace would join two genuinely separate
    // mentions that happen to be adjacent ("Berlin Munich" as one LOC), which
    // is a worse error than showing a two-word name as two marks — and with
    // `aggregation_strategy` pinned this shape does not arrive anyway.
    expect(out.map((s) => s.label)).toEqual(["PER", "PER"]);
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
