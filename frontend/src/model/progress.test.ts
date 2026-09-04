import { describe, expect, it } from "vitest";

import {
  initialProgress,
  reduceProgress,
  summarize,
  type ProgressState,
} from "./progress";

const MB = 1024 * 1024;

/** Fold a whole sequence, the way `useModelWorker` does event by event. */
function fold(...events: Parameters<typeof reduceProgress>[1][]): ProgressState {
  return events.reduce(reduceProgress, initialProgress);
}

describe("reduceProgress — aggregating a multi-file download", () => {
  it("sums bytes across files instead of reporting the last one", () => {
    const s = fold(
      { status: "initiate", file: "encoder.onnx", total: 100 * MB },
      { status: "initiate", file: "decoder.onnx", total: 100 * MB },
      { status: "progress", file: "encoder.onnx", loaded: 50 * MB, total: 100 * MB },
      { status: "progress", file: "decoder.onnx", loaded: 25 * MB, total: 100 * MB },
    );
    // 75 of 200 MB — not the 25% the last event alone would have shown.
    expect(s.percent).toBe(38);
    const out = summarize(s, 0);
    expect(out.loaded).toBe(75 * MB);
    expect(out.total).toBe(200 * MB);
    expect(out.files.count).toBe(2);
  });

  it("never goes backwards when a new file enlarges the denominator", () => {
    const first = fold({
      status: "progress",
      file: "a.onnx",
      loaded: 90 * MB,
      total: 100 * MB,
    });
    expect(first.percent).toBe(90);

    // A second, larger file is announced — raw arithmetic would drop to 18%.
    const second = reduceProgress(first, {
      status: "progress",
      file: "b.onnx",
      loaded: 0,
      total: 400 * MB,
    });
    expect(second.percent).toBe(90);
  });

  it("stays indeterminate while no file has announced a size", () => {
    const s = fold(
      { status: "initiate", file: "a.onnx" },
      { status: "progress", file: "a.onnx", progress: 40 },
    );
    expect(s.percent).toBeNull();
    expect(summarize(s, 0).percent).toBeNull();
  });

  it("derives bytes from a percentage when the total is already known", () => {
    const s = fold(
      { status: "initiate", file: "a.onnx", total: 200 * MB },
      { status: "progress", file: "a.onnx", progress: 25 },
    );
    expect(summarize(s, 0).loaded).toBe(50 * MB);
    expect(s.percent).toBe(25);
  });

  it("ignores an out-of-order event that would rewind a file", () => {
    const s = fold(
      { status: "progress", file: "a.onnx", loaded: 80 * MB, total: 100 * MB },
      { status: "progress", file: "a.onnx", loaded: 10 * MB, total: 100 * MB },
    );
    expect(summarize(s, 0).loaded).toBe(80 * MB);
  });

  it("counts a finished file exactly once", () => {
    const s = fold(
      { status: "progress", file: "a.onnx", loaded: 100 * MB, total: 100 * MB },
      { status: "done", file: "a.onnx" },
      { status: "done", file: "a.onnx" },
    );
    expect(summarize(s, 0).files).toEqual({ done: 1, count: 1 });
  });

  it("completes a file on `done` even with no byte events", () => {
    const s = fold(
      { status: "initiate", file: "a.onnx", total: 100 * MB },
      { status: "done", file: "a.onnx" },
    );
    expect(summarize(s, 0).loaded).toBe(100 * MB);
    expect(s.percent).toBe(100);
  });

  it("drops an event with no file to attribute it to", () => {
    expect(reduceProgress(initialProgress, { status: "progress" })).toBe(
      initialProgress,
    );
  });
});

describe("reduceProgress — phases", () => {
  it("starts connecting, moves to downloading on the first byte", () => {
    expect(initialProgress.phase).toBe("connecting");
    expect(fold({ status: "initiate", file: "a.onnx", total: 10 }).phase).toBe(
      "connecting",
    );
    expect(
      fold({ status: "progress", file: "a.onnx", loaded: 1, total: 10 }).phase,
    ).toBe("downloading");
  });

  it("reports warm-up as its own phase, with no percentage", () => {
    const s = fold(
      { status: "progress", file: "a.onnx", loaded: 100, total: 100 },
      { status: "warmup" },
    );
    expect(s.phase).toBe("warmup");
    // The bytes are all in; the bar must not sit at a misleading 100%.
    expect(summarize(s, 0).percent).toBeNull();
    expect(summarize(s, 0).loaded).toBe(100);
  });

  it("keeps the warm-up phase when a late file event arrives", () => {
    const s = fold(
      { status: "warmup" },
      { status: "done", file: "tokenizer.json" },
    );
    expect(s.phase).toBe("warmup");
  });
});

describe("summarize", () => {
  it("passes elapsed time through and names the moving file", () => {
    const s = fold({
      status: "progress",
      file: "encoder.onnx",
      loaded: 1,
      total: 2,
    });
    const out = summarize(s, 12_000);
    expect(out.elapsedMs).toBe(12_000);
    expect(out.current).toBe("encoder.onnx");
  });

  it("is safe on an empty table", () => {
    expect(summarize(initialProgress, 0)).toMatchObject({
      percent: null,
      loaded: 0,
      total: 0,
      files: { done: 0, count: 0 },
    });
  });
});
