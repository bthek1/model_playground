import { describe, expect, it } from "vitest";

import {
  combineProgress,
  initialProgress,
  reduceProgress,
  summarize,
  type LoadProgress,
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

describe("two models loading into one bar", () => {
  // `/pose` is the one route that holds two models live, and both repos publish
  // a file called `onnx/model_fp16.onnx`. Keyed on the file name alone the
  // second model's bytes would overwrite the first's: the denominator would be
  // one model's size, the bar would hit 100% halfway through, and the second
  // download would read as a stall.
  const event = (name: string, loaded: number, total: number) => ({
    status: "progress",
    name,
    file: "onnx/model_fp16.onnx",
    loaded,
    total,
  });

  it("counts both repos' identically-named files separately", () => {
    // Both downloads are started together (`Promise.all` in the pose engine),
    // so both announce their size before either has finished.
    let state = initialProgress;
    state = reduceProgress(state, event("onnx-community/dfine_n_coco-ONNX", 0, 8_000_000));
    state = reduceProgress(state, event("onnx-community/vitpose-base-simple", 0, 172_000_000));
    state = reduceProgress(state, event("onnx-community/dfine_n_coco-ONNX", 8_000_000, 8_000_000));

    const out = summarize(state, 0);
    expect(out.files.count).toBe(2);
    // The sum of both models, not whichever one wrote to the table last.
    expect(out.total).toBe(180_000_000);
    // Finishing the small model is not "nearly done" — it is 4% of the pair.
    expect(out.percent).toBeLessThan(10);
    expect(out.files.done).toBe(1);
  });

  it("still reports the plain file name in the detail line", () => {
    const state = reduceProgress(
      initialProgress,
      event("onnx-community/vitpose-base-simple", 1, 2),
    );
    expect(summarize(state, 0).current).toBe("onnx/model_fp16.onnx");
  });
});

// The `/pose` regression, re-pinned for the second route to hold two models
// live — and it turns out to hold for a *different reason* there, which #44's
// plan was right to ask about rather than assume.
//
// `/pose` loads its pair inside one worker, so one repo-keyed table covers both
// and the assertions above apply directly. `/text-ranking` loads its pair in a
// worker each, so there are two tables and nothing to key together: keeping the
// repo in the key does not help, because the two states never meet.
describe("aggregate progress across a ranking pair", () => {
  const event = (name: string, loaded: number, total: number) => ({
    status: "progress",
    name,
    // Both of these repos publish a file at exactly this path, which is the
    // hazard when they share a table.
    file: "onnx/model_quantized.onnx",
    loaded,
    total,
  });

  it("keeps two rows for two repos publishing the same filename", () => {
    let state = reduceProgress(
      initialProgress,
      event("Xenova/all-MiniLM-L6-v2", 0, 22_972_370),
    );
    state = reduceProgress(
      state,
      event("Xenova/ms-marco-MiniLM-L-6-v2", 0, 23_143_499),
    );

    const out = summarize(state, 0);
    expect(out.files.count, "two files, not one overwritten").toBe(2);
    expect(out.total).toBe(46_115_869);
  });

  // The per-worker bar's own monotonic clamp is what makes this fail: the first
  // model finishing sets `percent` to 100, and a later-announced file enlarges
  // the denominator without being allowed to move the bar back. Within one
  // model that clamp is right; across a pair it is exactly the "100% halfway
  // through" failure.
  it("shows why a stored monotonic percent cannot span two models", () => {
    let state = reduceProgress(
      initialProgress,
      event("Xenova/all-MiniLM-L6-v2", 22_972_370, 22_972_370),
    );
    state = reduceProgress(
      state,
      event("Xenova/ms-marco-MiniLM-L-6-v2", 0, 23_143_499),
    );

    const out = summarize(state, 0);
    // Not a bug in `reduceProgress` — a limit of what one clamped percent can
    // mean. `combineProgress` below is the answer.
    expect(out.percent).toBe(100);
    expect(out.files.done).toBe(1);
  });
});

describe("combineProgress", () => {
  const PAIR_BYTES = 46_115_869; // MiniLM 22_972_370 + ms-marco 23_143_499

  const bar = (over: Partial<LoadProgress> = {}): LoadProgress => ({
    phase: "downloading",
    percent: 0,
    loaded: 0,
    total: 0,
    files: { done: 0, count: 1 },
    elapsedMs: 0,
    ...over,
  });

  // The failure this exists for: one model finished, the other untouched, must
  // read about half rather than 100%.
  it("does not report a half-downloaded pair as finished", () => {
    const out = combineProgress(
      [
        bar({ percent: 100, loaded: 22_972_370, files: { done: 1, count: 1 } }),
        bar({ percent: 0, loaded: 0 }),
      ],
      PAIR_BYTES,
      0,
    )!;
    expect(out.percent).toBe(50);
    expect(out.files).toEqual({ done: 1, count: 2 });
    expect(out.total).toBe(PAIR_BYTES);
  });

  // Monotonic **by construction**: the denominator is the catalogue's constant,
  // so a newly-announced file cannot move the bar backwards the way a summed
  // estimate would.
  it("is monotonic because the denominator is a constant", () => {
    const at = (a: number, b: number) =>
      combineProgress([bar({ loaded: a }), bar({ loaded: b })], PAIR_BYTES, 0)!
        .percent!;
    expect(at(0, 0)).toBe(0);
    expect(at(10e6, 0)).toBeGreaterThan(at(0, 0));
    expect(at(10e6, 10e6)).toBeGreaterThan(at(10e6, 0));
    expect(at(22_972_370, 23_143_499)).toBe(100);
  });

  // Measured `bytes` counts ONNX graphs only, so the tokenizer and config files
  // push `loaded` past it near the end. 100% a moment early beats a bar that
  // appears to stall.
  it("clamps at 100 rather than overshooting", () => {
    const out = combineProgress(
      [bar({ loaded: PAIR_BYTES }), bar({ loaded: 2e6 })],
      PAIR_BYTES,
      0,
    )!;
    expect(out.percent).toBe(100);
  });

  it("reports the least advanced phase, since the pair is not ready until both are", () => {
    expect(
      combineProgress(
        [bar({ phase: "warmup" }), bar({ phase: "downloading" })],
        PAIR_BYTES,
        0,
      )!.phase,
    ).toBe("downloading");
    expect(
      combineProgress(
        [bar({ phase: "warmup" }), bar({ phase: "connecting" })],
        PAIR_BYTES,
        0,
      )!.phase,
    ).toBe("connecting");
    // Both warming up is indeterminate, as it is for a single model.
    const both = combineProgress(
      [bar({ phase: "warmup" }), bar({ phase: "warmup" })],
      PAIR_BYTES,
      0,
    )!;
    expect(both.phase).toBe("warmup");
    expect(both.percent).toBeNull();
  });

  it("returns null when neither half has started, and ignores a null half", () => {
    expect(combineProgress([null, null], PAIR_BYTES, 0)).toBeNull();
    const one = combineProgress([bar({ loaded: 5e6 }), null], PAIR_BYTES, 0)!;
    expect(one.loaded).toBe(5e6);
    expect(one.files.count).toBe(1);
  });

  it("is indeterminate when the catalogue quotes no size", () => {
    expect(combineProgress([bar({ loaded: 1 })], 0, 0)!.percent).toBeNull();
  });
});
