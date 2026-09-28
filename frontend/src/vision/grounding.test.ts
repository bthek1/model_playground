// The `/robotics` pair entry. It is composed from two catalogues it does not
// own, so what these tests defend is that it cannot drift from them — the
// combined `bytes` is also `combineProgress`'s denominator, and a stale
// constant there is a progress bar that is wrong with nothing failing.

import { describe, expect, it } from "vitest";

import { combineProgress, type LoadProgress } from "@/model/progress";
import {
  HEAVY_MODEL_BYTES,
  isHeavyDownload,
  LARGE_MODEL_BYTES,
  sizeEstimate,
} from "@/model/size";

import { DEPTH_MODELS } from "./depth";
import {
  DEFAULT_ROBOTICS_ENTRY,
  GROUNDING_DEPTH_ID,
  GROUNDING_DETECTOR_ID,
  GROUNDING_ENTRIES,
  GROUNDING_PAIR,
  MAX_PHRASE_WORDS,
  groundingPair,
  ROBOTICS_ENTRIES,
  sentenceLike,
} from "./grounding";
import { ZERO_SHOT_DETECTOR_MODELS } from "./zeroShotDetection";

const detector = ZERO_SHOT_DETECTOR_MODELS.find(
  (m) => m.id === GROUNDING_DETECTOR_ID,
)!;
const depth = DEPTH_MODELS.find((m) => m.id === GROUNDING_DEPTH_ID)!;

describe("the grounding pair entry", () => {
  it("names two checkpoints that already ship, and adds none", () => {
    expect(detector).toBeDefined();
    expect(depth).toBeDefined();
    expect(GROUNDING_PAIR.detector).toBe(detector);
    expect(GROUNDING_PAIR.depth).toBe(depth);
  });

  it("quotes the sum of the two source entries' measured bytes, per backend", () => {
    for (const backend of ["webgpu", "wasm"] as const) {
      expect(GROUNDING_PAIR.bytes[backend]).toBe(
        detector.bytes![backend]! + depth.bytes![backend]!,
      );
    }
  });

  it("records the figures the plan measured", () => {
    // Also written out in `grounding.ts`'s header. If a source entry is
    // re-measured, this changes with it — the assertion above is the real one.
    expect(GROUNDING_PAIR.bytes).toEqual({
      webgpu: 357_547_153,
      wasm: 182_571_555,
    });
  });

  it("sums the parameter count, for the picker's estimate line", () => {
    expect(GROUNDING_PAIR.params).toBeCloseTo(detector.params + depth.params);
  });

  it("runs only where both halves run, inheriting a restriction from either source", () => {
    const over = { id: "x", label: "x", hint: "x" };
    // Neither declares one today, so the pair declares none.
    expect(groundingPair(detector, depth, over).backends).toEqual(
      detector.backends ?? depth.backends,
    );
    expect(
      groundingPair({ ...detector, backends: ["webgpu"] }, depth, over).backends,
    ).toEqual(["webgpu"]);
    expect(
      groundingPair(
        { ...detector, backends: ["webgpu"] },
        { ...depth, backends: ["wasm"] },
        over,
      ).backends,
    ).toEqual([]);
  });

  it("carries the detector's query kind, so the route can say what to type", () => {
    expect(GROUNDING_PAIR.queries).toBe("label");
  });

  it("is large, so the picker warns — mandatory past 200 MiB", () => {
    expect(GROUNDING_PAIR.bytes.webgpu!).toBeGreaterThan(LARGE_MODEL_BYTES);
    expect(sizeEstimate(GROUNDING_PAIR.params, GROUNDING_PAIR.bytes).large).toBe(
      true,
    );
  });

  it("is not heavy, so the page must not invent a second opt-in", () => {
    // Both directions are pinned: adding the gate is as wrong as dropping the
    // warning, because a gate is a claim that the size is a decision.
    expect(GROUNDING_PAIR.bytes.webgpu!).toBeLessThan(HEAVY_MODEL_BYTES);
    expect(isHeavyDownload(GROUNDING_PAIR.bytes)).toBe(false);
  });
});

describe("the robotics picker list", () => {
  it("offers the grounding pair as the default", () => {
    expect(ROBOTICS_ENTRIES[0]).toBe(GROUNDING_PAIR);
    expect(DEFAULT_ROBOTICS_ENTRY).toBe(GROUNDING_PAIR.id);
  });

  it("discriminates entries on kind, and filters the grounding ones", () => {
    for (const e of ROBOTICS_ENTRIES) expect(e.kind).toBeDefined();
    expect(GROUNDING_ENTRIES).toContain(GROUNDING_PAIR);
    expect(GROUNDING_ENTRIES.every((e) => e.kind === "grounding")).toBe(true);
  });

  it("gives every entry a unique id that is neither half's", () => {
    const ids = ROBOTICS_ENTRIES.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
    // A composite id that equalled one half would make the cache badge claim
    // the pair was downloaded when only one model was.
    expect(ids).not.toContain(GROUNDING_DETECTOR_ID);
    expect(ids).not.toContain(GROUNDING_DEPTH_ID);
  });
});

describe("combineProgress, as the pair uses it", () => {
  const total = GROUNDING_PAIR.bytes.webgpu!;
  const part = (loaded: number, over: Partial<LoadProgress> = {}): LoadProgress => ({
    phase: "downloading",
    percent: 100,
    loaded,
    total: loaded,
    files: { done: 1, count: 1 },
    elapsedMs: 0,
    ...over,
  });

  it("does not report a finished depth model as a finished pair", () => {
    // Depth is the small half. Its own table says 100%; the pair is ~14% in.
    const bar = combineProgress([null, part(depth.bytes!.webgpu!)], total, 0)!;
    expect(bar.percent).toBeLessThan(20);
    expect(bar.total).toBe(total);
  });

  it("is monotonic as either half advances, whatever order the events arrive", () => {
    const steps: [number, number][] = [
      [0, 1e6],
      [5e7, 1e6],
      [5e7, 3e7],
      [2e8, 3e7],
      [2e8, depth.bytes!.webgpu!],
      [detector.bytes!.webgpu!, depth.bytes!.webgpu!],
    ];
    let last = -1;
    for (const [d, p] of steps) {
      const pct = combineProgress([part(d), part(p)], total, 0)!.percent!;
      expect(pct).toBeGreaterThanOrEqual(last);
      last = pct;
    }
    expect(last).toBe(100);
  });

  it("is indeterminate with no known total, and through warm-up", () => {
    expect(combineProgress([part(10)], 0, 0)!.percent).toBeNull();
    const warm = combineProgress(
      [part(1, { phase: "warmup" }), part(1, { phase: "warmup" })],
      total,
      0,
    )!;
    expect(warm.percent).toBeNull();
  });

  it("shows the least advanced half's phase — the pair waits for both", () => {
    const bar = combineProgress(
      [part(1, { phase: "warmup" }), part(1, { phase: "downloading" })],
      total,
      0,
    )!;
    expect(bar.phase).toBe("downloading");
  });
});

describe("sentenceLike", () => {
  it("flags a phrase that reads like an instruction", () => {
    expect(
      sentenceLike(["a red block", "pick up the red block nearest the camera"]),
    ).toEqual(["pick up the red block nearest the camera"]);
  });

  it(`passes noun phrases of up to ${MAX_PHRASE_WORDS} words`, () => {
    expect(sentenceLike(["a car", "the big red block", "  a   cup  "])).toEqual([]);
  });
});
