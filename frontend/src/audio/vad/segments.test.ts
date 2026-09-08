// Segment derivation, tested twice over: synthetic probability arrays for the
// boundary conditions, and a **captured fixture** for the real thing.
//
// The fixture is 343 real Silero probabilities for the JFK clip, produced by
// running `onnx-community/silero-vad` outside the test (the same tactic
// `enhance/` uses for its DSP). It is what stops this file from only ever
// asserting what the author already believed — a synthetic array can be made to
// agree with a wrong threshold rule, real speech cannot.

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/sileroJfk.json";
import { speechFraction, speechSeconds, toSegments } from "./segments";
import { FRAME_SAMPLES } from "./vad";

/** Frames at 16 kHz are 32 ms, so 1000 ms is 31.25 frames. */
const opts = { frameSamples: FRAME_SAMPLES, sampleRate: 16000 };

/** `n` frames at `p`. */
const flat = (n: number, p: number) => new Float32Array(n).fill(p);

function join(...runs: Float32Array[]): Float32Array {
  const out = new Float32Array(runs.reduce((n, r) => n + r.length, 0));
  let at = 0;
  for (const r of runs) {
    out.set(r, at);
    at += r.length;
  }
  return out;
}

describe("toSegments", () => {
  it("finds nothing in silence", () => {
    expect(toSegments(flat(100, 0.01), opts)).toEqual([]);
  });

  it("returns one segment spanning a clip that is speech throughout", () => {
    const segments = toSegments(flat(100, 0.99), opts);
    expect(segments).toHaveLength(1);
    expect(segments[0].start).toBe(0);
    // 100 frames × 32 ms — the segment covers the last frame, not just its start.
    expect(segments[0].end).toBeCloseTo(3.2, 5);
  });

  it("closes a segment that runs to the end of the clip", () => {
    // The common case for a mic take cut off mid-word, and the one a naive
    // `else`-only loop drops entirely.
    const segments = toSegments(join(flat(20, 0.01), flat(30, 0.9)), opts);
    expect(segments).toHaveLength(1);
    expect(segments[0].end).toBeCloseTo(50 * 0.032, 5);
  });

  it("bridges a dropout shorter than minSilenceMs", () => {
    // Two seconds of speech with a single 32 ms dip — a stop consonant, not a
    // pause. One segment, not two.
    const speech = join(flat(30, 0.9), flat(1, 0.1), flat(30, 0.9));
    expect(toSegments(speech, opts)).toHaveLength(1);
  });

  it("splits on a silence longer than minSilenceMs", () => {
    const speech = join(flat(30, 0.9), flat(20, 0.05), flat(30, 0.9));
    expect(toSegments(speech, opts)).toHaveLength(2);
  });

  it("drops a run shorter than minSpeechMs", () => {
    // Three frames is 96 ms — a click or a cough, under the 250 ms floor.
    expect(toSegments(join(flat(50, 0.01), flat(3, 0.99)), opts)).toEqual([]);
  });

  it("moves the boundaries when the threshold moves", () => {
    const ramp = Float32Array.from({ length: 60 }, (_, i) => i / 60);
    const low = toSegments(ramp, { ...opts, threshold: 0.2 });
    const high = toSegments(ramp, { ...opts, threshold: 0.8 });
    expect(low[0].start).toBeLessThan(high[0].start);
  });
});

describe("against real Silero output (jfk.wav, captured fixture)", () => {
  const probabilities = Float32Array.from(fixture.probabilities);

  it("matches the fixture's shape", () => {
    expect(fixture.frameSamples).toBe(FRAME_SAMPLES);
    expect(probabilities).toHaveLength(fixture.frames);
  });

  it("finds speech across most of the clip, but not all of it", () => {
    const fraction = speechFraction(probabilities);
    // The clip is continuous speech with a lead-in and a tail; anything near
    // 0 or 1 means the frame loop or the threshold rule is broken, not subtle.
    expect(fraction).toBeGreaterThan(0.5);
    expect(fraction).toBeLessThan(0.8);
  });

  it("starts after the lead-in silence, not at zero", () => {
    const segments = toSegments(probabilities, opts);
    expect(segments.length).toBeGreaterThan(0);
    expect(segments[0].start).toBeGreaterThan(0.2);
    expect(segments[0].start).toBeLessThan(0.6);
  });

  it("covers most of the clip's duration in a handful of segments", () => {
    const segments = toSegments(probabilities, opts);
    const duration = (fixture.frames * FRAME_SAMPLES) / fixture.sampleRate;
    expect(segments.length).toBeLessThanOrEqual(6);
    expect(speechSeconds(segments) / duration).toBeGreaterThan(0.6);
  });

  it("collapses towards fewer, longer segments as the threshold drops", () => {
    const strict = toSegments(probabilities, { ...opts, threshold: 0.9 });
    const loose = toSegments(probabilities, { ...opts, threshold: 0.1 });
    expect(speechSeconds(loose)).toBeGreaterThan(speechSeconds(strict));
  });
});

describe("speechFraction", () => {
  it("is 0 for an empty clip rather than NaN", () => {
    expect(speechFraction(new Float32Array(0))).toBe(0);
  });

  it("counts frames above the threshold", () => {
    expect(speechFraction(join(flat(3, 0.9), flat(1, 0.1)))).toBeCloseTo(0.75);
  });
});
