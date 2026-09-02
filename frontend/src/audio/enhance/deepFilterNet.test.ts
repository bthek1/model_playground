// The mask/deep-filter stage. Both of its failure modes are silent — a
// one-frame offset in the filter taps sounds metallic rather than throwing, and
// filtering the already-masked spectrum just suppresses twice — so the tests
// below pin the tap alignment and the noisy-copy rule with coefficients chosen
// so the correct answer is exact.

import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/deepFilterNetFeatures.json";
import { ERB_BANDS } from "./aux";
import {
  applyEnhancement,
  DF_LOOKAHEAD,
  DF_ORDER,
  enhanceAudio,
  type DeepFilterOutputs,
} from "./deepFilterNet";
import { DF_BINS } from "./features";
import { FFT_BINS, type Stft } from "./stft";

const WIDTHS = Int32Array.from(golden.widths);
const FRAMES = 8;

/** A spectrum whose every value identifies its own frame and bin. */
function rampSpec(): Stft {
  const data = new Float32Array(FRAMES * FFT_BINS * 2);
  for (let t = 0; t < FRAMES; t++) {
    for (let bin = 0; bin < FFT_BINS; bin++) {
      data[(t * FFT_BINS + bin) * 2] = t + 1;
      data[(t * FFT_BINS + bin) * 2 + 1] = -(bin + 1);
    }
  }
  return { data, frames: FRAMES, bins: FFT_BINS };
}

/** Outputs with a flat mask and a single unit tap at filter order `tap`. */
function outputs(gain: number, tap: number): DeepFilterOutputs {
  const erbMask = new Float32Array(FRAMES * ERB_BANDS).fill(gain);
  const dfCoefs = new Float32Array(DF_ORDER * FRAMES * DF_BINS * 2);
  for (let t = 0; t < FRAMES; t++) {
    for (let f = 0; f < DF_BINS; f++) {
      dfCoefs[((tap * FRAMES + t) * DF_BINS + f) * 2] = 1;
    }
  }
  return { erbMask, dfCoefs };
}

const re = (spec: Stft, t: number, bin: number) => spec.data[(t * FFT_BINS + bin) * 2];

describe("applyEnhancement", () => {
  it("passes the signal through untouched with a unit mask and a centre tap", () => {
    const spec = rampSpec();
    applyEnhancement(spec, outputs(1, DF_ORDER - 1 - DF_LOOKAHEAD), WIDTHS);
    for (let t = 0; t < FRAMES; t++) {
      expect(re(spec, t, 0)).toBeCloseTo(t + 1, 5);
      expect(re(spec, t, FFT_BINS - 1)).toBeCloseTo(t + 1, 5);
    }
  });

  it("places the filter taps at t-2 … t+2", () => {
    // Tap 0 is the oldest frame. With df_order 5 and df_lookahead 2 it must
    // reach back exactly two frames; off by one here is the metallic-artefact
    // bug the plan warns about.
    for (let tap = 0; tap < DF_ORDER; tap++) {
      const spec = rampSpec();
      applyEnhancement(spec, outputs(1, tap), WIDTHS);
      const offset = tap - (DF_ORDER - 1 - DF_LOOKAHEAD);
      for (let t = 2; t < FRAMES - 2; t++) {
        expect(re(spec, t, 0)).toBeCloseTo(t + 1 + offset, 5);
      }
    }
  });

  it("zeroes filtered bins whose taps fall outside the clip", () => {
    const spec = rampSpec();
    applyEnhancement(spec, outputs(1, 0), WIDTHS); // reads frame t-2
    expect(re(spec, 0, 0)).toBe(0);
    expect(re(spec, 1, 0)).toBe(0);
    expect(re(spec, 2, 0)).toBeCloseTo(1, 5);
  });

  it("filters from the noisy spectrum, not the masked one", () => {
    // Bins below DF_BINS are overwritten by the filter, which reads the copy
    // taken before the mask; only bins above it keep the mask's gain. Applying
    // the mask first and then filtering *that* would halve the low bins too.
    const spec = rampSpec();
    applyEnhancement(spec, outputs(0.5, DF_ORDER - 1 - DF_LOOKAHEAD), WIDTHS);
    expect(re(spec, 3, 0)).toBeCloseTo(4, 5);
    expect(re(spec, 3, DF_BINS - 1)).toBeCloseTo(4, 5);
    expect(re(spec, 3, DF_BINS)).toBeCloseTo(2, 5);
    expect(re(spec, 3, FFT_BINS - 1)).toBeCloseTo(2, 5);
  });

  it("broadcasts each band's gain over exactly its own bins", () => {
    const spec = rampSpec();
    const out = outputs(0, DF_ORDER - 1 - DF_LOOKAHEAD);
    out.erbMask[3 * ERB_BANDS + ERB_BANDS - 1] = 1; // last band, frame 3 only
    applyEnhancement(spec, out, WIDTHS);
    const lastBandStart = FFT_BINS - WIDTHS[ERB_BANDS - 1];
    expect(re(spec, 3, lastBandStart)).toBeCloseTo(4, 5);
    expect(re(spec, 3, lastBandStart - 1)).toBe(0);
    expect(re(spec, 2, lastBandStart)).toBe(0);
  });
});

describe("enhanceAudio", () => {
  const aux = {
    fwd: new Float32Array(0),
    inv: new Float32Array(0),
    window: Float32Array.from(golden.window),
    widths: WIDTHS,
  };

  it("returns audio of the same length, unchanged by an identity model", async () => {
    const x = new Float32Array(4800);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * 300 * i) / 48000) * 0.4;
    const out = await enhanceAudio(x, aux, async (_erb, _spec, frames) => ({
      erbMask: new Float32Array(frames * ERB_BANDS).fill(1),
      dfCoefs: identityCoefs(frames),
    }));
    expect(out).toHaveLength(x.length);
    // Edge frames lose their taps, so compare away from the very start.
    for (let i = 1440; i < x.length - 1440; i++) expect(out[i]).toBeCloseTo(x[i], 4);
  });

  it("hands the model the tensor shapes its contract declares", async () => {
    const x = new Float32Array(4800);
    let seen: [number, number, number] | null = null;
    await enhanceAudio(x, aux, async (erb, spec, frames) => {
      seen = [erb.length, spec.length, frames];
      return {
        erbMask: new Float32Array(frames * ERB_BANDS),
        dfCoefs: new Float32Array(DF_ORDER * frames * DF_BINS * 2),
      };
    });
    expect(seen).toEqual([golden.frames * ERB_BANDS, 2 * golden.frames * DF_BINS, golden.frames]);
  });

  it("short-circuits on empty input", async () => {
    const out = await enhanceAudio(new Float32Array(0), aux, async () => {
      throw new Error("should not run the model");
    });
    expect(out).toHaveLength(0);
  });
});

function identityCoefs(frames: number): Float32Array {
  const coefs = new Float32Array(DF_ORDER * frames * DF_BINS * 2);
  const tap = DF_ORDER - 1 - DF_LOOKAHEAD;
  for (let t = 0; t < frames; t++) {
    for (let f = 0; f < DF_BINS; f++) coefs[((tap * frames + t) * DF_BINS + f) * 2] = 1;
  }
  return coefs;
}
