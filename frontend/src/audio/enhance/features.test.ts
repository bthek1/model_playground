// Feature extraction is the step with the least documentation and the most
// silent failure modes, so these tests compare against arrays captured from the
// **official** implementation (DeepFilterNet v0.5.6 / `libDF`) rather than
// against our own expectations. The fixture holds `feat_erb` and `feat_spec`
// for a fixed 0.1 s three-tone signal; regenerate it only against libDF.

import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/deepFilterNetFeatures.json";
import { ERB_BANDS } from "./aux";
import { analyse, SPEC_SCALE, synthesise } from "./deepFilterNet";
import { DF_BINS, extractFeatures, FeatureNormalizer, NORM_ALPHA } from "./features";

const WIDTHS = Int32Array.from(golden.widths);
const WINDOW = Float32Array.from(golden.window);

/** The exact signal the fixture was generated from. */
function testSignal(): Float32Array {
  const x = new Float32Array(golden.samples);
  for (let i = 0; i < x.length; i++) {
    x[i] =
      0.5 * Math.sin((2 * Math.PI * 440 * i) / 48000) +
      0.25 * Math.sin((2 * Math.PI * 1237 * i) / 48000 + 1.0) +
      0.1 * Math.sin((2 * Math.PI * 61 * i) / 48000);
  }
  return x;
}

describe("FeatureNormalizer", () => {
  it("starts the ERB mean state on the documented -60 → -90 dB ramp", () => {
    const norm = new FeatureNormalizer();
    // Feed 0 dB in every band. After one frame the state is `alpha * init_b`,
    // so the feature reads back the init ramp scaled by alpha and /40 — which
    // pins both endpoints (-60, -90) and the linear interpolation between them.
    const feat = norm.normErb(new Float32Array(ERB_BANDS).fill(1));
    for (let b = 0; b < ERB_BANDS; b++) {
      const init = -60 + b * (-30 / (ERB_BANDS - 1));
      expect(feat[b]).toBeCloseTo((-NORM_ALPHA * init) / 40, 5);
    }
  });

  it("decays the state towards the input at alpha per frame", () => {
    const norm = new FeatureNormalizer();
    const step = () => norm.normErb(new Float32Array(ERB_BANDS).fill(1))[0];
    // Input is 0 dB; state starts at -60 dB and moves 1% of the gap per frame.
    const first = step();
    const second = step();
    expect(first * 40).toBeCloseTo(60 * (1 - NORM_ALPHA) * 0 + 60 * NORM_ALPHA, 3);
    expect(second).toBeLessThan(first); // the gap is closing
  });
});

describe("extractFeatures vs the reference implementation", () => {
  const spec = analyse(testSignal(), WINDOW);
  const feats = extractFeatures(spec, WIDTHS);

  it("produces the expected frame count", () => {
    expect(feats.frames).toBe(golden.frames);
  });

  it("matches libDF's feat_erb", () => {
    expect(feats.featErb).toHaveLength(golden.featErb.length);
    for (let i = 0; i < golden.featErb.length; i++) {
      expect(feats.featErb[i]).toBeCloseTo(golden.featErb[i], 4);
    }
  });

  it("matches libDF's feat_spec, real plane then imaginary", () => {
    const { frames } = feats;
    for (let i = 0; i < frames * DF_BINS; i++) {
      expect(feats.featSpec[i]).toBeCloseTo(golden.featSpecRe[i], 4);
      expect(feats.featSpec[frames * DF_BINS + i]).toBeCloseTo(golden.featSpecIm[i], 4);
    }
  });

  it("is NOT level-invariant — dropping SPEC_SCALE scales feat_spec by sqrt(960)", () => {
    // The unit norm divides by sqrt(state), so a global gain g on the spectrum
    // multiplies the feature by sqrt(g). This is why the analysis scaling has to
    // match libDF's `wnorm` exactly; get it wrong and the network sees a signal
    // at the wrong apparent level and masks it away as noise.
    expect(SPEC_SCALE).toBeCloseTo(1 / 960, 12);
    const unscaled = analyse(testSignal(), WINDOW);
    for (let i = 0; i < unscaled.data.length; i++) unscaled.data[i] /= SPEC_SCALE;
    const wrong = extractFeatures(unscaled, WIDTHS);
    // Roughly sqrt(960) ≈ 31, not exactly: the running state is still warming
    // up over so short a clip. An order of magnitude is the point.
    const at = Math.floor(feats.frames / 2) * DF_BINS + 10;
    expect(wrong.featSpec[at] / feats.featSpec[at]).toBeGreaterThan(20);
  });
});

describe("analyse / synthesise", () => {
  it("round-trips a signal through the scaled STFT", () => {
    const x = testSignal();
    const back = synthesise(analyse(x, WINDOW), x.length, WINDOW);
    expect(back).toHaveLength(x.length);
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(back[i] - x[i]));
    expect(worst).toBeLessThan(1e-5);
  });

  it("matches the reference's spectrum values", () => {
    const spec = analyse(testSignal(), WINDOW);
    for (let i = 0; i < golden.specHead.length; i++) {
      expect(spec.data[spec.bins * 2 + i]).toBeCloseTo(golden.specHead[i], 6);
    }
  });
});
