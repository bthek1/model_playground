// The auxiliary file is 124 KB of untyped float32. These tests build a
// well-formed copy from the band widths captured from the reference
// implementation, then corrupt it in each of the ways that would otherwise
// produce quietly-wrong audio.

import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/deepFilterNetFeatures.json";
import { AUX_BYTES, ERB_BANDS, erbAnalysis, erbSynthesis, parseAux } from "./aux";
import { FFT_BINS, vorbisWindow } from "./stft";

const WIDTHS = golden.widths;

/** A byte-exact stand-in for `deepfilter-auxiliary.bin`. */
function buildAux(): ArrayBuffer {
  const buf = new ArrayBuffer(AUX_BYTES);
  const all = new Float32Array(buf);
  const matrix = FFT_BINS * ERB_BANDS;
  let bin = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    for (let j = 0; j < WIDTHS[band]; j++, bin++) {
      all[bin * ERB_BANDS + band] = 1 / WIDTHS[band];
      all[matrix + band * FFT_BINS + bin] = 1;
    }
  }
  all.set(golden.window, matrix * 2);
  return buf;
}

describe("parseAux", () => {
  it("parses the three arrays at the documented offsets", () => {
    const aux = parseAux(buildAux());
    expect(aux.fwd).toHaveLength(FFT_BINS * ERB_BANDS);
    expect(aux.inv).toHaveLength(ERB_BANDS * FFT_BINS);
    expect(aux.window).toHaveLength(960);
    expect(Array.from(aux.widths)).toEqual(WIDTHS);
  });

  it("rejects a truncated file", () => {
    expect(() => parseAux(new ArrayBuffer(AUX_BYTES - 4))).toThrow(/expected/);
  });

  it("rejects a transposed inverse matrix", () => {
    // Reading `inv` as [bin][band] is the mistake the model card's own wording
    // invites; it leaves entries that are neither 0 nor 1.
    const buf = buildAux();
    const all = new Float32Array(buf);
    const matrix = FFT_BINS * ERB_BANDS;
    const inv = all.slice(matrix, matrix * 2);
    for (let band = 0; band < ERB_BANDS; band++) {
      for (let bin = 0; bin < FFT_BINS; bin++) {
        all[matrix + bin * ERB_BANDS + band] = inv[band * FFT_BINS + bin];
      }
    }
    expect(() => parseAux(buf)).toThrow(/inverse ERB/);
  });

  it("rejects a forward matrix whose bands do not average", () => {
    const buf = buildAux();
    new Float32Array(buf)[0] = 0.9;
    expect(() => parseAux(buf)).toThrow(/sums to/);
  });

  it("rejects a window that is not power-complementary", () => {
    const buf = buildAux();
    const all = new Float32Array(buf);
    const offset = FFT_BINS * ERB_BANDS * 2;
    all.set(new Float32Array(960).fill(0.5), offset);
    expect(() => parseAux(buf)).toThrow(/power-complementary/);
  });

  it("agrees with our own Vorbis window — the shipped one is the ground truth", () => {
    const ours = vorbisWindow(960);
    for (let i = 0; i < 960; i++) {
      expect(ours[i]).toBeCloseTo(golden.window[i], 6);
    }
  });
});

describe("erbAnalysis / erbSynthesis", () => {
  const { widths } = parseAux(buildAux());

  it("returns the mean power of each band", () => {
    const spec = new Float32Array(FFT_BINS * 2);
    // Band 0 is bins 0-1: powers 1 and 9 → mean 5.
    spec[0] = 1;
    spec[2] = 3;
    const bands = erbAnalysis(spec, widths);
    expect(bands[0]).toBeCloseTo(5, 6);
    expect(bands[1]).toBe(0);
  });

  it("broadcasts a gain back over every bin of its band", () => {
    const gains = new Float32Array(ERB_BANDS);
    gains[1] = 0.25;
    const bins = erbSynthesis(gains, widths);
    expect(bins[2]).toBe(0.25);
    expect(bins[3]).toBe(0.25);
    expect(bins[4]).toBe(0);
  });

  it("round-trips a flat spectrum to a flat set of bands", () => {
    const spec = new Float32Array(FFT_BINS * 2);
    for (let bin = 0; bin < FFT_BINS; bin++) spec[bin * 2] = 2;
    const bands = erbAnalysis(spec, widths);
    for (let band = 0; band < ERB_BANDS; band++) expect(bands[band]).toBeCloseTo(4, 5);
  });
});
