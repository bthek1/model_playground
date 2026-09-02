import { describe, expect, it } from "vitest";

import { dft, fft, irfft, nextPow2, rfft } from "./fft";

/** O(n^2) DFT — the ground truth the fast transform must match. */
function naiveDft(re: Float32Array, im: Float32Array) {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const ang = (-2 * Math.PI * k * t) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      outRe[k] += re[t] * c - im[t] * s;
      outIm[k] += re[t] * s + im[t] * c;
    }
  }
  return { outRe, outIm };
}

function randomSignal(n: number, seed = 1): Float32Array {
  // Deterministic LCG — a flaky numeric test is worse than no test.
  let x = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    out[i] = x / 2147483648 - 0.5;
  }
  return out;
}

describe("nextPow2", () => {
  it("rounds up to a power of two", () => {
    expect(nextPow2(1)).toBe(1);
    expect(nextPow2(960)).toBe(1024);
    expect(nextPow2(1024)).toBe(1024);
  });
});

describe("fft", () => {
  it("matches a naive DFT on random input", () => {
    const n = 256;
    const re = randomSignal(n);
    const im = randomSignal(n, 7);
    const { outRe, outIm } = naiveDft(re, im);

    const fr = Float32Array.from(re);
    const fi = Float32Array.from(im);
    fft(fr, fi);

    for (let k = 0; k < n; k++) {
      expect(fr[k]).toBeCloseTo(outRe[k], 3);
      expect(fi[k]).toBeCloseTo(outIm[k], 3);
    }
  });

  it("puts a pure tone in exactly one bin pair", () => {
    const n = 64;
    const k0 = 5;
    const re = new Float32Array(n);
    for (let t = 0; t < n; t++) re[t] = Math.cos((2 * Math.PI * k0 * t) / n);
    const im = new Float32Array(n);
    fft(re, im);

    const mag = (i: number) => Math.hypot(re[i], im[i]);
    expect(mag(k0)).toBeCloseTo(n / 2, 2);
    expect(mag(n - k0)).toBeCloseTo(n / 2, 2);
    // Everything else is numerically zero.
    for (let i = 0; i < n; i++) {
      if (i !== k0 && i !== n - k0) expect(mag(i)).toBeLessThan(1e-3);
    }
  });

  it("inverts itself", () => {
    const n = 128;
    const re = randomSignal(n, 3);
    const original = Float32Array.from(re);
    const im = new Float32Array(n);

    fft(re, im);
    fft(re, im, true);

    for (let i = 0; i < n; i++) expect(re[i]).toBeCloseTo(original[i], 5);
  });

  it("rejects a non-power-of-two length", () => {
    expect(() => fft(new Float32Array(960), new Float32Array(960))).toThrow(
      /power of two/,
    );
  });

  it("rejects mismatched real and imaginary lengths", () => {
    expect(() => fft(new Float32Array(8), new Float32Array(4))).toThrow(
      /same length/,
    );
  });
});

describe("rfft / irfft", () => {
  it("agrees with the complex FFT on the retained half", () => {
    const n = 128;
    const sig = randomSignal(n, 11);
    const half = rfft(sig);

    const re = Float32Array.from(sig);
    const im = new Float32Array(n);
    fft(re, im);

    for (let k = 0; k <= n / 2; k++) {
      expect(half[k * 2]).toBeCloseTo(re[k], 3);
      expect(half[k * 2 + 1]).toBeCloseTo(im[k], 3);
    }
  });

  it("round-trips a real signal", () => {
    const n = 256;
    const sig = randomSignal(n, 5);
    const restored = irfft(rfft(sig), n);
    for (let i = 0; i < n; i++) expect(restored[i]).toBeCloseTo(sig[i], 4);
  });

  it("transforms a non-power-of-two length exactly, without padding", () => {
    // DeepFilterNet's 960-sample frame is the real case: its 481 bins are
    // exactly 960/2 + 1. Padding to 1024 would shift the bin spacing from
    // 50 Hz to 46.875 Hz and misalign every ERB band.
    const spec = rfft(new Float32Array(960));
    expect(spec.length).toBe((960 / 2 + 1) * 2);
  });

  it("matches a naive DFT at a non-power-of-two length", () => {
    const n = 60; // 2^2 * 15 — the same shape of factorisation as 960
    const re = randomSignal(n, 17);
    const im = randomSignal(n, 19);
    const { outRe, outIm } = naiveDft(re, im);

    const fr = Float32Array.from(re);
    const fi = Float32Array.from(im);
    dft(fr, fi);

    for (let k = 0; k < n; k++) {
      expect(fr[k]).toBeCloseTo(outRe[k], 3);
      expect(fi[k]).toBeCloseTo(outIm[k], 3);
    }
  });

  it("round-trips a 960-point real frame", () => {
    const x = randomSignal(960, 23);
    const back = irfft(rfft(x), 960);
    for (let i = 0; i < 960; i++) expect(back[i]).toBeCloseTo(x[i], 4);
  });

  it("returns only the requested number of bins", () => {
    const spec = rfft(randomSignal(1024), 481);
    expect(spec.length).toBe(481 * 2);
  });
});
