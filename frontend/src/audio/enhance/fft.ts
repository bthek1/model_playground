// Fourier transforms for the speech-enhancement DSP. The browser ships no FFT
// primitive, and DeepFilterNet3 needs a **960-point** real transform per frame
// (docs/plans/in-progress/audio-to-audio-deepfilternet.md).
//
// 960 = 2^6 x 15 is not a power of two, and the model's 481 bins are exactly
// 960/2 + 1 — so zero-padding to 1024 is not an option: it changes the bin
// spacing (46.875 Hz instead of 50 Hz) and would feed the network features that
// do not line up with its ERB matrices. We therefore implement **Bluestein's
// algorithm** (chirp-z), which evaluates an exact DFT of arbitrary length using
// power-of-two FFTs.

/** Next power of two ≥ n. */
export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/**
 * In-place iterative Cooley-Tukey FFT over a power-of-two length.
 * Pass `inverse` for the IFFT (scaled by 1/N).
 */
export function fft(re: Float32Array, im: Float32Array, inverse = false): void {
  const n = re.length;
  if (n !== im.length) throw new Error("fft: re and im must be the same length");
  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`fft: length must be a power of two, got ${n}`);
  }

  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (sign * 2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + len / 2] = aRe - bRe;
        im[i + k + len / 2] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/** Per-length Bluestein tables. One entry (n = 960) is reused every frame. */
interface Chirp {
  n: number;
  m: number;
  /** e^{-i pi j^2 / n} for j < n. */
  wRe: Float64Array;
  wIm: Float64Array;
  /** FFT of the mirrored conjugate chirp, length m. */
  bRe: Float32Array;
  bIm: Float32Array;
}

const chirpCache = new Map<number, Chirp>();

function chirpFor(n: number): Chirp {
  const cached = chirpCache.get(n);
  if (cached) return cached;

  const m = nextPow2(2 * n - 1);
  const wRe = new Float64Array(n);
  const wIm = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    // (j*j) % (2n) keeps the angle exact for large j.
    const ang = (-Math.PI * ((j * j) % (2 * n))) / n;
    wRe[j] = Math.cos(ang);
    wIm[j] = Math.sin(ang);
  }

  const bRe = new Float32Array(m);
  const bIm = new Float32Array(m);
  bRe[0] = wRe[0];
  bIm[0] = -wIm[0];
  for (let j = 1; j < n; j++) {
    bRe[j] = bRe[m - j] = wRe[j];
    bIm[j] = bIm[m - j] = -wIm[j];
  }
  fft(bRe, bIm);

  const entry: Chirp = { n, m, wRe, wIm, bRe, bIm };
  chirpCache.set(n, entry);
  return entry;
}

/**
 * Exact DFT of any length, in place. Uses the radix-2 path when the length is a
 * power of two and Bluestein's chirp-z transform otherwise.
 */
export function dft(re: Float32Array, im: Float32Array, inverse = false): void {
  const n = re.length;
  if (n !== im.length) throw new Error("dft: re and im must be the same length");
  if (n === 0) return;
  if ((n & (n - 1)) === 0) {
    fft(re, im, inverse);
    return;
  }

  // The inverse is the forward transform of the conjugate, conjugated and scaled.
  if (inverse) for (let i = 0; i < n; i++) im[i] = -im[i];

  const { m, wRe, wIm, bRe, bIm } = chirpFor(n);
  const aRe = new Float32Array(m);
  const aIm = new Float32Array(m);
  for (let j = 0; j < n; j++) {
    aRe[j] = re[j] * wRe[j] - im[j] * wIm[j];
    aIm[j] = re[j] * wIm[j] + im[j] * wRe[j];
  }
  fft(aRe, aIm);

  for (let i = 0; i < m; i++) {
    const r = aRe[i] * bRe[i] - aIm[i] * bIm[i];
    const s = aRe[i] * bIm[i] + aIm[i] * bRe[i];
    aRe[i] = r;
    aIm[i] = s;
  }
  fft(aRe, aIm, true);

  for (let k = 0; k < n; k++) {
    const r = aRe[k] * wRe[k] - aIm[k] * wIm[k];
    const s = aRe[k] * wIm[k] + aIm[k] * wRe[k];
    re[k] = inverse ? r / n : r;
    im[k] = inverse ? -s / n : s;
  }
}

/**
 * Real-input DFT of exactly `input.length` points. Returns the first `bins`
 * complex values (default the non-redundant half, `floor(n/2) + 1`) as
 * interleaved `[re, im, …]`.
 */
export function rfft(input: Float32Array, bins?: number): Float32Array {
  const n = input.length;
  const re = Float32Array.from(input);
  const im = new Float32Array(n);
  dft(re, im);

  const count = bins ?? Math.floor(n / 2) + 1;
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count && i < n; i++) {
    out[i * 2] = re[i];
    out[i * 2 + 1] = im[i];
  }
  return out;
}

/**
 * Inverse of {@link rfft}: takes the non-redundant half as interleaved complex
 * values and reconstructs `n` real samples, mirroring the conjugate half.
 */
export function irfft(spectrum: Float32Array, n: number): Float32Array {
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  const bins = spectrum.length / 2;

  for (let i = 0; i < bins && i < n; i++) {
    re[i] = spectrum[i * 2];
    im[i] = spectrum[i * 2 + 1];
  }
  // Hermitian symmetry: X[n-k] = conj(X[k]).
  for (let k = 1; k < n - k; k++) {
    re[n - k] = re[k];
    im[n - k] = -im[k];
  }
  // A real signal of even length has a real Nyquist bin.
  if (n % 2 === 0) im[n / 2] = 0;

  dft(re, im, true);
  return re;
}
