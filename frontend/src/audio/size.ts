// Size-before-load guardrail for in-browser models. Weights are downloaded to
// the user's machine and held in GPU/CPU memory for the tab's lifetime, so the
// UI tells them what a model costs *before* the download starts — a browser tab
// has a much tighter budget than the notebooks' 12 GB box.
//
// The estimate comes from the parameter count and the backend's weight
// precision: fp16 on WebGPU is 2 bytes per parameter, the quantized q8 used on
// WASM is 1 (see `loadOpts` in `backend.ts`). It ignores tokenizer/config files
// and per-backend graph overhead, so treat it as a floor, not a promise.

import type { Dtype } from "./backend";

const BYTES_PER_PARAM: Record<Dtype, number> = { fp32: 4, fp16: 2, q8: 1 };

/**
 * Warn past this download size. 200 MB is deliberately below the heaviest models
 * we ship (CLAP ≈292 MB, Whisper-base on WASM ≈221 MB) — a threshold above them
 * would never fire, and those two are exactly the downloads worth warning about
 * on a phone or a slow link.
 */
export const LARGE_MODEL_BYTES = 200 * 1024 * 1024;

/** Approximate weight bytes for `params` (in millions) at a given precision. */
export function estimateBytes(params: number, dtype: Dtype): number {
  return params * 1e6 * BYTES_PER_PARAM[dtype];
}

export function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/**
 * Measured download sizes in bytes, when the params estimate would mislead —
 * e.g. ASR on WASM, where the decoder stays fp32 (see `asrLoadOpts`) and the
 * real download is ~3x what a uniform-q8 estimate suggests.
 */
export interface MeasuredBytes {
  webgpu?: number;
  wasm?: number;
}

export interface SizeEstimate {
  /** Bytes on WebGPU (fp16 weights). */
  fp16: number;
  /** Bytes on WASM (q8 weights, unless the model overrides it). */
  q8: number;
  /** e.g. `"≈148 MB on WebGPU · 74 MB on WASM"`. */
  label: string;
  /** True when the worst-case download passes `LARGE_MODEL_BYTES`. */
  large: boolean;
}

/**
 * Describe what loading a model will cost. Both backends are quoted because the
 * picker runs before a backend is chosen. `measured` overrides the params-based
 * estimate where it would be wrong. `large` keys off the **bigger** of the two
 * downloads so the warning never under-promises — which for the ASR models is
 * the WASM side, not WebGPU.
 */
export function sizeEstimate(
  params: number,
  measured?: MeasuredBytes,
): SizeEstimate {
  const fp16 = measured?.webgpu ?? estimateBytes(params, "fp16");
  const q8 = measured?.wasm ?? estimateBytes(params, "q8");
  return {
    fp16,
    q8,
    label: `≈${formatBytes(fp16)} on WebGPU · ${formatBytes(q8)} on WASM`,
    large: Math.max(fp16, q8) > LARGE_MODEL_BYTES,
  };
}
