// Super-resolution catalogue — the browser-runnable half of **Image to Image**.
//
// Image-to-image *editing* is diffusion, and diffusion is dozens of denoising
// passes over a latent: it stays on a server, and `docs/roadmaps/vision.md`
// §3.12 says why. Super-resolution is the opposite shape — **one forward pass,
// no sampling loop at all** — so it runs in a tab, and it is the half of the
// taxonomy slug this route actually delivers. The page has to say that out loud;
// the slug promises more than the model does.
//
// Sizes are read off the Hub's blob listing, not estimated:
//
//   Swin2SR classical x2   fp32 51.9 MB · fp16 30.9 MB · q8 20.5 MB

import type { VisionModel } from "./types";

export interface SuperResModel extends VisionModel {
  task: "image-to-image";
  /** The factor the model upscales by. Drives the output canvas and the tiling. */
  scale: number;
}

export const SUPER_RES_MODELS: SuperResModel[] = [
  {
    id: "Xenova/swin2SR-classical-sr-x2-64",
    label: "Swin2SR classical x2",
    hint: "12M params, 2x upscale, one pass per tile. Apache-2.0.",
    params: 12.1,
    task: "image-to-image",
    scale: 2,
    // **Precision is pinned on WASM, and this is a measurement.**
    //
    // Measured with `just fe-e2e-superres` — a 320x320 crop upscaled to 640x640
    // and scored by PSNR against the crop it was downscaled from, alongside a
    // plain bicubic resize of the same input:
    //
    //     fp32   model 27.80 dB   bicubic 27.55 dB   32.1 s/tile
    //     q8     model 27.39 dB   bicubic 27.55 dB   24.2 s/tile
    //
    // **At q8 the model is worse than not running it at all** — it loses to
    // bicubic — for a 25% saving in time and 33 MB in download. That is the
    // `onnx-community/mobilenetv4_conv_small` failure again in a different
    // guise: super-resolution is dense regression, so int8 error lands directly
    // in the picture instead of being absorbed by an argmax. Do not "optimise"
    // this away; re-measure with the spec if the export is ever rebuilt.
    //
    // fp16 on WebGPU is left at the default and is **not** covered by that
    // measurement. Swin2SR sets `img_range: 1.0`, so activations stay small and
    // half precision has room, but the spec above runs on WASM — point it at the
    // `webgpu` project before trusting the GPU path's numbers.
    dtypes: { wasm: "fp32" },
    bytes: { webgpu: 32_428_109, wasm: 54_428_699 },
  },
];

export const DEFAULT_SUPER_RES_MODEL = SUPER_RES_MODELS[0].id;

/**
 * Longest side of the **source** accepted for upscaling.
 *
 * Not a throttle like the other routes' `MAX_INFERENCE_SIDE` — it is a time and
 * memory ceiling, and the binding constraint is time. Tiles grow with the
 * *area* of the source: a 512 px photo is about 6 tiles, a 1024 px one about 35,
 * and on WASM that is the difference between fifteen seconds and a minute and a
 * half for the same demonstration. 512 -> 1024 already shows everything a
 * super-resolution model has to show.
 *
 * The memory ceiling is real too — at 2x a 1024 px source needs 16 MB of RGBA in
 * the accumulator plus the same again in float sums — but a user abandons the
 * page long before a tab starts trading.
 */
export const MAX_SOURCE_SIDE = 512;

/**
 * Roughly how long one tile takes, per backend, in milliseconds.
 *
 * Used only to quote a duration **before** the run starts — a run is many
 * inferences, and a transformer per tile on CPU is minutes, not seconds. A rough
 * number stated up front is worth far more than an exact one discovered halfway
 * through, so these are deliberately coarse and deliberately pessimistic.
 *
 * **`wasm` is measured; `webgpu` is not.** `just fe-e2e-superres` clocked 32 s
 * and 40 s per tile at fp32 across two runs in headless Chromium — the first
 * guess here was 2.5 s, out by more than a factor of ten, which had the guard
 * promising 23 s for a run that took 273. 35 s is the middle of the observed
 * range; note that a *quoted* estimate is better slightly high than slightly
 * low, since the cost of over-quoting is a pleasant surprise. The WebGPU figure
 * is still a guess — run the spec in the `webgpu` project to replace it.
 */
export const MS_PER_TILE: Record<string, number> = { webgpu: 1500, wasm: 35_000 };

/** "about 45 s" / "about 3 min" — the estimate, in words. */
export function formatDuration(ms: number): string {
  if (ms < 45_000) return `about ${Math.max(1, Math.round(ms / 1000))} s`;
  const minutes = Math.round(ms / 60_000);
  return `about ${Math.max(1, minutes)} min`;
}
