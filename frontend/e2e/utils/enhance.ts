import type { Page } from "@playwright/test";

export interface EnhanceMeasurement {
  /** Scale-invariant SDR of the noisy input against the clean reference, in dB. */
  before: number;
  /** Scale-invariant SDR of the enhanced output, in dB. */
  after: number;
  samples: number;
}

/**
 * Run the real enhancement pipeline inside the page on a synthetic noisy clip
 * and measure the result.
 *
 * It drives the modules directly rather than the UI because the assertion is
 * numeric: the route offers no way to read samples back out, and "a waveform
 * appeared" is exactly the check that cannot tell good audio from metallic.
 * Shared by the WASM spec (`audio-models.spec.ts`) and the WebGPU one
 * (`webgpu/enhance.spec.ts`) so both measure the same thing the same way.
 */
export function measureEnhancement(
  page: Page,
  backend: "webgpu" | "wasm",
): Promise<EnhanceMeasurement> {
  return page.evaluate(async (device: "webgpu" | "wasm") => {
    // Dev-server URLs, resolved by the browser at runtime — Vite serves the
    // app's own TypeScript. They go through variables so TypeScript treats them
    // as dynamic specifiers: from `e2e/` these paths do not resolve statically,
    // and `tsc -b` would fail on the literals.
    const app = (path: string) => import(path);
    const [{ loadDeepFilterNet }, { enhanceAudio }, { decodeToMono }] =
      await Promise.all([
        app("/src/audio/enhance/session.ts"),
        app("/src/audio/enhance/deepFilterNet.ts"),
        app("/src/audio/io.ts"),
      ]);

    // Real speech, decoded at DeepFilterNet's native 48 kHz.
    const url =
      "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
    const clean = await decodeToMono(await (await fetch(url)).arrayBuffer(), 48000);

    // Additive white noise at roughly 0 dB SNR — a level DeepFilterNet should
    // cut decisively. A deterministic LCG keeps the run reproducible.
    let seed = 12345;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    let power = 0;
    for (const v of clean) power += v * v;
    const rms = Math.sqrt(power / clean.length);
    const noisy = new Float32Array(clean.length);
    for (let i = 0; i < clean.length; i++) noisy[i] = clean[i] + rand() * rms * 3.46;

    const session = await loadDeepFilterNet(
      "soniqo/DeepFilterNet3-ONNX",
      undefined,
      device,
    );
    const enhanced = await enhanceAudio(noisy.slice(), session.aux, session.infer);
    await session.dispose();

    const sisdr = (est: Float32Array) => {
      let dot = 0;
      let ref = 0;
      for (let i = 0; i < clean.length; i++) {
        dot += est[i] * clean[i];
        ref += clean[i] * clean[i];
      }
      const a = dot / ref;
      let signal = 0;
      let error = 0;
      for (let i = 0; i < clean.length; i++) {
        const t = a * clean[i];
        signal += t * t;
        error += (est[i] - t) * (est[i] - t);
      }
      return 10 * Math.log10(signal / error);
    };

    return { before: sisdr(noisy), after: sisdr(enhanced), samples: enhanced.length };
  }, backend);
}

/**
 * What a correct pipeline must achieve. A broken one (wrong spectrum scaling,
 * wrong deep-filter offset) still returns plausible audio, but scores at or
 * below its own input — so the *margin*, not the sign, is the assertion.
 */
export const MIN_SDR_GAIN_DB = 6;
