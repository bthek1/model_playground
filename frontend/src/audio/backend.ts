// Execution-backend selection for in-browser pretrained models (Transformers.js /
// ONNX Runtime Web). This is the audio domain's counterpart to the raw-WebGPU
// runtime's capability probe (`webgpu/capabilities.ts`): WebGPU when a usable GPU
// adapter is present, otherwise the universal WASM (CPU) fallback.
//
// NOTE: this is separate from `webgpu/device.ts` on purpose — that file owns the
// hand-written WGSL runtime; this one only picks a Transformers.js `device`.

export type Backend = "webgpu" | "wasm";

/**
 * Pick the best available inference backend. Requesting an adapter (not merely
 * checking `navigator.gpu`) is the real gate — some browsers expose the API but
 * have no usable adapter. Never throws; degrades to `"wasm"`.
 */
export async function pickBackend(): Promise<Backend> {
  if (typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      /* fall through to wasm */
    }
  }
  return "wasm";
}

export interface LoadOpts {
  device: Backend;
  /** Weight precision: fp16 on GPU (matches the notebooks), quantized on CPU. */
  dtype: "fp16" | "q8";
}

/**
 * Transformers.js load options for a backend. WebGPU prefers `fp16` (half the
 * memory, matches the notebooks' `torch.float16`); WASM uses a quantized `q8` to
 * keep the download and RAM small.
 */
export function loadOpts(backend: Backend): LoadOpts {
  return backend === "webgpu"
    ? { device: "webgpu", dtype: "fp16" }
    : { device: "wasm", dtype: "q8" };
}
