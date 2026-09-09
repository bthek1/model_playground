// Execution-backend selection for in-browser pretrained models (Transformers.js /
// ONNX Runtime Web), for every modality: WebGPU when a usable GPU adapter is
// present, otherwise the universal WASM (CPU) fallback. The pretrained-model
// counterpart to the raw-WGSL runtime's capability probe
// (`webgpu/capabilities.ts`).
//
// It lived in `audio/` until vision arrived, which is the whole reason it moved:
// `pickBackend()` and `loadOpts()` never knew anything about audio, and a second
// copy under `vision/` would have been two probes to keep in step.
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

/**
 * Weight precision: fp16 on GPU (matches the notebooks), quantized on CPU.
 * `fp32` is the escape hatch for modules that can't be quantized — see `asrLoadOpts`.
 */
export type Dtype = "fp16" | "q8" | "fp32";

/**
 * A precision for the whole model, or one per ONNX module (`encoder_model`,
 * `decoder_model_merged`, …) when they can't share one — see `asrLoadOpts`.
 */
export type DtypeSpec = Dtype | Record<string, Dtype>;

export interface LoadOpts {
  device: Backend;
  dtype: DtypeSpec;
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

/**
 * Load options for the **ASR** models specifically. Identical to `loadOpts`
 * except on WASM, where the decoder must stay **fp32**.
 *
 * Why: the quantized (q8) Whisper/Moonshine decoders fail to even open a session
 * on the WASM execution provider bundled with `@huggingface/transformers` 4.2.0 —
 * ONNX Runtime throws `qdq_actions.cc:137 TransposeDQWeightsForMatMulNBits
 * Missing required scale: model.decoder.embed_tokens.weight_merged_0_scale`. It
 * reproduces on every ASR repo tried (`onnx-community/whisper-base`,
 * `whisper-tiny.en`, `Xenova/whisper-tiny.en`, `moonshine-tiny`) and at every
 * dtype whose decoder is quantized, so it is an ORT bug, not a bad export. The
 * **encoder** quantizes fine, so only the decoder pays full precision.
 *
 * Cost: whisper-base on WASM is ~221 MB instead of ~71 MB (see `bytes` in the
 * ASR catalogue). Worth it — the alternative is a fallback path that cannot load
 * a model at all. Revisit when the bundled ORT version updates.
 */
export function asrLoadOpts(backend: Backend): LoadOpts {
  return backend === "webgpu"
    ? { device: "webgpu", dtype: "fp16" }
    : { device: "wasm", dtype: { encoder_model: "q8", decoder_model_merged: "fp32" } };
}
