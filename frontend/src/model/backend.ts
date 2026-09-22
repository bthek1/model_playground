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
 * `q4f16` / `q4` are the 4-bit weight formats a generative decoder needs to fit in a
 * tab at all — see `vlmLoadOpts`.
 */
export type Dtype = "fp16" | "q8" | "fp32" | "q4f16" | "q4";

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
 * **The bug below is not ASR-specific, despite this function's name.** It was
 * characterised here first because ASR hit it first, but
 * `/document-question-answering` reproduced it exactly — same error, same
 * missing `embed_tokens.weight_merged_0_scale` — on Donut, which is not an ASR
 * model. Read it as: *any encoder-decoder whose decoder is quantized* fails to
 * open a session on the bundled WASM provider. That catalogue expresses the fix
 * per entry (`dtypes: { wasm: { encoder_model: "q8", decoder_model_merged:
 * "fp32" } }`) rather than through a second named helper; if a third family
 * arrives, generalise this function rather than copying it again.
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

/**
 * Load options for a **vision-language model** — a generative decoder with an image
 * encoder bolted on. 4-bit weights, because `loadOpts()`'s fp16 is not a viable
 * download for this family and the gap is not marginal:
 *
 *   SmolVLM-256M   189 MB at q4f16  ·  514 MB at fp16   (2.7x)
 *   Qwen3-VL-2B   1373 MB at q4f16  · 3380 MB at fp16   (2.5x)
 *
 * (Summed over every ONNX graph each repo publishes, read off the Hub. Qwen3-VL keeps
 * its weights in external `.onnx_data` files, so summing only the `.onnx` stubs
 * measures it at 1.2 **MB**.)
 *
 * On WASM the `f16` half is dropped: fp16 activations are a GPU format, so the CPU
 * path takes plain `q4`. It is a fallback that exists to be *typed*, not one to
 * recommend — an autoregressive decoder on WASM is seconds per token, which is why
 * every VLM catalogue entry declares `backends: ["webgpu"]` and lets `useBackendProbe`
 * disable the row rather than offering a page that looks broken.
 *
 * The per-family override lives here beside `asrLoadOpts` rather than as a literal in
 * the worker, for the same reason that one does: two copies of a precision decision
 * drift, and the drift is silent.
 */
export function vlmLoadOpts(backend: Backend): LoadOpts {
  return backend === "webgpu"
    ? { device: "webgpu", dtype: "q4f16" }
    : { device: "wasm", dtype: "q4" };
}

/**
 * Does this machine's GPU adapter support **f16 in shaders** (`shader-f16`)?
 *
 * A separate question from "is there an adapter", and the gap between them is not
 * hypothetical: an adapter without this feature loads `q4f16` weights perfectly
 * happily and then fails on the **first operator**, with
 *
 *   Non-zero status code returned while running Gather node …
 *   Program Gather requires f16 but the device does not support it.
 *
 * So a page that gates on `pickBackend()` alone downloads its weights, reports
 * `ready`, and then fails every single run — which is the worst of the three
 * possible outcomes, because the user paid for the download first. Measured on
 * Chromium's SwiftShader fallback (`--enable-unsafe-swiftshader`), which is
 * exactly what a CI runner with no `/dev/dri` gets.
 *
 * Never throws; a machine we cannot ask about is reported as not supporting it,
 * because the failure mode of a false positive is a wasted download and a dead
 * page, and of a false negative is a model that is merely unavailable.
 */
export async function supportsShaderF16(): Promise<boolean> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  } catch {
    return false;
  }
}
