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
 * The WASM precision an **encoder-decoder** has to load at, whatever it is.
 *
 * Not a preference, and no longer a suspicion: the quantized decoder of *any*
 * seq2seq export fails to open a session on the WASM execution provider bundled
 * with `@huggingface/transformers` 4.2.0. ONNX Runtime throws
 *
 *   Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: qdq_actions.cc:137
 *   TransposeDQWeightsForMatMulNBits Missing required scale: …_merged_0_scale
 *
 * — the same error, from the same line, on five families now. And the fifth one
 * corrects what the first four suggested:
 *
 *   Whisper / Moonshine   ASR, where it was first characterised
 *   Donut                 document QA, which proved it was not ASR-specific
 *   Marian (opus-mt)      `/translation`, measured 2026-09-25
 *   BART (distilbart)     `/summarization`, measured 2026-09-25
 *   **GPT-2**             `/text-generation`, measured 2026-09-25 — and it is
 *                         **decoder-only**, so this was never a property of
 *                         encoder-decoders. Its message names
 *                         `transformer.wte.weight_merged_0_scale`, the same
 *                         embedding matrix under a different name.
 *
 * Nor is it "any model with tied embeddings": `SmolLM2-360M-Instruct` has
 * `tie_word_embeddings: true` and its `model_quantized.onnx` loads and generates
 * on WASM perfectly well (measured: 34 s to load, ~180 ms a token). **It is a
 * property of the export, not of the architecture** — the older `Xenova/*` q8
 * builds fuse the embedding matmul into `MatMulNBits` with a merged scale the
 * bundled provider cannot find, and newer first-party and `onnx-community/*`
 * exports do not. So the rule for a new entry is: try the q8 build, and if the
 * session fails naming a `*_merged_0_scale`, either pin the graph that holds the
 * embeddings to fp32 or find a newer export.
 *
 * The **per-module fallback below only exists for a multi-graph model**, because
 * it works by leaving one graph unquantized. A decoder-only model has a single
 * graph, so there is nothing to pin: GPT-2's only unquantized build is 500 MB
 * and over the size bar, which is why `/text-generation` has no GPT-2 CPU path
 * at all. So the spec stays named for the seq2seq case, which is the only one it
 * can serve, and every such catalogue entry references it:
 *
 *   dtypes: { wasm: SEQ2SEQ_WASM_DTYPES }
 *
 * The **encoder** quantizes fine, so only the decoder pays full precision. The
 * cost is real and is the reason a page can fail the feasibility bar on its
 * WASM side alone — a Marian pair is 101 MB at a uniform q8 and 271 MB like
 * this — but the alternative is a fallback path that cannot load a model at
 * all. Revisit when the bundled ORT version updates; `just fe-e2e-models`
 * checks the files, and only a real in-browser load catches the session
 * failure (the identical call loads cleanly under `onnxruntime-node`).
 */
export const SEQ2SEQ_WASM_DTYPES: DtypeSpec = {
  encoder_model: "q8",
  decoder_model_merged: "fp32",
};

/**
 * Load options for the **ASR** models specifically. Identical to `loadOpts`
 * except on WASM, where the decoder must stay **fp32**.
 *
 * **The bug it works around is not ASR-specific, despite this function's
 * name**, and it is now written up once in {@link SEQ2SEQ_WASM_DTYPES} above —
 * which this function is expressed in terms of, so the literal exists in one
 * place. ASR keeps a named helper only because it is the one family that needs
 * the whole `LoadOpts` pair rather than a per-entry `dtypes` override: its
 * catalogue predates `dtypes` and its worker asks for load options directly.
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
    : { device: "wasm", dtype: SEQ2SEQ_WASM_DTYPES };
}

/**
 * `pickBackend()`, but honest about `f16` weights.
 *
 * An adapter without the `shader-f16` feature loads `fp16`/`q4f16` weights
 * happily, reports `ready`, and then fails on the **first operator** of every
 * run — the worst outcome available, because the user pays for the download
 * first. `useBackendProbe({ requireShaderF16: true })` already answers that for
 * the *picker*; this is the same question asked inside a **worker**, at load
 * time, which is where the decision is actually made.
 *
 * Until `/text-generation` the two could not disagree, and that was luck rather
 * than design: every `q4f16` entry in the app declared `backends: ["webgpu"]`,
 * so the picker disabled the row and the worker was never asked on a machine
 * that would have got it wrong. The first entry with an f16 GPU path **and** a
 * working CPU fallback breaks that — the row is legitimately enabled, the user
 * clicks Load, and a plain `pickBackend()` sends them to a GPU that cannot run
 * the weights. Use this wherever the resolved precision may be an f16 one.
 *
 * Never throws; a machine that cannot be asked is reported as `"wasm"`, because
 * the cost of a false negative is a slower run and the cost of a false positive
 * is a dead page.
 */
export async function pickBackendForF16(): Promise<Backend> {
  const backend = await pickBackend();
  if (backend !== "webgpu") return backend;
  return (await supportsShaderF16()) ? "webgpu" : "wasm";
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
