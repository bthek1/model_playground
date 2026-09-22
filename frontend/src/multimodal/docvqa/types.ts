// Document Question Answering: a photographed document and a question in, the
// answer extracted from it out. Roadmap §3.3.
//
// **The one route in this category that rides a real pipeline.** 4.2.0's
// `SUPPORTED_TASKS` carries `document-question-answering`, its registry
// (`MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES`) maps exactly one entry
// — `vision-encoder-decoder → VisionEncoderDecoderModel`, which is Donut's model
// type — and `Xenova/donut-base-finetuned-docvqa` is the pipeline's own default
// model. Checked before planning rather than after the first load failed, which
// is the step `/image-text-to-text`, `/image-to-text` and `/text-to-audio` each
// learned the hard way (see `multimodal/types.ts`).
//
// So there is no engine here beyond the generic one: this is `adding-a-model.md`
// §8, a plain `pipeline()` call.
//
// **It is one checkpoint, not a catalogue, and that is a property of the
// pipeline rather than a shortage of models.** `DocumentQuestionAnsweringPipeline`
// hardcodes Donut's prompt format:
//
//   `<s_docvqa><s_question>${question}</s_question><s_answer>`
//
// A different architecture handed to this pipeline would be prompted with tokens
// it has never seen — the Florence-2 failure mode again, fluent and unrelated.
// The alternatives are unavailable anyway: `impira/layoutlm-document-qa` needs
// OCR boxes as *input*, which the browser would have to produce first, and
// `stepfun-ai/GOT-OCR-2.0-hf` has no export (Florence-2 on `/image-to-text`
// already covers plain OCR).

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { ImagePayload } from "@/vision/image";

export interface DocVqaModelEntry {
  id: string;
  label: string;
  hint: string;
  params: number;
  bytes: MeasuredBytes;
  backends?: readonly Backend[];
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
  graphs: readonly string[];
}

/**
 * The catalogue — one entry, for the reason in the header.
 *
 * **Measured, and the naive sum is wrong by an order of magnitude.** The repo
 * totals ~4 GB at fp32 because it publishes **three alternative decoders**
 * (`decoder_model`, `decoder_with_past_model`, `decoder_model_merged`); only the
 * merged one is ever loaded. Summed per graph, off the Hub:
 *
 *   encoder_model          157.8 MB fp16 ·  91.4 MB q8 ·  51.5 MB q4f16
 *   decoder_model_merged   252.9 MB fp16 · 127.3 MB q8 · 189.5 MB q4f16
 *   ----------------------------------------------------------------
 *   total                  410.7 MB fp16 · 218.7 MB q8 · 241.0 MB q4f16
 *
 * **The WASM decoder must stay fp32, and this is a measurement.** A uniform q8
 * cannot open a session at all in the browser:
 *
 *   Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: qdq_actions.cc:137
 *   TransposeDQWeightsForMatMulNBits Missing required scale:
 *   decoder.model.decoder.embed_tokens.weight_merged_0_scale
 *
 * That is **the same ONNX Runtime bug `asrLoadOpts` was written for**, and
 * finding it here is the news: its note says it "reproduces on every ASR repo
 * tried", but Donut is not an ASR model, so the fault is not ASR-specific. It is
 * **any encoder-decoder whose decoder is quantized**, on the WASM execution
 * provider bundled with `@huggingface/transformers` 4.2.0. The encoder quantizes
 * fine; only the decoder pays full precision.
 *
 * Cost: 596.7 MB on WASM instead of 218.7 MB — the same ~3x the ASR catalogue
 * pays, and worth it for the same reason. The alternative is not a cheaper CPU
 * path but **no CPU path at all**. Revisit when the bundled ORT updates; the
 * check is `just fe-e2e-docvqa`, which is how this was found.
 *
 * It also reproduces in the browser only: the identical call loads cleanly under
 * `onnxruntime-node`, so a Node-side sanity check cannot catch it.
 *
 * **WebGPU keeps fp16 and is left alone.** The ASR precedent confines the bug to
 * WASM, and no adapter was available on the machine that found this, so pinning
 * the GPU path either way would be a guess rather than a measurement.
 *
 * **`q4f16` is still not pinned** for WebGPU. It would cut 410.7 MB to 241.0 MB,
 * but that would be a *precaution, not a measurement*, and document QA is where
 * quantization error lands directly on small printed characters.
 *
 * **Both backends are offered.** Unlike every model in `multimodal/types.ts`
 * this one is not an autoregressive chat decoder: it is an encoder plus a short
 * extractive decode, so the CPU path is real rather than theoretical — it is
 * merely expensive.
 */
export const DOCVQA_MODELS: DocVqaModelEntry[] = [
  {
    id: "Xenova/donut-base-finetuned-docvqa",
    label: "Donut base (DocVQA)",
    hint: "OCR-free: it reads the page and answers in one pass, with no text-detection step in between.",
    params: 200,
    graphs: ["encoder_model", "decoder_model_merged"],
    // Per-module precision, exactly as `asrLoadOpts` does it — the decoder
    // cannot be quantized on WASM without failing to open a session at all.
    dtypes: { wasm: { encoder_model: "q8", decoder_model_merged: "fp32" } },
    bytes: { webgpu: 410_768_162, wasm: 596_678_834 },
  },
];

export const DEFAULT_DOCVQA_MODEL = DOCVQA_MODELS[0].id;

/**
 * **This route does not downscale for the model, and that is the opposite of
 * every other vision-family page in the repo.**
 *
 * `preprocessor_config.json` is `do_resize` + `do_thumbnail` + `do_pad` at a
 * fixed `{ height: 2560, width: 1920 }`, and `thumbnail()` in 4.2.0 never
 * upscales:
 *
 *   height = min(input_height, output_height)
 *   width  = min(input_width,  output_width)
 *   if (unchanged) return image
 *
 * It shrinks to fit, preserving aspect, and then `do_pad` pads to exactly
 * 2560x1920. **So the encoder always sees a 2560x1920 tensor**, whatever it was
 * handed, and two things follow that invert the usual advice:
 *
 *   inference cost is constant   Capping the source saves nothing at the model.
 *   a small source is padded     Not upscaled. Fewer real pixels, same compute —
 *                                legibility given away for no return.
 *
 * `/image-text-to-text` caps at 512 because that is SmolVLM's tile size and
 * tiles cost tokens. Here there is no equivalent trade, so there is no slider:
 * the plan for this page originally proposed one and it was wrong, for exactly
 * the reason the repo already records about `/link-prediction` — reuse that
 * looks like a decision is often an inheritance.
 *
 * The cap below is therefore a **memory bound, not preprocessing**: 2560 is the
 * processor's own longest dimension, so clamping a 12-megapixel phone photo to
 * it is lossless with respect to what the model will see, while keeping the
 * decoded bitmap and the `postMessage` copy to a sane size.
 */
export const MAX_SOURCE_SIDE = 2560;

/** Generation cap. An extractive answer is a span, not a paragraph. */
export const MAX_NEW_TOKENS = 128;

export interface DocVqaLoad {
  model: string;
  opts?: LoadOpts;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export interface DocVqaRun {
  image: ImagePayload;
  question: string;
  maxNewTokens: number;
}

export interface DocVqaResult {
  /**
   * The extracted span — or **null**, which is an ordinary outcome rather than
   * an error.
   *
   * The pipeline pulls the answer out with
   * `decoded.match(/<s_answer>(.*?)<\/s_answer>/)` and returns `[{ answer: null }]`
   * when that misses. Nothing throws. So "the model produced nothing usable" has
   * to be rendered explicitly, or the page shows an empty panel and looks broken
   * on precisely the documents it found hardest.
   */
  answer: string | null;
  /** The question this answer was given, captured with it. */
  question: string;
  ms: number;
}

export type DocVqaRequest = ModelRequest<DocVqaLoad, DocVqaRun>;
export type DocVqaResponse = ModelResponse<DocVqaResult>;
