// Parity harness for the zero-shot scoring arithmetic, loaded *into the page* by
// `zero-shot-parity.spec.ts`.
//
// It lives here rather than in the spec body for one reason that matters: it
// imports the **shipped** `vision/zeroshot/scoring.ts`, so the comparison
// exercises the code the route actually runs. A reimplementation inside
// `page.evaluate` would only ever prove that two copies of the same idea agree.
//
// Vite serves and transforms any source file under the project root, so the spec
// can `import("/e2e/fixtures/clipParity.ts")` from the browser and get bare
// specifiers and the `@/` alias resolved the same way the app does.

import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  pipeline,
} from "@huggingface/transformers";

import { normalizedRows, scoreImage } from "@/vision/zeroshot/scoring";
import { ZERO_SHOT_MODELS, scoringSpec } from "@/vision/zeroShot";

const CATS =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";

export interface ParityReport {
  prompts: string[];
  /** What the full `model.onnx` graph says, via the pipeline. */
  reference: number[];
  /** What our two towers plus `scoring.ts` say. */
  ours: number[];
  /** True when the label embeddings were reused across the two image calls. */
  reusedTextEmbeddings: boolean;
  /** Our raw cosine similarities, before any scale — the diagnostic half. */
  cosines: number[];
  /** The dtype both paths ran at. */
  dtype: string;
}

/**
 * Score one image two ways on the same checkpoint, in the same browser, with the
 * same weights cache, and hand both sets of numbers back for comparison.
 */
export async function compareClipScoring(
  // **fp32, and that is the point.** `model.onnx` and `text_model.onnx` /
  // `vision_model.onnx` are *separately quantized* exports, so at q8 their
  // embeddings differ slightly — and a softmax at scale 100 turns a ~0.001
  // cosine difference into a ~0.07 probability difference. Comparing the two
  // paths at q8 measures the quantizer, not our arithmetic. At fp32 the paths
  // agree to six decimal places, which is what makes this a real check.
  dtype: "q8" | "fp32" = "fp32",
): Promise<ParityReport> {
  const meta = ZERO_SHOT_MODELS[0]; // CLIP ViT-B/32
  const prompts = ["a photo of a cat", "a photo of a dog", "an empty sofa"];
  const opts = { device: "wasm", dtype } as const;

  const image = await RawImage.fromURL(CATS);

  // --- reference: the full graph, with logit_scale baked into the ONNX --------
  const zs = await pipeline(
    "zero-shot-image-classification",
    meta.id,
    opts as never,
  );
  // Pass-through template: without it the pipeline wraps these prompts in its
  // own default, "This is a photo of {}", and the two paths score different text.
  const referenceRows = (await zs(image, prompts, {
    hypothesis_template: "{}",
  })) as { label: string; score: number }[];
  await (zs as unknown as { dispose: () => Promise<void> }).dispose();
  const reference = prompts.map(
    (p) => referenceRows.find((r) => r.label === p)?.score ?? -1,
  );

  // --- ours: two towers, then the arithmetic in scoring.ts -------------------
  const tokenizer = await AutoTokenizer.from_pretrained(meta.id);
  const processor = await AutoProcessor.from_pretrained(meta.id);
  const textModel = await CLIPTextModelWithProjection.from_pretrained(
    meta.id,
    opts as never,
  );
  const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
    meta.id,
    opts as never,
  );

  const textInputs = tokenizer(prompts, { padding: true, truncation: true });
  const { text_embeds } = (await textModel(textInputs)) as {
    text_embeds: { data: ArrayLike<number>; dims: number[] };
  };
  const [tRows, tDim] = text_embeds.dims;
  const texts = normalizedRows(text_embeds.data, tRows, tDim);

  const encodeImage = async () => {
    const { pixel_values } = await processor(image);
    const { image_embeds } = (await visionModel({ pixel_values })) as {
      image_embeds: { data: ArrayLike<number>; dims: number[] };
    };
    const [, dim] = image_embeds.dims;
    return normalizedRows(image_embeds.data, 1, dim)[0];
  };

  const first = await encodeImage();
  const ours = scoreImage(first, texts, scoringSpec(meta));
  const cosines = texts.map((t) =>
    t.reduce((sum, v, i) => sum + v * first[i], 0),
  );

  // The cache claim, exercised rather than asserted in a comment: a second image
  // is scored against the *same* text embeddings, with no second text pass.
  const second = await encodeImage();
  const again = scoreImage(second, texts, scoringSpec(meta));
  const reusedTextEmbeddings = again.every(
    (v, i) => Math.abs(v - ours[i]) < 1e-6,
  );

  await Promise.all([
    (textModel as unknown as { dispose: () => Promise<void> }).dispose(),
    (visionModel as unknown as { dispose: () => Promise<void> }).dispose(),
  ]);

  return { prompts, reference, ours, reusedTextEmbeddings, cosines, dtype };
}
