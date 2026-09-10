// Image feature extraction — the task with no visible output of its own, so the
// page *is* the similarity search. The whole index lives in the tab's memory,
// which is what makes an in-browser version genuinely private rather than merely
// convenient.
//
// **"Which vector do you actually take" is a real UI control here** (roadmap
// §3.8), and it is the reason this page exists rather than a hidden helper
// module. A ViT's last hidden state is `[1, 1 + patches, dim]`: row 0 is the CLS
// token the model was trained to summarise with, and the remaining rows are the
// patches. Mean-pooling the patches gives visibly different neighbours from
// taking CLS, and a toggle makes that legible in a way prose cannot.
//
// One forward pass yields **both** vectors, so the toggle re-derives rather than
// re-running — the same rule the detection threshold follows. A CLIP-shaped
// model returns one already-projected vector instead and offers no choice; the
// page says so rather than showing a toggle that does nothing.
//
// Sizes are ONNX blob totals read off the Hub:
//
//   DINOv2 small   fp16  44 MB · q8  24 MB
//   DINOv2 base    fp16 173 MB · q8  91 MB
//   CLIP ViT-B/32  fp16 176 MB · q8  89 MB   (the vision tower only)
//
// `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX` is the current leader
// and is in the catalogue, but it **publishes no fp16 export** — only fp32, q4
// and q8 — so its precision is pinned to q8 on both backends rather than left to
// `loadOpts()`, which would ask WebGPU for a file that does not exist. This is
// the same class of failure as the missing `mobilevitv2` fp16 export, and
// `just fe-e2e-models` is what catches it.

import type { PlainTensor } from "./serialize";
import { l2norm, normalize } from "./similarity";
import type { VisionModel } from "./types";

export interface FeatureModel extends VisionModel {
  task: "image-feature-extraction";
  /** Embedding width, for the page to state before anything is downloaded. */
  dim: number;
}

export const FEATURE_MODELS: FeatureModel[] = [
  {
    id: "Xenova/dinov2-small",
    label: "DINOv2 small",
    hint: "Self-supervised features with no labels anywhere in training. 384-d, and the default.",
    params: 22,
    dim: 384,
    task: "image-feature-extraction",
    bytes: { webgpu: 44_427_534, wasm: 24_451_943 },
  },
  {
    id: "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX",
    label: "DINOv3 small/16",
    hint: "The current leader. No fp16 export exists, so it runs q8 on both backends.",
    params: 21,
    dim: 384,
    task: "image-feature-extraction",
    // Not a preference — see the header. The repo publishes fp32, q4 and q8 only.
    dtypes: { webgpu: "q8", wasm: "q8" },
    bytes: { webgpu: 21_945_937, wasm: 21_945_937 },
  },
  {
    id: "Xenova/dinov2-base",
    label: "DINOv2 base",
    hint: "Four times the weights of small, 768-d, and visibly better neighbours on the hard pairs.",
    params: 86,
    dim: 768,
    task: "image-feature-extraction",
    bytes: { webgpu: 173_475_319, wasm: 90_974_542 },
  },
  {
    id: "Xenova/clip-vit-base-patch32",
    label: "CLIP ViT-B/32",
    hint: "Language-aligned: one projected 512-d vector, so there is no pooling choice to make.",
    params: 88,
    dim: 512,
    task: "image-feature-extraction",
    // Only the vision tower is downloaded here — `AutoModelForImageFeatureExtraction`
    // resolves CLIP to `CLIPVisionModelWithProjection`, whose `model_file_name`
    // is `vision_model`. The text tower is never fetched on this route.
    graphs: ["vision_model"],
    bytes: { webgpu: 176_080_659, wasm: 89_117_001 },
  },
];

export const DEFAULT_FEATURE_MODEL = FEATURE_MODELS[0].id;

/** Longest side fed to the model — resolution is the throttle, not the model. */
export const MAX_INFERENCE_SIDE = 640;

/** How many neighbours the page shows by default. */
export const DEFAULT_K = 5;

// --- Pooling -----------------------------------------------------------------

/**
 * Which vector out of one forward pass.
 *
 * `pooled` is not a choice the user makes — it is what a model that returns an
 * already-projected embedding (CLIP) gives you, named so the page can say
 * "this model offers no choice" instead of showing a dead toggle.
 */
export type VectorKind = "cls" | "mean" | "pooled";

export const VECTOR_LABELS: Record<VectorKind, string> = {
  cls: "CLS token",
  mean: "Mean of patches",
  pooled: "Projected embedding",
};

export const VECTOR_HINTS: Record<VectorKind, string> = {
  cls: "Row 0 — the token the model was trained to summarise the image with.",
  mean: "The patch rows averaged. More about texture and layout, less about the subject.",
  pooled: "The model's own projected output. This checkpoint returns one vector, so there is nothing to choose.",
};

export interface Embedding {
  /** Unit-length vectors, one per kind this forward pass can offer. */
  vectors: Partial<Record<VectorKind, Float32Array>>;
  /**
   * The L2 norm each vector had **before** normalising. Displayed, because
   * "the vectors are normalised" is a claim the page can simply show rather
   * than assert in prose.
   */
  norms: Partial<Record<VectorKind, number>>;
  dim: number;
  /** Token rows the model returned. 1 when it returned a pooled vector. */
  tokens: number;
}

/**
 * Derive every vector this page can offer from one forward pass.
 *
 * `[1, T, D]` — a ViT's last hidden state — gives both CLS (row 0) and the mean
 * of the remaining patch rows. `[1, D]` or `[D]` — an already-projected output —
 * gives one `pooled` vector and no choice. Anything else is a model whose output
 * we do not understand, and it throws rather than silently reinterpreting
 * someone's numbers as a vector.
 */
export function poolEmbedding(tensor: PlainTensor): Embedding {
  const dims = tensor.dims ?? [];
  const data = tensor.data;

  // [batch, tokens, dim] — the interesting case.
  if (dims.length === 3) {
    const [, tokens, dim] = dims;
    if (tokens < 1 || dim < 1) {
      throw new Error(`Empty embedding: dims [${dims.join(", ")}]`);
    }
    const cls = new Float32Array(dim);
    for (let d = 0; d < dim; d++) cls[d] = data[d];

    // Rows 1… are the patches. A single-row output has no patches to average,
    // so CLS is all there is.
    const patches = tokens - 1;
    const vectors: Partial<Record<VectorKind, Float32Array>> = {
      cls: normalize(cls),
    };
    const norms: Partial<Record<VectorKind, number>> = { cls: l2norm(cls) };

    if (patches > 0) {
      const mean = new Float32Array(dim);
      for (let t = 1; t < tokens; t++) {
        const base = t * dim;
        for (let d = 0; d < dim; d++) mean[d] += data[base + d];
      }
      for (let d = 0; d < dim; d++) mean[d] /= patches;
      vectors.mean = normalize(mean);
      norms.mean = l2norm(mean);
    }

    return { vectors, norms, dim, tokens };
  }

  // [batch, dim] or [dim] — already pooled and projected.
  if (dims.length === 2 || dims.length === 1) {
    const dim = dims[dims.length - 1];
    if (dim < 1) throw new Error(`Empty embedding: dims [${dims.join(", ")}]`);
    const pooled = new Float32Array(dim);
    for (let d = 0; d < dim; d++) pooled[d] = data[d];
    return {
      vectors: { pooled: normalize(pooled) },
      norms: { pooled: l2norm(pooled) },
      dim,
      tokens: 1,
    };
  }

  throw new Error(
    `Unexpected embedding shape [${dims.join(", ")}] — expected [batch, tokens, dim] or [batch, dim]`,
  );
}

/** The kinds a given embedding actually carries, in a stable display order. */
export function vectorKinds(embedding: Embedding): VectorKind[] {
  return (["cls", "mean", "pooled"] as const).filter(
    (kind) => embedding.vectors[kind] != null,
  );
}
