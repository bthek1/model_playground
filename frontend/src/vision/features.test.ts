import { describe, expect, it } from "vitest";

import {
  FEATURE_MODELS,
  poolEmbedding,
  vectorKinds,
  type VectorKind,
} from "./features";
import { l2norm } from "./similarity";

/** `[1, tokens, dim]` in row-major order, the shape a ViT hands back. */
function tokens(rows: number[][]) {
  return {
    data: Float32Array.from(rows.flat()),
    dims: [1, rows.length, rows[0].length],
  };
}

describe("poolEmbedding", () => {
  it("takes CLS from row 0 and the mean from the patch rows", () => {
    // Hand-computed: patches are [2,0,0] and [0,4,0], so the mean is [1,2,0],
    // which normalises to [0.4472, 0.8944, 0]. Row 0 is CLS and is *not* part
    // of the average — including it is the classic off-by-one here, and it
    // produces a plausible vector with quietly wrong neighbours.
    const out = poolEmbedding(
      tokens([
        [1, 0, 0],
        [2, 0, 0],
        [0, 4, 0],
      ]),
    );

    expect([...out.vectors.cls!]).toEqual([1, 0, 0]);
    expect(out.vectors.mean![0]).toBeCloseTo(1 / Math.sqrt(5), 5);
    expect(out.vectors.mean![1]).toBeCloseTo(2 / Math.sqrt(5), 5);
    expect(out.tokens).toBe(3);
    expect(out.dim).toBe(3);
  });

  it("returns unit vectors, and the norms they had before", () => {
    // The page displays both, because "the vectors are normalised" is a claim
    // it can simply show.
    const out = poolEmbedding(
      tokens([
        [3, 4, 0],
        [0, 0, 5],
      ]),
    );
    expect(l2norm(out.vectors.cls!)).toBeCloseTo(1, 6);
    expect(l2norm(out.vectors.mean!)).toBeCloseTo(1, 6);
    expect(out.norms.cls).toBeCloseTo(5, 6);
    expect(out.norms.mean).toBeCloseTo(5, 6);
  });

  it("offers CLS only when there are no patch rows to average", () => {
    const out = poolEmbedding(tokens([[1, 0]]));
    expect(vectorKinds(out)).toEqual(["cls"]);
  });

  it("treats an already-pooled [1, dim] output as one projected vector", () => {
    // CLIP as a feature extractor: `CLIPVisionModelWithProjection` emits
    // `image_embeds` and there is no pooling choice to offer. The page must say
    // so rather than show a toggle that does nothing.
    const out = poolEmbedding({ data: Float32Array.from([3, 4]), dims: [1, 2] });
    expect(vectorKinds(out)).toEqual<VectorKind[]>(["pooled"]);
    expect(out.tokens).toBe(1);
    expect(out.dim).toBe(2);
    expect(out.norms.pooled).toBeCloseTo(5, 6);
    expect(l2norm(out.vectors.pooled!)).toBeCloseTo(1, 6);
  });

  it("accepts a bare [dim] output", () => {
    const out = poolEmbedding({ data: Float32Array.from([0, 1]), dims: [2] });
    expect(vectorKinds(out)).toEqual<VectorKind[]>(["pooled"]);
  });

  it("throws on a shape it does not understand rather than guessing", () => {
    // Reinterpreting someone's numbers as a vector produces confident,
    // plausible, meaningless neighbours — the exact failure mode this file's
    // header warns about.
    expect(() => poolEmbedding({ data: [1, 2, 3, 4], dims: [1, 2, 2, 1] })).toThrow(
      /unexpected embedding shape/i,
    );
    expect(() => poolEmbedding({ data: [], dims: [1, 0, 0] })).toThrow(
      /empty embedding/i,
    );
  });
});

describe("FEATURE_MODELS", () => {
  it("pins a precision for the checkpoint that publishes no fp16", () => {
    // DINOv3 small ships fp32, q4 and q8 only. Left to `loadOpts()`, WebGPU
    // would ask for a file that does not exist and the load would 404 — the
    // same class of failure as the missing mobilevitv2 export.
    const dinov3 = FEATURE_MODELS.find((m) => m.id.includes("dinov3"))!;
    expect(dinov3.dtypes?.webgpu).toBe("q8");
    expect(dinov3.dtypes?.wasm).toBe("q8");
  });

  it("declares CLIP's vision tower as the only graph it downloads", () => {
    // `AutoModelForImageFeatureExtraction` resolves CLIP to
    // `CLIPVisionModelWithProjection`, whose `model_file_name` is
    // `vision_model` — there is no `model.onnx` to check against.
    const clip = FEATURE_MODELS.find((m) => m.id.includes("clip"))!;
    expect(clip.graphs).toEqual(["vision_model"]);
  });

  it("gives every entry a measured download for both backends", () => {
    for (const model of FEATURE_MODELS) {
      expect(model.bytes?.webgpu, model.id).toBeGreaterThan(0);
      expect(model.bytes?.wasm, model.id).toBeGreaterThan(0);
    }
  });
});
