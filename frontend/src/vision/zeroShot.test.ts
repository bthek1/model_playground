import { describe, expect, it } from "vitest";

import {
  applyTemplate,
  buildPrompts,
  DEFAULT_LABELS,
  DEFAULT_ZERO_SHOT_MODEL,
  scoringSpec,
  TEMPLATES,
  ZERO_SHOT_MODELS,
} from "./zeroShot";

describe("prompt templates", () => {
  it("substitutes the label into the placeholder", () => {
    expect(applyTemplate(TEMPLATES.photo, "cat")).toBe("a photo of a cat");
    expect(applyTemplate(TEMPLATES.bare, "cat")).toBe("cat");
  });

  it("replaces every placeholder, not just the first", () => {
    expect(applyTemplate("{} and {}", "cat")).toBe("cat and cat");
  });

  it("falls back to the bare label for a template with no placeholder", () => {
    // Otherwise a user who deletes the `{}` sends the same constant string for
    // every label, and every score comes back identical with no explanation.
    expect(applyTemplate("a photo", "cat")).toBe("cat");
  });

  it("applies the template across the list, in order", () => {
    expect(buildPrompts(TEMPLATES.photo, ["cat", "dog"])).toEqual([
      "a photo of a cat",
      "a photo of a dog",
    ]);
  });

  it("ships bare nouns as defaults, so a template composes correctly", () => {
    // `"a cat"` with `"a photo of a {}"` gives "a photo of a a cat" — visibly
    // broken in the UI and a worse prompt than either template alone.
    for (const label of DEFAULT_LABELS) {
      expect(label).not.toMatch(/^(a|an|the)\s/i);
      expect(applyTemplate(TEMPLATES.photo, label)).not.toMatch(/of a a /);
    }
  });
});

describe("the zero-shot catalogue", () => {
  it("defaults to CLIP", () => {
    expect(DEFAULT_ZERO_SHOT_MODEL).toBe(ZERO_SHOT_MODELS[0].id);
  });

  it("carries each checkpoint's own learned scoring constants", () => {
    // **Read from the published weights, not guessed** — CLIP's raw logit_scale
    // is 4.605170 in `openai/clip-vit-base-patch32`'s pytorch_model.bin, and
    // exp(4.605170) = 100.000006; the SigLIP pair come from their safetensors.
    // Splitting the towers to cache the text side means reapplying these
    // ourselves, and a wrong value fails *silently*: every score stays in [0, 1]
    // and the ranking is untouched. This test stops an accidental edit; only
    // `just fe-e2e-zeroshot` can prove the values are right.
    const by = (id: string) => ZERO_SHOT_MODELS.find((m) => m.id === id)!;

    expect(by("Xenova/clip-vit-base-patch32")).toMatchObject({
      family: "clip",
      scoring: "softmax",
      logitScale: 100.000006,
    });
    expect(by("Xenova/clip-vit-base-patch32").logitBias).toBeUndefined();

    expect(by("Xenova/siglip-base-patch16-224")).toMatchObject({
      family: "siglip",
      scoring: "sigmoid",
      logitScale: 117.330795,
      logitBias: -12.932437,
    });
    expect(by("onnx-community/siglip2-base-patch16-224-ONNX")).toMatchObject({
      family: "siglip",
      scoring: "sigmoid",
      logitScale: 112.668907,
      logitBias: -16.771725,
    });
  });

  it("gives every entry the parameters its scoring needs", () => {
    for (const model of ZERO_SHOT_MODELS) {
      expect(model.logitScale).toBeGreaterThan(1);
      // A sigmoid model without a bias would score every label near 1: the bias
      // is what shifts an unmatched label below the midpoint.
      if (model.scoring === "sigmoid") {
        expect(model.logitBias).toBeTypeOf("number");
      }
      expect(model.bytes).toBeDefined(); // measured, never estimated
    }
  });

  it("hands the engine the spec shape, dropping a bias softmax cannot use", () => {
    // A constant added to every logit cancels in a softmax, so carrying it would
    // be a silent no-op that looks meaningful.
    expect(scoringSpec(ZERO_SHOT_MODELS[0])).toEqual({
      kind: "softmax",
      scale: 100.000006,
    });
    expect(scoringSpec(ZERO_SHOT_MODELS[1])).toEqual({
      kind: "sigmoid",
      scale: 117.330795,
      bias: -12.932437,
    });
  });
});
