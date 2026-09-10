import { describe, expect, it } from "vitest";

import { DEFAULT_THRESHOLD as CLOSED_VOCAB_DEFAULT } from "./detection";
import {
  DEFAULT_QUERIES,
  DEFAULT_THRESHOLD,
  DEFAULT_ZERO_SHOT_DETECTOR,
  MODEL_THRESHOLD,
  THRESHOLD_RANGE,
  ZERO_SHOT_DETECTOR_MODELS,
} from "./zeroShotDetection";

describe("ZERO_SHOT_DETECTOR_MODELS", () => {
  it("makes OWLv2 the default", () => {
    expect(DEFAULT_ZERO_SHOT_DETECTOR).toBe(
      "Xenova/owlv2-base-patch16-ensemble",
    );
    expect(ZERO_SHOT_DETECTOR_MODELS[0].id).toBe(DEFAULT_ZERO_SHOT_DETECTOR);
  });

  it("gives every entry a measured download for both backends", () => {
    // These carry a text tower and a vision tower in one graph, so a
    // params-based estimate would be badly wrong.
    for (const model of ZERO_SHOT_DETECTOR_MODELS) {
      expect(model.bytes?.webgpu, model.id).toBeGreaterThan(0);
      expect(model.bytes?.wasm, model.id).toBeGreaterThan(0);
      expect(model.task, model.id).toBe("zero-shot-object-detection");
    }
  });

  it("marks Grounding DINO as reading phrases, not class names", () => {
    // The page's hint changes with this, because it changes what the user
    // should type — and Grounding DINO answers with a fragment of the query.
    const dino = ZERO_SHOT_DETECTOR_MODELS.find((m) => m.id.includes("grounding"))!;
    expect(dino.queries).toBe("phrase");
    expect(
      ZERO_SHOT_DETECTOR_MODELS.filter((m) => m.queries === "label").length,
    ).toBe(2);
  });
});

describe("the zero-shot detection thresholds", () => {
  it("starts far below a closed-vocabulary detector's default", () => {
    // **The decision this page most needed to get right.** An open-vocabulary
    // model spreads its probability over an unbounded label space, so OWLv2's
    // confident hits land where D-FINE's uncertain ones do. A COCO-detector
    // default renders an empty canvas over a picture full of correctly-found
    // objects, which reads as a broken page rather than a bad number.
    expect(DEFAULT_THRESHOLD).toBeLessThan(CLOSED_VOCAB_DEFAULT);
  });

  it("asks the model for far more than the page shows", () => {
    // So the slider re-filters instead of re-running — which here would cost a
    // 300 MB model a second of work per drag.
    expect(MODEL_THRESHOLD).toBeLessThan(DEFAULT_THRESHOLD);
    expect(MODEL_THRESHOLD).toBeLessThanOrEqual(THRESHOLD_RANGE.min);
  });

  it("keeps the slider inside the range these models actually use", () => {
    expect(DEFAULT_THRESHOLD).toBeGreaterThanOrEqual(THRESHOLD_RANGE.min);
    expect(DEFAULT_THRESHOLD).toBeLessThanOrEqual(THRESHOLD_RANGE.max);
    // Above ~0.5 an open-vocabulary detector finds nothing at all.
    expect(THRESHOLD_RANGE.max).toBeLessThanOrEqual(0.5);
  });
});

describe("DEFAULT_QUERIES", () => {
  it("keeps its articles, because this route applies no template", () => {
    // The opposite of `zeroShot.ts`'s DEFAULT_LABELS, and the difference
    // matters: the pipeline tokenizes `candidate_labels` verbatim, so what is
    // on screen is exactly what the text tower is asked. Bare nouns here would
    // be a weaker prompt, not a neutral one.
    expect(DEFAULT_QUERIES.length).toBeGreaterThan(0);
    for (const query of DEFAULT_QUERIES) {
      expect(query, query).toMatch(/^(a|an|the) /);
    }
  });
});
