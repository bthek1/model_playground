import { describe, expect, it } from "vitest";

import { LARGE_MODEL_BYTES, sizeEstimate } from "@/model/size";

import {
  DEFAULT_IMAGE_CLASSIFIER,
  IMAGE_CLASSIFIER_MODELS,
  TOP_K,
} from "./classification";

describe("IMAGE_CLASSIFIER_MODELS", () => {
  it("gives every entry a unique id, a positive param count and the right task", () => {
    const ids = IMAGE_CLASSIFIER_MODELS.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const m of IMAGE_CLASSIFIER_MODELS) {
      expect(m.params, m.id).toBeGreaterThan(0);
      expect(m.task, m.id).toBe("image-classification");
      expect(m.label, m.id).toBeTruthy();
      expect(m.hint, m.id).toBeTruthy();
    }
  });

  it("defaults to a model that is in the catalogue", () => {
    expect(IMAGE_CLASSIFIER_MODELS.map((m) => m.id)).toContain(
      DEFAULT_IMAGE_CLASSIFIER,
    );
  });

  it("can always quote a download before the user commits to one", () => {
    for (const m of IMAGE_CLASSIFIER_MODELS) {
      const size = sizeEstimate(m.params, m.bytes);
      expect(size.label, m.id).toMatch(/MB|GB/);
      // A tab is not a workstation.
      expect(Math.max(size.fp16, size.q8), m.id).toBeLessThan(LARGE_MODEL_BYTES);
    }
  });

  it("keeps MobileNetV4 off q8, because its quantized export is wrong", () => {
    // Not slower — *wrong*. The q8 build labels the bundled tiger photo
    // "sidewinder, horned rattlesnake" at 44%; at fp32 the same weights say
    // "tiger 62%". Depthwise-separable convolutions are the classic casualty of
    // per-tensor int8 quantization. Deleting this override is a silent accuracy
    // regression, not a load failure, so it is pinned here as well as measured
    // in the @slow spec.
    const mobilenet = IMAGE_CLASSIFIER_MODELS.find((m) =>
      m.id.includes("mobilenetv4"),
    );
    expect(mobilenet?.dtypes).toEqual({ wasm: "fp32" });
  });

  it("quotes MobileNetV4's real fp32 download, not the q8 estimate", () => {
    // The override makes the params estimate lie by ~4x (3.8 MB quoted for a
    // 15 MB download), which is worse than not quoting at all.
    const mobilenet = IMAGE_CLASSIFIER_MODELS.find((m) =>
      m.id.includes("mobilenetv4"),
    )!;
    expect(mobilenet.bytes?.wasm).toBe(15_086_122);
    const size = sizeEstimate(mobilenet.params, mobilenet.bytes);
    expect(size.q8).toBeGreaterThan(sizeEstimate(mobilenet.params).q8 * 3);
  });

  it("only overrides precision where a measurement says to", () => {
    // Every other entry rides the shared `loadOpts()` defaults; an override
    // nobody has measured is just a slower download.
    const overridden = IMAGE_CLASSIFIER_MODELS.filter((m) => m.dtypes);
    expect(overridden.map((m) => m.id)).toEqual([
      "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    ]);
  });
});

describe("TOP_K", () => {
  it("is five, so a near-tie at the top is visible", () => {
    // The interesting case is 0.31 / 0.29, and a single label hides it.
    expect(TOP_K).toBe(5);
  });
});
