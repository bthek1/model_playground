import { describe, expect, it } from "vitest";

import { sizeEstimate } from "@/model/size";

import { DEFAULT_SAM_MODEL, MAX_INFERENCE_SIDE, SAM_MODELS } from "./types";

describe("SAM_MODELS", () => {
  it("makes the small one the default", () => {
    expect(DEFAULT_SAM_MODEL).toBe("Xenova/slimsam-77-uniform");
    expect(SAM_MODELS[0].id).toBe(DEFAULT_SAM_MODEL);
  });

  it("names both graphs, because neither repo publishes model.onnx", () => {
    // SAM ships a vision encoder and a prompt-encoder/mask-decoder. A Hub-id
    // check hard-coded to `onnx/model.onnx` would look at a file that does not
    // exist in either repo and pass.
    for (const model of SAM_MODELS) {
      expect(model.graphs, model.id).toEqual([
        "vision_encoder",
        "prompt_encoder_mask_decoder",
      ]);
    }
  });

  it("quotes both graphs' blobs, not just the decoder", () => {
    // The decoder is the small half — quoting it alone would understate
    // SlimSAM by ~60% and SAM 2.1 by ~85%.
    for (const model of SAM_MODELS) {
      expect(model.bytes.webgpu, model.id).toBeGreaterThan(0);
      expect(model.bytes.wasm, model.id).toBeGreaterThan(0);
      // q8 is smaller than fp16 for both, which is the sanity check that
      // catches a transposed pair of measurements.
      expect(model.bytes.wasm, model.id).toBeLessThan(model.bytes.webgpu!);
    }
  });

  it("keeps the default small enough that the guardrail stays meaningful", () => {
    // ~21 MB. A warning on everything is a warning on nothing, and this is the
    // cheapest interactive model in the whole vision category.
    const slim = SAM_MODELS.find((m) => m.id === DEFAULT_SAM_MODEL)!;
    expect(sizeEstimate(slim.params, slim.bytes).large).toBe(false);
  });

  it("caps the encoded frame, because the masks come back at its resolution", () => {
    // Not only a speed throttle here: the decoder returns three masks per click
    // at the source resolution, one byte per pixel. A 12-megapixel photo would
    // post 36 MB across the worker boundary on every click.
    expect(MAX_INFERENCE_SIDE).toBeGreaterThan(0);
    expect(MAX_INFERENCE_SIDE).toBeLessThanOrEqual(1024);
  });
});
