import { describe, expect, it } from "vitest";

import { LARGE_MODEL_BYTES, sizeEstimate } from "@/model/size";

import {
  DEFAULT_MAX_PEOPLE,
  DEFAULT_PERSON_THRESHOLD,
  DEFAULT_POSE_MODEL,
  DETECTOR_THRESHOLD,
  MAX_PEOPLE_LIMIT,
  POSE_MODELS,
} from "./types";

describe("POSE_MODELS", () => {
  it("quotes each pair's download as the sum of both halves", () => {
    // **The number most likely to drift, and the one the guardrail depends on.**
    // A catalogue entry here is a *pair*, and a size that quotes half the bytes
    // is worse than quoting none. Derived rather than trusted: updating one
    // half's measurement without the composite is exactly the silent mistake.
    for (const pair of POSE_MODELS) {
      for (const backend of ["webgpu", "wasm"] as const) {
        expect(pair.bytes[backend], `${pair.id} (${backend})`).toBe(
          (pair.detector.bytes[backend] ?? 0) + (pair.pose.bytes[backend] ?? 0),
        );
      }
    }
  });

  it("crosses the large-model threshold on the sum, not on either half", () => {
    // What "less obviously than a single big model" means, concretely: the
    // RT-DETR pair is 88 MB of detector and 172 MB of pose model, each
    // comfortably under LARGE_MODEL_BYTES and 260 MB together, over it.
    const heavy = POSE_MODELS.find((m) => m.id.startsWith("rtdetr"))!;
    expect(sizeEstimate(0, heavy.detector.bytes).fp16).toBeLessThan(
      LARGE_MODEL_BYTES,
    );
    expect(sizeEstimate(0, heavy.pose.bytes).fp16).toBeLessThan(
      LARGE_MODEL_BYTES,
    );
    expect(sizeEstimate(heavy.params, heavy.bytes).large).toBe(true);
  });

  it("keeps the default pair under the threshold, so the warning still means something", () => {
    // A warning on everything is a warning on nothing.
    const nano = POSE_MODELS.find((m) => m.id === DEFAULT_POSE_MODEL)!;
    expect(sizeEstimate(nano.params, nano.bytes).large).toBe(false);
  });

  it("offers live mode only on the pair that fits a webcam", () => {
    // Two forward passes per person, per frame. A 260 MB detector is not a
    // webcam demo, and the page says so rather than dropping frames.
    const nano = POSE_MODELS.find((m) => m.id === DEFAULT_POSE_MODEL)!;
    expect(nano.live).toBe(true);
    expect(POSE_MODELS.filter((m) => m.live)).toHaveLength(1);
  });

  it("names both graphs of both halves, for the Hub-id spec", () => {
    // The pair's own `id` is a composite that resolves to nothing on the Hub —
    // the two halves are what get downloaded, and what get checked.
    for (const pair of POSE_MODELS) {
      for (const stage of [pair.detector, pair.pose]) {
        expect(stage.id, pair.id).toMatch(/\//);
        expect(stage.graphs.length, stage.id).toBeGreaterThan(0);
      }
    }
  });

  it("reuses the detection catalogue's checkpoints rather than forking them", () => {
    // Step 1 is /object-detection's model. A second copy of those ids would
    // drift the moment either is re-measured.
    const ids = POSE_MODELS.map((m) => m.detector.id);
    expect(ids).toContain("onnx-community/dfine_n_coco-ONNX");
    expect(ids).toContain("onnx-community/rtdetr_r50vd");
  });
});

describe("the pose thresholds", () => {
  it("asks the detector for far more than the page shows", () => {
    // The page's slider starts well above the detector's floor, so dragging it
    // down reveals what the model already returned.
    expect(DETECTOR_THRESHOLD).toBeLessThan(DEFAULT_PERSON_THRESHOLD);
  });

  it("caps the crowd below the hard limit, so a busy scene cannot wedge a tab", () => {
    expect(DEFAULT_MAX_PEOPLE).toBeLessThan(MAX_PEOPLE_LIMIT);
    expect(MAX_PEOPLE_LIMIT).toBeGreaterThan(0);
  });
});
