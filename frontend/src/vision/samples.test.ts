import { describe, expect, it } from "vitest";

import { DEFAULT_IMAGE_SAMPLE, IMAGE_SAMPLES } from "./samples";

describe("IMAGE_SAMPLES", () => {
  it("lists the demo images with unique ids and https urls", () => {
    expect(IMAGE_SAMPLES.length).toBeGreaterThan(0);
    const ids = IMAGE_SAMPLES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const s of IMAGE_SAMPLES) {
      expect(s.url, s.id).toMatch(/^https:\/\/.+\.(jpg|jpeg|png)$/);
      expect(s.label, s.id).toBeTruthy();
      expect(s.hint, s.id).toBeTruthy();
    }
  });

  it("keeps every sample on the CORS-enabled Hub origin the weights come from", () => {
    // No binaries in the repo, and no origin that would need a proxy — these are
    // the canonical Transformers.js demo images.
    for (const s of IMAGE_SAMPLES) {
      expect(s.url, s.id).toContain(
        "huggingface.co/datasets/Xenova/transformers.js-docs",
      );
    }
  });

  it("says what a healthy model should report, so a sample is a smoke test", () => {
    // Without this a sample is decoration: a user cannot tell a working model
    // from a broken one by looking at the output.
    for (const s of IMAGE_SAMPLES) {
      expect(s.expect, s.id).toBeTruthy();
    }
    expect(IMAGE_SAMPLES.find((s) => s.id === "tiger")?.expect).toMatch(/tiger/i);
  });

  it("defaults to a sample that exists", () => {
    expect(IMAGE_SAMPLES.map((s) => s.id)).toContain(DEFAULT_IMAGE_SAMPLE);
  });
});
