import { describe, expect, it } from "vitest";

import {
  DEFAULT_MATTE_MODEL,
  MATTE_MODELS,
  MAX_INFERENCE_SIDE,
} from "./backgroundRemoval";

describe("MATTE_MODELS", () => {
  it("defaults to a model that permits commercial use", () => {
    // **The invariant #24 exists to protect, and the reason it is a test rather
    // than a comment.** RMBG-1.4 is the better matte and is Creative Commons
    // non-commercial; this repo is MIT. A default most downstream users may not
    // legally use is a trap, and it is a one-line edit away at any time.
    const fallback = MATTE_MODELS.find((m) => m.id === DEFAULT_MATTE_MODEL)!;
    expect(fallback).toBeDefined();
    expect(fallback.licence.commercial).toBe(true);
    // `useModelSelection` falls back to `MATTE_MODELS[0]` for an unknown stored
    // id, so the *first* entry has to clear the same bar as the named default.
    expect(MATTE_MODELS[0].licence.commercial).toBe(true);
  });

  it("gives the non-commercial entry a note and a link to its terms", () => {
    // A restriction the UI cannot explain is a restriction the user cannot act
    // on. `LicenceNote` renders `note` and `url`; neither may be empty.
    for (const model of MATTE_MODELS.filter((m) => !m.licence.commercial)) {
      expect(model.licence.note, model.id).toBeTruthy();
      expect(model.licence.url, model.id).toMatch(/^https:\/\//);
    }
  });

  it("names every licence, permissive or not", () => {
    for (const model of MATTE_MODELS) {
      expect(model.licence.name, model.id).toBeTruthy();
      expect(model.licence.url, model.id).toMatch(/^https:\/\//);
    }
  });

  it("runs the background-removal pipeline, not image-segmentation", () => {
    // The distinction is load-bearing: `background-removal` subclasses the
    // segmentation pipeline and puts the mask into the source's alpha channel,
    // which is the whole result this route displays.
    for (const model of MATTE_MODELS) {
      expect(model.task, model.id).toBe("background-removal");
    }
  });

  it("gives every entry a measured download for both backends", () => {
    for (const model of MATTE_MODELS) {
      expect(model.bytes?.webgpu, model.id).toBeGreaterThan(0);
      expect(model.bytes?.wasm, model.id).toBeGreaterThan(0);
    }
  });

  it("keeps enough resolution for an edge to be judged", () => {
    // Higher than the detection routes' 640 on purpose: a matte is judged on its
    // edge, and 640 px of a portrait leaves too few pixels across a strand of
    // hair for the difference between these two models to be visible at all.
    expect(MAX_INFERENCE_SIDE).toBeGreaterThanOrEqual(1024);
  });
});
