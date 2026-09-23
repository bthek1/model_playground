import { describe, expect, it } from "vitest";

import { sizeEstimate } from "@/model/size";

import {
  CLASSIFIER_SAMPLES,
  DEFAULT_TEXT_CLASSIFIER,
  TEXT_CLASSIFIER_MODELS,
  TOP_K,
} from "./catalogue";

describe("the text catalogue", () => {
  it("measures every entry's download for both backends", () => {
    // The category-wide rule, and it is not decoration: #5's size tables were
    // all q8 figures while `loadOpts()` asks for fp16 on WebGPU, which is
    // roughly double all the way down. An entry without both numbers quotes
    // the user a price for a download they are not making.
    for (const m of TEXT_CLASSIFIER_MODELS) {
      expect(m.bytes.webgpu, `${m.id} webgpu bytes`).toBeGreaterThan(0);
      expect(m.bytes.wasm, `${m.id} wasm bytes`).toBeGreaterThan(0);
      // fp16 is two bytes a parameter and q8 is one, so the GPU download is
      // the bigger of the pair. A pair the other way round is a transposed
      // measurement, which reads as plausible and is not.
      expect(m.bytes.webgpu!, `${m.id}`).toBeGreaterThan(m.bytes.wasm!);
    }
  });

  it("warns on the size the user will actually pay, not the smaller one", () => {
    // `sizeEstimate` keys `large` off the bigger of the two downloads. Two of
    // these three cross LARGE_MODEL_BYTES on WebGPU while reading as
    // comfortably under it at q8 — the guardrail only gets that right because
    // the measured bytes are here rather than a params estimate.
    const twitter = TEXT_CLASSIFIER_MODELS.find(
      (m) => m.id === "Xenova/twitter-roberta-base-sentiment-latest",
    )!;
    expect(sizeEstimate(twitter.params, twitter.bytes).large).toBe(true);

    const distil = TEXT_CLASSIFIER_MODELS.find(
      (m) => m.id === DEFAULT_TEXT_CLASSIFIER,
    )!;
    expect(sizeEstimate(distil.params, distil.bytes).large).toBe(false);
  });

  it("offers only checkpoints with a trained classification head", () => {
    // `onnx-community/ModernBERT-base-ONNX` is deliberately absent against the
    // roadmap's §3.1 table: a base encoder has no trained head, so it emits
    // LABEL_0/LABEL_1 from random weights — confident, fluent and meaningless,
    // with nothing failing on the way there. Pinned so it cannot drift back in.
    const ids = TEXT_CLASSIFIER_MODELS.map((m) => m.id);
    expect(ids).not.toContain("onnx-community/ModernBERT-base-ONNX");
    for (const m of TEXT_CLASSIFIER_MODELS) {
      expect(m.labels.length, `${m.id} declares its labels`).toBeGreaterThan(1);
      expect(m.labels.some((l) => /^LABEL_\d+$/.test(l))).toBe(false);
    }
  });

  it("gives every entry a distinct training domain — the page's whole subject", () => {
    const domains = TEXT_CLASSIFIER_MODELS.map((m) => m.domain);
    expect(new Set(domains).size).toBe(domains.length);
  });

  it("shows enough labels to expose a near-tie", () => {
    // TOP_K covers the widest head in the catalogue, so no page ever renders a
    // truncated distribution that looks like a confident one.
    const widest = Math.max(...TEXT_CLASSIFIER_MODELS.map((m) => m.labels.length));
    expect(TOP_K).toBeGreaterThanOrEqual(widest);
  });

  it("defaults to a model in the catalogue", () => {
    expect(TEXT_CLASSIFIER_MODELS.map((m) => m.id)).toContain(
      DEFAULT_TEXT_CLASSIFIER,
    );
  });

  it("ships samples the models disagree on", () => {
    expect(CLASSIFIER_SAMPLES.length).toBeGreaterThanOrEqual(3);
    for (const s of CLASSIFIER_SAMPLES) {
      expect(s.text.trim().length).toBeGreaterThan(0);
      // Each sample says which model it is a trap for — a sample set every
      // model gets right demonstrates nothing about any of them.
      expect(s.hint.trim().length).toBeGreaterThan(0);
    }
    expect(new Set(CLASSIFIER_SAMPLES.map((s) => s.id)).size).toBe(
      CLASSIFIER_SAMPLES.length,
    );
  });
});
