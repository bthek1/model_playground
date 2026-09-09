import { describe, expect, it } from "vitest";

import { sizeEstimate } from "@/model/size";
import {
  DEFAULT_MUSIC_MODEL,
  DEFAULT_SECONDS,
  MAX_SECONDS,
  MUSIC_MODELS,
  TOKENS_PER_SECOND,
  tokensForSeconds,
} from "./textToAudio";

describe("MUSIC_MODELS", () => {
  it("has a default that exists in the catalogue", () => {
    expect(MUSIC_MODELS.some((m) => m.id === DEFAULT_MUSIC_MODEL)).toBe(true);
  });

  it("carries measured bytes — the params estimate is useless for a 3-graph model", () => {
    for (const m of MUSIC_MODELS) {
      // 300M params would estimate ~600 MB at fp16; the real download is ~1 GB
      // because MusicGen ships a text encoder, a decoder and a vocoder.
      expect(m.bytes.wasm, m.id).toBeGreaterThan(0);
      expect(m.bytes.webgpu, m.id).toBeGreaterThan(0);
    }
  });

  it("always trips the size guardrail — this is the heaviest thing we ship", () => {
    for (const m of MUSIC_MODELS) {
      expect(sizeEstimate(m.params, m.bytes).large, m.id).toBe(true);
    }
  });
});

describe("tokensForSeconds", () => {
  it("scales with the requested duration", () => {
    expect(tokensForSeconds(1)).toBe(TOKENS_PER_SECOND);
    expect(tokensForSeconds(4)).toBe(4 * TOKENS_PER_SECOND);
  });

  it("keeps a floor so a 0-second request still produces audio", () => {
    expect(tokensForSeconds(0)).toBeGreaterThanOrEqual(16);
  });

  it("covers the route's slider range without overflowing", () => {
    expect(DEFAULT_SECONDS).toBeLessThanOrEqual(MAX_SECONDS);
    expect(tokensForSeconds(MAX_SECONDS)).toBe(MAX_SECONDS * TOKENS_PER_SECOND);
  });
});
