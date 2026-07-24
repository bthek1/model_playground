import { describe, expect, it } from "vitest";

import { AUDIO_SAMPLES } from "./samples";

describe("AUDIO_SAMPLES", () => {
  it("lists the sanity-check clips with unique ids and https urls", () => {
    expect(AUDIO_SAMPLES.length).toBeGreaterThan(0);
    const ids = AUDIO_SAMPLES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const s of AUDIO_SAMPLES) {
      expect(s.url).toMatch(/^https:\/\/.+\.wav$/);
      expect(s.label).toBeTruthy();
      expect(s.hint).toBeTruthy();
    }
  });

  it("gives the short clips a reference transcript to compare against", () => {
    const jfk = AUDIO_SAMPLES.find((s) => s.id === "jfk");
    expect(jfk?.reference).toMatch(/ask not what your country/i);
  });
});
