import { describe, expect, it } from "vitest";

import { DEFAULT_SPEED, MAX_EPISODES, SPEEDS } from "./limits";
import { PG_DEFAULTS } from "./policyGradient";

describe("rl/limits", () => {
  it("offers the default speed on the dial, slow to fast, ending at 'as fast as the CPU goes'", () => {
    expect(SPEEDS).toContain(DEFAULT_SPEED);
    expect(SPEEDS[SPEEDS.length - 1]).toBeNull();
    const finite = SPEEDS.filter((s): s is number => s != null);
    expect(finite).toEqual([...finite].sort((a, b) => a - b));
  });

  it("caps episodes above what the 8×8 grid needs to find its goal", () => {
    // The page's own default for the 8×8 is 20 000 (see routes/rl.tsx).
    expect(MAX_EPISODES).toBeGreaterThanOrEqual(20_000);
    expect(PG_DEFAULTS.episodes).toBeLessThanOrEqual(MAX_EPISODES);
  });
});
