import { describe, expect, it } from "vitest";

import { familyInfo } from "./families";
import {
  describeDuration,
  estimateFitMs,
  MAX_BOOST_DEPTH,
  MAX_ROWS,
  MAX_TREE_DEPTH,
} from "./limits";

const hp = (over: Partial<Parameters<typeof estimateFitMs>[2]> = {}) => ({
  nTrees: 60,
  maxDepth: 8,
  epochs: 30,
  hidden: 48,
  ...over,
});

describe("the caps", () => {
  it("holds the row cap Phase 0 set, and the depth caps that came with it", () => {
    // These are measurements, not preferences — the table in limits.ts is the
    // evidence. Raising MAX_ROWS without re-measuring puts the worst case on the
    // ladder past what a progress bar and a Stop button make honest.
    expect(MAX_ROWS).toBe(50_000);
    expect(MAX_BOOST_DEPTH).toBe(6);
    expect(MAX_TREE_DEPTH).toBeGreaterThan(MAX_BOOST_DEPTH);
  });

  it("caps boosting lower than a bagged tree, because it fits far more trees", () => {
    // A forest fits nTrees; a booster fits nTrees x classes. The asymmetry is
    // the reason the two caps differ, so they must not converge.
    expect(MAX_BOOST_DEPTH).toBeLessThan(MAX_TREE_DEPTH);
  });
});

describe("estimateFitMs", () => {
  it("reproduces Phase 0's own measurements at the sizes it measured", () => {
    // The estimate is an interpolation of the table in limits.ts. Anchoring it
    // to that table is what keeps the number on screen honest: a page that
    // quotes "about 3 s" and takes thirty is worse than one that says nothing.
    expect(estimateFitMs("forest", 10_000, hp({ nTrees: 60, maxDepth: 8 }))).toBeCloseTo(1000, -2);
    expect(estimateFitMs("boosting", 10_000, hp({ nTrees: 120, maxDepth: 4 }))).toBeCloseTo(1700, -2);
    expect(estimateFitMs("logistic", 10_000, hp({ epochs: 30 }))).toBeCloseTo(140, -1);
    expect(estimateFitMs("mlp", 10_000, hp({ epochs: 40, hidden: 48 }))).toBeCloseTo(1850, -2);
  });

  it("is linear in the row count", () => {
    const small = estimateFitMs("forest", 10_000, hp());
    const large = estimateFitMs("forest", 50_000, hp());
    expect(large / small).toBeCloseTo(5, 3);
  });

  it("grows with the tree count and with the depth", () => {
    expect(estimateFitMs("forest", 10_000, hp({ nTrees: 120 }))).toBeGreaterThan(
      estimateFitMs("forest", 10_000, hp({ nTrees: 60 })),
    );
    expect(estimateFitMs("boosting", 10_000, hp({ maxDepth: 6 }))).toBeGreaterThan(
      estimateFitMs("boosting", 10_000, hp({ maxDepth: 4 })),
    );
  });

  it("charges boosting per class, because it fits one tree per class per round", () => {
    const binary = estimateFitMs("boosting", 10_000, hp(), 2);
    const ten = estimateFitMs("boosting", 10_000, hp(), 10);
    expect(ten / binary).toBeCloseTo(5, 3);
  });

  it("reaches the number that set the row cap", () => {
    // Phase 0's condemning measurement: boosting at depth 6 on 200k rows is
    // ~51 s, which is why MAX_ROWS is 50 000 rather than 200 000.
    const worst = estimateFitMs("boosting", 200_000, hp({ nTrees: 120, maxDepth: 6 }), 2);
    expect(worst).toBeGreaterThan(40_000);
    // …and at the cap the worst case is seconds, not a minute.
    const atCap = estimateFitMs("boosting", MAX_ROWS, hp({ nTrees: 120, maxDepth: 6 }), 2);
    expect(atCap).toBeLessThan(20_000);
  });

  it("keeps every shipped default under ten seconds at the row cap", () => {
    // The promise the caps are there to keep. If a default ever breaks this, it
    // is the default that is wrong, not the assertion.
    for (const objective of ["classification", "regression"] as const) {
      for (const f of objective === "classification"
        ? (["logistic", "forest", "boosting", "mlp"] as const)
        : (["forest", "boosting"] as const)) {
        const info = familyInfo(f, objective);
        const ms = estimateFitMs(f, MAX_ROWS, info.defaults, 2);
        expect(ms, `${objective}/${f}`).toBeLessThan(15_000);
      }
    }
  });

  it("falls back to a plausible figure for a family it has no curve for", () => {
    // `ridge` and `quantile` are not in the table (one solve, and a gradient
    // loop over three lines). A missing case must not render "NaN".
    for (const family of ["ridge", "quantile", "future-rung"]) {
      const ms = estimateFitMs(family, 10_000, hp());
      expect(Number.isFinite(ms), family).toBe(true);
      expect(ms, family).toBeGreaterThan(0);
    }
  });
});

describe("describeDuration", () => {
  it("rounds hard, because the estimate has no precision to spend", () => {
    expect(describeDuration(120)).toBe("under a second");
    expect(describeDuration(899)).toBe("under a second");
    expect(describeDuration(3200)).toBe("about 3 s");
    expect(describeDuration(11_800)).toBe("about 12 s");
    expect(describeDuration(125_000)).toBe("about 2 min");
  });

  it("never renders a bare number, so the estimate cannot read as a measurement", () => {
    for (const ms of [0, 1, 999, 1000, 59_999, 60_000, 600_000]) {
      expect(describeDuration(ms)).toMatch(/^(under a second|about \d+ (s|min))$/);
    }
  });
});
