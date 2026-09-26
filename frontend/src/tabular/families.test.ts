import { describe, expect, it } from "vitest";

import {
  FAMILIES,
  familiesFor,
  familyInfo,
  KNOB_LABELS,
  REGRESSION_FAMILIES,
} from "./families";
import { MAX_BOOST_DEPTH, MAX_TREE_DEPTH } from "./limits";
import type { Family, Hyperparams } from "./types";

const ALL = [...FAMILIES, ...REGRESSION_FAMILIES];

describe("the ladder", () => {
  it("offers a classification rung for each step of the argument", () => {
    // The ladder is an argument, not a menu: a floor to beat, a zero-tuning
    // baseline, the workhorse, and a deep model that loses. Dropping any of the
    // four removes a claim the page makes.
    expect(FAMILIES.map((f) => f.id)).toEqual([
      "logistic",
      "forest",
      "boosting",
      "mlp",
    ]);
  });

  it("offers the regression rungs as their own list, not a filtered view", () => {
    // Separate entries rather than an `objective` flag on the four above,
    // because the defaults are measured per page: a depth that suits a Gini
    // split is not automatically right for variance reduction.
    expect(REGRESSION_FAMILIES.map((f) => f.id)).toEqual([
      "ridge",
      "forest",
      "boosting",
      "quantile",
    ]);
    expect(familiesFor("classification")).toBe(FAMILIES);
    expect(familiesFor("regression")).toBe(REGRESSION_FAMILIES);
  });

  it("gives the shared tree families different defaults per objective", () => {
    // The `/link-prediction` lesson: reuse that looks like a decision is often
    // an inheritance. If these ever match exactly, someone has shared an object.
    const clsBoost = familyInfo("boosting", "classification").defaults;
    const regBoost = familyInfo("boosting", "regression").defaults;
    expect(regBoost).not.toEqual(clsBoost);
    expect(regBoost.nTrees).not.toBe(clsBoost.nTrees);
  });

  it("resolves a family by id within its own objective first", () => {
    expect(familyInfo("boosting", "regression").blurb).toContain("residuals");
    expect(familyInfo("forest", "classification").objectives).toEqual([
      "classification",
      "regression",
    ]);
  });

  it("still resolves a family that belongs only to the other objective", () => {
    // The route passes its own objective, but a persisted selection or a test
    // can ask for a rung the other list owns; falling back beats throwing.
    expect(familyInfo("ridge").id).toBe("ridge");
    expect(familyInfo("mlp", "regression").id).toBe("mlp");
  });

  it("throws on an unknown family rather than rendering an empty picker", () => {
    expect(() => familyInfo("gradient-descent" as Family)).toThrow(/Unknown model family/);
  });
});

describe("every entry", () => {
  it("declares where its arithmetic runs, and says why", () => {
    // The one sentence this category exists to say out loud. A rung with no
    // reason attached is a rung whose GPU/CPU choice reads as an accident.
    for (const f of ALL) {
      expect(["gpu", "cpu"], f.id).toContain(f.compute);
      expect(f.computeNote.length, f.id).toBeGreaterThan(40);
      expect(f.blurb.length, f.id).toBeGreaterThan(40);
    }
  });

  it("puts the trees on the CPU and the linear algebra on the GPU", () => {
    // Recursive splitting is branch-heavy and does not vectorise; a matmul is
    // what the hardware is for. Flipping either would make the page's own
    // explanation untrue.
    const compute = new Map(ALL.map((f) => [f.id, f.compute]));
    expect(compute.get("forest")).toBe("cpu");
    expect(compute.get("boosting")).toBe("cpu");
    expect(compute.get("logistic")).toBe("gpu");
    expect(compute.get("mlp")).toBe("gpu");
    expect(compute.get("ridge")).toBe("gpu");
    expect(compute.get("quantile")).toBe("gpu");
  });

  it("names only knobs it actually reads, and only knobs that have labels", () => {
    // A knob with no label renders as `undefined`; a knob the fitter ignores is
    // a control that silently does nothing, which is worse.
    for (const f of ALL) {
      expect(f.knobs.length, f.id).toBeGreaterThan(0);
      for (const knob of f.knobs) {
        expect(KNOB_LABELS[knob], `${f.id}/${knob}`).toBeDefined();
      }
    }
  });

  it("gives the tree families the tree knobs and the gradient families the gradient knobs", () => {
    const knobs = (id: Family, objective: "classification" | "regression") =>
      familyInfo(id, objective).knobs;
    expect(knobs("forest", "classification")).toContain("nTrees");
    expect(knobs("forest", "classification")).not.toContain("epochs");
    expect(knobs("logistic", "classification")).toContain("epochs");
    expect(knobs("logistic", "classification")).not.toContain("nTrees");
    // Ridge has no iterations at all — it is one solve.
    expect(knobs("ridge", "regression")).toEqual(["lambda"]);
  });

  it("carries a complete Hyperparams object, so no knob can render as undefined", () => {
    const keys = Object.keys(KNOB_LABELS) as (keyof Hyperparams)[];
    for (const f of ALL) {
      for (const key of keys) {
        expect(typeof f.defaults[key], `${f.id}/${key}`).toBe("number");
        expect(Number.isFinite(f.defaults[key]), `${f.id}/${key}`).toBe(true);
      }
    }
  });

  it("keeps every default inside its knob's own range", () => {
    // A default outside the slider's range snaps on first drag, which looks
    // like the page changing the user's settings by itself.
    for (const f of ALL) {
      for (const knob of f.knobs) {
        const meta = KNOB_LABELS[knob];
        const value = f.defaults[knob];
        expect(value, `${f.id}/${knob}`).toBeGreaterThanOrEqual(meta.min);
        expect(value, `${f.id}/${knob}`).toBeLessThanOrEqual(meta.max);
      }
    }
  });

  it("keeps boosting's default depth inside the cap Phase 0 measured", () => {
    // Phase 0: depth 6 costs 1.6x depth 4 at every size, and depth 8 would put
    // the worst case back over twenty seconds at the row cap.
    for (const objective of ["classification", "regression"] as const) {
      expect(familyInfo("boosting", objective).defaults.maxDepth).toBeLessThanOrEqual(
        MAX_BOOST_DEPTH,
      );
      expect(familyInfo("forest", objective).defaults.maxDepth).toBeLessThanOrEqual(
        MAX_TREE_DEPTH,
      );
    }
  });
});

describe("KNOB_LABELS", () => {
  it("gives every knob a label, a range and a reason", () => {
    for (const [knob, meta] of Object.entries(KNOB_LABELS)) {
      expect(meta.label.length, knob).toBeGreaterThan(2);
      expect(meta.max, knob).toBeGreaterThan(meta.min);
      expect(meta.step, knob).toBeGreaterThan(0);
      // The hint is what makes a slider a teaching surface rather than a dial.
      expect(meta.hint.length, knob).toBeGreaterThan(20);
    }
  });

  it("says that ridge's lambda is what makes the solve possible", () => {
    // Not a tuning knob bolted on: λ > 0 is what makes XᵀX + λI positive
    // definite, which is what makes the Cholesky factorisation legitimate.
    expect(KNOB_LABELS.lambda.hint).toMatch(/positive definite|rank-deficient/i);
    expect(KNOB_LABELS.lambda.min).toBe(0);
  });
});
