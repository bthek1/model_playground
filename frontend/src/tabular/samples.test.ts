import { describe, expect, it } from "vitest";

import { parseCsv } from "./csv";
import { MAX_ROWS } from "./limits";
import { SAMPLES, sampleById } from "./samples";

describe("the bundled samples", () => {
  it("offers the synthetic one first, because it is the one that demonstrates", () => {
    // A sample every family gets right proves nothing about any of them, and is
    // satisfied by a page that fits one model and draws it four times. The
    // interaction sample is the demonstration; penguins and wine are controls.
    expect(SAMPLES[0].id).toBe("credit-risk");
    expect(SAMPLES.map((s) => s.id)).toEqual(["credit-risk", "penguins", "wine"]);
  });

  it("labels the generated one synthetic, everywhere it appears", () => {
    // Real data and generated data are different claims. The flag is what lets
    // the panel say so rather than the reader having to know.
    const synthetic = SAMPLES.filter((s) => s.synthetic);
    expect(synthetic.map((s) => s.id)).toEqual(["credit-risk"]);
    expect(synthetic[0].source).toMatch(/synthetic/i);
    expect(synthetic[0].label).toMatch(/synthetic/i);
  });

  it("states a licence and a source for every one", () => {
    // #24's lesson: a licence is the one thing that can invalidate a finished
    // route, and it is cheaper to read the card first.
    for (const s of SAMPLES) {
      expect(s.licence.length, s.id).toBeGreaterThan(3);
      expect(s.source.length, s.id).toBeGreaterThan(3);
      expect(s.blurb.length, s.id).toBeGreaterThan(40);
    }
  });

  it("finds a sample by id, and nothing by a wrong one", () => {
    expect(sampleById("wine")?.label).toBe("Wine recognition");
    expect(sampleById("iris")).toBeUndefined();
  });
});

describe("every sample's own file", () => {
  it("parses, with no rejected rows", async () => {
    // A bundled file that parses with issues would teach the parser's error
    // path rather than the model ladder, on the page's first click.
    for (const s of SAMPLES) {
      const { dataset, issues } = parseCsv(await s.load(), {
        name: s.label,
        maxRows: MAX_ROWS,
      });
      expect(issues, s.id).toEqual([]);
      expect(dataset.rowCount, s.id).toBeGreaterThan(100);
      expect(dataset.sampled, s.id).toBe(false);
    }
  });

  it("actually contains the target column each entry names, as a class column", () => {
    // The route preselects `sample.target`. A drifted name silently falls back
    // to "the first categorical column", which on penguins is the island.
    return Promise.all(
      SAMPLES.map(async (s) => {
        const { dataset } = parseCsv(await s.load());
        const target = dataset.columns.find((c) => c.name === s.target);
        expect(target, `${s.id}: ${s.target}`).toBeDefined();
        expect(target?.kind, s.id).toBe("categorical");
        // Two or more classes, and few enough to be a classification problem
        // rather than an id column wearing the wrong hat.
        expect(target?.levels?.length ?? 0, s.id).toBeGreaterThan(1);
        expect(target?.levels?.length ?? 0, s.id).toBeLessThan(12);
      }),
    );
  });

  it("contains the regression target each entry names, as a numeric column", () => {
    return Promise.all(
      SAMPLES.map(async (s) => {
        if (!s.regressionTarget) return;
        const { dataset } = parseCsv(await s.load());
        const target = dataset.columns.find((c) => c.name === s.regressionTarget);
        expect(target, `${s.id}: ${s.regressionTarget}`).toBeDefined();
        expect(target?.kind, s.id).toBe("numeric");
      }),
    );
  });

  it("leaves at least two usable features beside each target", () => {
    return Promise.all(
      SAMPLES.map(async (s) => {
        const { dataset } = parseCsv(await s.load());
        expect(dataset.columns.length, s.id).toBeGreaterThan(2);
      }),
    );
  });

  it("keeps the missing-value path exercised", async () => {
    // A sample set with no gaps never reaches the imputation, the is-missing
    // indicator or the drop-the-row rule — three things that fail silently.
    const withGaps: string[] = [];
    for (const s of SAMPLES) {
      const { dataset } = parseCsv(await s.load());
      if (dataset.columns.some((c) => c.missingCount > 0)) withGaps.push(s.id);
    }
    expect(withGaps.length).toBeGreaterThan(0);
  });

  it("gives the synthetic sample the interaction it claims, and a pure-noise column", () => {
    // The two things that make it the demonstration: `income` x `debt_ratio`
    // decides the label, and `postcode_noise` decides nothing — so permutation
    // importance has something to correctly rank near zero.
    return sampleById("credit-risk")!
      .load()
      .then((csv) => {
        const { dataset } = parseCsv(csv);
        const names = dataset.columns.map((c) => c.name);
        expect(names).toContain("income");
        expect(names).toContain("debt_ratio");
        expect(names).toContain("postcode_noise");
        expect(names).toContain("defaulted");
      });
  });
});
