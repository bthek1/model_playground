import { describe, expect, it } from "vitest";

import { parseCsv } from "./csv";
import { encodeOne, encodeRows, fitEncoder, MAX_LEVELS } from "./design";

const rows = (n: number) => Int32Array.from({ length: n }, (_, i) => i);

function frame(csv: string) {
  return parseCsv(csv).dataset;
}

describe("fitEncoder", () => {
  it("one-hot encodes a categorical column", () => {
    const d = frame("city,n\nx,1\ny,2\nx,3\n");
    const e = fitEncoder(d, [0], rows(3));
    expect(e.columns.map((c) => c.name)).toEqual(["city = x", "city = y"]);
    const m = encodeRows(d, e, rows(3), false);
    expect(Array.from(m)).toEqual([1, 0, 0, 1, 1, 0]);
  });

  it("adds an explicit is-missing indicator to a numeric column with gaps", () => {
    // Imputing silently throws away the one thing a missing cell reliably
    // carries — that it was missing — and on plenty of real files that is the
    // strongest feature in the frame.
    const d = frame("v\n1\n\n3\n");
    const e = fitEncoder(d, [0], rows(3));
    expect(e.columns.map((c) => c.name)).toEqual(["v", "v is missing"]);
    const m = encodeRows(d, e, rows(3), false);
    expect(m[2]).toBeCloseTo(2); // imputed with the training mean of 1 and 3
    expect(m[3]).toBe(1);
  });

  it("imputes with the mean of the PRESENT cells, not of the encoded zeros", () => {
    const d = frame("v\n10\n20\n\n");
    const e = fitEncoder(d, [0], rows(3));
    const m = encodeRows(d, e, rows(3), false);
    expect(m[4]).toBeCloseTo(15);
  });

  it("fits its statistics on the training rows only", () => {
    // The whole point of taking `train` rather than the frame: an encoding
    // fitted over the held-out rows raises the held-out score and looks like a
    // better page. `/link-prediction` in its cheapest form.
    const d = frame("v\n0\n1\n0\n1\n100\n");
    const train = Int32Array.from([0, 1, 2, 3]);
    const e = fitEncoder(d, [0], train);
    expect(e.mean[0]).toBeCloseTo(0.5);
    const standardised = encodeRows(d, e, Int32Array.from([4]), true);
    // The held-out outlier is far from the training mean, as it should be — an
    // encoder that had seen it would have pulled the mean up to 20 and the
    // standard deviation up with it, shrinking this to under 2.
    expect(standardised[0]).toBeGreaterThan(10);
  });

  it("drops a high-cardinality column with its reason rather than encoding it", () => {
    const values = Array.from({ length: MAX_LEVELS + 5 }, (_, i) => `id${i}`);
    const d = frame(`k,n\n${values.map((v, i) => `${v},${i}`).join("\n")}\n`);
    const e = fitEncoder(d, [0, 1], rows(values.length));
    expect(e.used).toEqual([1]);
    expect(e.dropped[0].name).toBe("k");
    expect(e.dropped[0].reason).toMatch(/one-hot/);
  });

  it("drops a constant column", () => {
    const d = frame("c,v\n1,5\n1,6\n1,7\n");
    const e = fitEncoder(d, [0, 1], rows(3));
    expect(e.used).toEqual([1]);
    expect(e.dropped[0].reason).toMatch(/constant/);
  });

  it("throws when nothing is left to fit on", () => {
    const d = frame("c\n1\n1\n1\n");
    expect(() => fitEncoder(d, [0], rows(3))).toThrow(/usable feature/i);
  });

  it("standardises to zero mean and unit variance on the training rows", () => {
    const d = frame("v\n1\n2\n3\n4\n");
    const e = fitEncoder(d, [0], rows(4));
    const m = encodeRows(d, e, rows(4), true);
    const mean = m.reduce((a, b) => a + b, 0) / 4;
    expect(mean).toBeCloseTo(0, 5);
    const sd = Math.sqrt(m.reduce((a, b) => a + b * b, 0) / 4);
    expect(sd).toBeCloseTo(1, 4);
  });
});

describe("encodeOne", () => {
  it("encodes a typed row the same way as the matrix", () => {
    const d = frame("city,v\nx,1\ny,3\n");
    const e = fitEncoder(d, [0, 1], rows(2));
    const one = encodeOne(d, e, ["y", 3], [0, 1], false);
    const all = encodeRows(d, e, rows(2), false);
    const width = e.columns.length;
    expect(Array.from(one)).toEqual(Array.from(all.subarray(width, 2 * width)));
  });

  it("treats an unreadable cell as missing rather than as zero", () => {
    const d = frame("v\n10\n20\n\n");
    const e = fitEncoder(d, [0], rows(3));
    const one = encodeOne(d, e, [""], [0], false);
    expect(one[0]).toBeCloseTo(15);
    expect(one[1]).toBe(1);
  });
});
