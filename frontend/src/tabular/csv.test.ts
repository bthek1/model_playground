import { describe, expect, it } from "vitest";

import { cellText, parseCsv, splitLine } from "./csv";

describe("splitLine", () => {
  it("keeps commas inside quoted fields", () => {
    expect(splitLine('a,"b,c",d')).toEqual(["a", "b,c", "d"]);
  });

  it("reads a doubled quote as one literal quote", () => {
    expect(splitLine('"she said ""hi""",2')).toEqual(['she said "hi"', "2"]);
  });

  it("keeps an empty trailing field", () => {
    expect(splitLine("a,b,")).toEqual(["a", "b", ""]);
  });
});

describe("parseCsv", () => {
  it("parses a header and typed columns", () => {
    const { dataset } = parseCsv("a,b\n1,x\n2,y\n");
    expect(dataset.rowCount).toBe(2);
    expect(dataset.columns[0].kind).toBe("numeric");
    expect(Array.from(dataset.columns[0].values)).toEqual([1, 2]);
    expect(dataset.columns[1].kind).toBe("categorical");
    expect(dataset.columns[1].levels).toEqual(["x", "y"]);
  });

  it("survives a newline inside a quoted field", () => {
    // Splitting on "\n" first and reassembling later turns this one row into
    // two ragged ones, which the parser would then correctly reject — a
    // convincing-looking failure on a perfectly valid file.
    const { dataset, issues } = parseCsv('name,note\n"Ada","line one\nline two"\n');
    expect(issues).toEqual([]);
    expect(dataset.rowCount).toBe(1);
    expect(dataset.columns[1].levels?.[0]).toBe("line one\nline two");
  });

  it("treats CRLF as one break and strips a BOM", () => {
    const { dataset } = parseCsv("﻿a,b\r\n1,2\r\n3,4\r\n");
    expect(dataset.columns[0].name).toBe("a");
    expect(dataset.rowCount).toBe(2);
  });

  it("rejects a ragged row with its line number, and keeps the rest", () => {
    const { dataset, issues } = parseCsv("a,b\n1,2\n3\n5,6\n");
    expect(dataset.rowCount).toBe(2);
    expect(issues).toHaveLength(1);
    // The line number is the point: a best-effort pad would fit a model on
    // shifted columns, so the user is told where to look instead.
    expect(issues[0].line).toBe(3);
  });

  it("stays numeric when a decimal only appears late in the column", () => {
    const rows = Array.from({ length: 900 }, (_, i) => `${i}`);
    rows.push("900.5");
    const { dataset } = parseCsv(`n\n${rows.join("\n")}\n`);
    expect(dataset.columns[0].kind).toBe("numeric");
    expect(dataset.columns[0].values[900]).toBeCloseTo(900.5);
  });

  it("becomes categorical when any cell is not a number", () => {
    const rows = Array.from({ length: 900 }, (_, i) => `${i}`);
    rows.push("unknown");
    const { dataset } = parseCsv(`n\n${rows.join("\n")}\n`);
    expect(dataset.columns[0].kind).toBe("categorical");
  });

  it("distinguishes a missing value from 0", () => {
    // `Number("")` is 0, which is the coercion that silently shifts every mean
    // and every split threshold a column touches.
    const { dataset } = parseCsv("v\n0\n\n2\n");
    const col = dataset.columns[0];
    expect(Array.from(col.missing)).toEqual([0, 1, 0]);
    expect(col.missingCount).toBe(1);
    expect(col.values[0]).toBe(0);
  });

  it("treats NA as missing rather than as a level", () => {
    const { dataset } = parseCsv("v\n1\nNA\n3\n");
    expect(dataset.columns[0].kind).toBe("numeric");
    expect(dataset.columns[0].missingCount).toBe(1);
  });

  it("disambiguates duplicate header names", () => {
    const { dataset } = parseCsv("x,x\n1,2\n");
    expect(dataset.columns.map((c) => c.name)).toEqual(["x", "x (2)"]);
  });

  it("samples evenly and reports both counts when the cap bites", () => {
    const rows = Array.from({ length: 100 }, (_, i) => `${i}`);
    const { dataset } = parseCsv(`n\n${rows.join("\n")}\n`, { maxRows: 10 });
    expect(dataset.rowCount).toBe(10);
    expect(dataset.sourceRowCount).toBe(100);
    expect(dataset.sampled).toBe(true);
    // Evenly across the file, not the first ten: a file sorted by date would
    // otherwise become a different dataset without saying so.
    expect(dataset.columns[0].values[9]).toBe(90);
  });

  it("throws on an empty file", () => {
    expect(() => parseCsv("")).toThrow(/empty/i);
  });

  it("throws when every row is ragged, naming the first line", () => {
    expect(() => parseCsv("a,b\n1\n2\n")).toThrow(/line 2/);
  });
});

describe("cellText", () => {
  it("renders a level for a categorical cell and blank for a missing one", () => {
    const { dataset } = parseCsv("c\nx\n\n");
    expect(cellText(dataset.columns[0], 0)).toBe("x");
    expect(cellText(dataset.columns[0], 1)).toBe("");
  });
});
