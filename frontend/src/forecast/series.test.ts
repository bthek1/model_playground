import { describe, expect, it } from "vitest";

import { describe as describeSeries, futureTimes, parseSeries, pointLabel } from "./series";

describe("parseSeries", () => {
  it("reads a bare column of numbers", () => {
    const s = parseSeries("1\n2\n3\n4\n");
    expect(Array.from(s.values)).toEqual([1, 2, 3, 4]);
    expect(s.times).toBeNull();
    expect(s.frequency).toBeNull();
  });

  it("reads date,value pairs and detects the frequency", () => {
    const rows = Array.from(
      { length: 24 },
      (_, i) => `2024-${String((i % 12) + 1).padStart(2, "0")}-01,${i}`,
    );
    const s = parseSeries(rows.join("\n"));
    expect(s.values.length).toBe(24);
    expect(s.frequency).toBe("monthly");
    expect(s.suggestedSeason).toBe(12);
  });

  it("parses a date,value CSV and a bare column to the same values", () => {
    const bare = parseSeries("5\n6\n7\n");
    const dated = parseSeries("date,v\n2024-01-01,5\n2024-01-02,6\n2024-01-03,7\n");
    expect(Array.from(dated.values)).toEqual(Array.from(bare.values));
    expect(dated.frequency).toBe("daily");
  });

  it("reports a gap rather than filling it", () => {
    // A silently interpolated gap gives a seasonal naive forecast that is
    // confidently off by a phase, and the error is then blamed on the method.
    const s = parseSeries(
      ["2024-01-01,1", "2024-01-02,2", "2024-01-05,3", "2024-01-06,4"].join("\n"),
    );
    expect(s.values.length).toBe(4);
    expect(s.gaps).toEqual([{ index: 1, missing: 2 }]);
    expect(s.irregular).toBe(false);
  });

  it("flags irregular spacing instead of pretending to a frequency", () => {
    const s = parseSeries(
      [
        "2024-01-01T00:00:00Z,1",
        "2024-01-01T01:00:00Z,2",
        "2024-01-01T01:37:00Z,3",
        "2024-01-01T02:37:00Z,4",
      ].join("\n"),
    );
    expect(s.irregular).toBe(true);
  });

  it("skips a header line", () => {
    const s = parseSeries("date,value\n2024-01-01,10\n2024-01-02,11\n2024-01-03,12\n");
    expect(Array.from(s.values)).toEqual([10, 11, 12]);
  });

  it("names the line a non-numeric value is on", () => {
    expect(() => parseSeries("1\n2\noops\n4\n")).toThrow(/Line 3/);
  });

  it("refuses a series too short to forecast from", () => {
    expect(() => parseSeries("1\n2\n")).toThrow(/three points/);
    expect(() => parseSeries("")).toThrow(/column of numbers/);
  });
});

describe("describe", () => {
  it("uses the median step, so one long gap does not make everything irregular", () => {
    const day = 86_400_000;
    const times = Float64Array.from([0, day, 2 * day, 40 * day, 41 * day, 42 * day]);
    const s = describeSeries("s", Float32Array.from([1, 2, 3, 4, 5, 6]), times);
    expect(s.frequency).toBe("daily");
    expect(s.irregular).toBe(false);
    expect(s.gaps).toHaveLength(1);
  });
});

describe("pointLabel and futureTimes", () => {
  it("labels by index without times, and by date with them", () => {
    const bare = parseSeries("1\n2\n3\n");
    expect(pointLabel(bare, 0)).toBe("1");
    const dated = parseSeries("2024-03-04,1\n2024-03-05,2\n2024-03-06,3\n");
    expect(pointLabel(dated, 0)).toBe("2024-03-04");
  });

  it("extends the axis by the detected step", () => {
    const s = parseSeries("2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n");
    const future = futureTimes(s, 2);
    expect(future).not.toBeNull();
    expect(new Date(future![1]).toISOString().slice(0, 10)).toBe("2024-01-05");
    expect(futureTimes(parseSeries("1\n2\n3\n"), 2)).toBeNull();
  });
});
