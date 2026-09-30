import { describe, expect, it } from "vitest";

import { backtest, noLeakage, planWindows, type BacktestOptions } from "./backtest";

const base: BacktestOptions = {
  horizon: 4,
  windows: 5,
  stride: 4,
  mode: "expanding",
  windowLength: 30,
  season: 1,
};

/** A series with an obvious trend, so windows differ from each other. */
function trending(n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = i * 2 + (i % 5);
  return out;
}

describe("backtest", () => {
  it("never lets a training index reach a test index — in any window", () => {
    // **The most valuable assertion in this module.** Shuffled cross-validation
    // on a time series trains on the future and scores the past, and it fails
    // *upward*: the metric improves and nothing throws.
    for (const mode of ["expanding", "sliding"] as const) {
      const result = backtest(trending(120), "naive", { ...base, mode, windows: 8 });
      expect(result.windows.length).toBeGreaterThan(0);
      for (const w of result.windows) {
        expect(noLeakage(w), JSON.stringify(w)).toBe(true);
        expect(w.trainEnd).toBeLessThanOrEqual(w.testStart);
      }
      expect(noLeakage(result.singleWindow)).toBe(true);
    }
  });

  it("produces the documented number of windows, ending at the series' end", () => {
    const result = backtest(trending(120), "naive", base);
    expect(result.windows).toHaveLength(5);
    const last = result.windows[result.windows.length - 1];
    expect(last.testEnd).toBe(120);
    // Origins step back by `stride` from the end, so the most recent — and most
    // relevant — window is always present.
    expect(result.windows[0].origin).toBe(120 - 4 - 4 * 4);
  });

  it("grows an expanding window's train set and does not grow a sliding one's", () => {
    const expanding = backtest(trending(120), "naive", base).windows;
    const sliding = backtest(trending(120), "naive", { ...base, mode: "sliding" }).windows;
    const size = (w: { trainStart: number; trainEnd: number }) => w.trainEnd - w.trainStart;
    expect(size(expanding[expanding.length - 1])).toBeGreaterThan(size(expanding[0]));
    expect(size(sliding[sliding.length - 1])).toBe(size(sliding[0]));
    expect(size(sliding[0])).toBe(base.windowLength);
  });

  it("reports the single split from the SAME call as the windows", () => {
    // If the two came from separate runs they could differ in the season, the
    // horizon or the method, and the gap the page draws would be attributed to
    // the splitting rather than to the settings.
    const result = backtest(trending(120), "seasonal-naive", { ...base, season: 5 });
    expect(result.singleWindow.testEnd).toBe(120);
    expect(result.singleWindow.trainEnd).toBe(120 - base.horizon);
    // It is the same window the last rolling origin produces at this stride.
    const last = result.windows[result.windows.length - 1];
    expect(last.origin).toBe(result.singleWindow.origin);
    expect(last.metrics.mae).toBeCloseTo(result.single.mae, 6);
  });

  it("says so when fewer windows fit than were asked for", () => {
    const result = backtest(trending(30), "naive", { ...base, windows: 20 });
    expect(result.windows.length).toBeLessThan(20);
    expect(result.note).toMatch(/of the 20 windows/);
  });

  it("refuses a series too short to hold anything out", () => {
    expect(() => backtest(trending(4), "naive", base)).toThrow(/at least 6 points/);
  });

  it("swings across windows on a series with a level shift", () => {
    // The page's whole argument, as a test. A single number cannot report this
    // and a page that only showed the spread would not make the point either.
    const n = 200;
    const values = new Float32Array(n);
    // The shift sits inside the range the origins cover, which is the whole
    // point: a run whose windows all fall on one side of it would report a
    // tight spread and demonstrate nothing.
    for (let i = 0; i < n; i++) values[i] = 100 + (i % 7) * 3 + (i > 150 ? 60 : 0);
    const result = backtest(values, "naive", { ...base, windows: 12, stride: 5 });
    const maes = result.windows.map((w) => w.metrics.mae);
    expect(Math.max(...maes)).toBeGreaterThan(Math.min(...maes) + 1);
  });
});

describe("planWindows", () => {
  it("drops a window with too little history rather than scoring it", () => {
    const planned = planWindows(20, { ...base, windows: 10, stride: 3 });
    for (const p of planned) expect(p.origin - p.trainStart).toBeGreaterThanOrEqual(2);
    expect(planned.every((p) => p.origin + base.horizon <= 20)).toBe(true);
  });
});
