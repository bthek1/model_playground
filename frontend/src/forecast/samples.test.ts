import { describe, expect, it } from "vitest";

import { backtest } from "./backtest";
import { FORECAST_SAMPLES } from "./samples";
import { parseSeries } from "./series";

describe("the bundled series", () => {
  it("offers one for each thing the page has to demonstrate", () => {
    // airline: a real period, so the season control has something to be
    // visibly right and wrong about. retail: a level shift, which is what makes
    // the spread wide. random walk: nothing to learn, so MASE lands near 1.
    expect(FORECAST_SAMPLES.map((s) => s.id)).toEqual([
      "airline",
      "retail",
      "random-walk",
    ]);
  });

  it("labels the generated ones synthetic and states every licence", () => {
    for (const s of FORECAST_SAMPLES) {
      expect(s.licence.length, s.id).toBeGreaterThan(3);
      expect(s.source.length, s.id).toBeGreaterThan(3);
      expect(s.blurb.length, s.id).toBeGreaterThan(40);
      if (s.synthetic) expect(s.label, s.id).toMatch(/synthetic/i);
    }
    // The airline series has no licence attached to it, and saying so is more
    // honest than implying one.
    const airline = FORECAST_SAMPLES.find((s) => s.id === "airline");
    expect(airline?.licence).toMatch(/no explicit licence/i);
    expect(airline?.synthetic).toBeFalsy();
  });
});

describe("every series' own file", () => {
  it("parses, with dates, a detected frequency and no gaps", async () => {
    for (const s of FORECAST_SAMPLES) {
      const series = parseSeries(await s.load(), s.label);
      expect(series.times, s.id).not.toBeNull();
      expect(series.frequency, s.id).not.toBeNull();
      expect(series.gaps, s.id).toEqual([]);
      expect(series.irregular, s.id).toBe(false);
      expect(series.values.length, s.id).toBeGreaterThan(100);
    }
  });

  it("carries a season the series actually has, and a horizon it can afford", async () => {
    for (const s of FORECAST_SAMPLES) {
      const series = parseSeries(await s.load(), s.label);
      // The horizon must leave a backtest room to move its origin.
      expect(s.horizon, s.id).toBeGreaterThan(0);
      expect(s.horizon * 3, s.id).toBeLessThan(series.values.length);
      expect(s.season, s.id).toBeGreaterThan(1);
      expect(s.season, s.id).toBeLessThan(series.values.length / 3);
    }
  });

  it("agrees with the airline series' own detected period", async () => {
    // Monthly data implies a season of 12, and the entry says 12. If the file's
    // spacing ever changes, these must disagree loudly rather than quietly.
    const airline = FORECAST_SAMPLES.find((s) => s.id === "airline")!;
    const series = parseSeries(await airline.load(), airline.label);
    expect(series.frequency).toBe("monthly");
    expect(series.suggestedSeason).toBe(12);
    expect(airline.season).toBe(series.suggestedSeason);
  });

  it("gives the airline series a backtest spread far wider than one split", async () => {
    // **The page's whole claim, pinned to the sample that carries it.** A
    // backtest reusing one split's numbers for every window renders a flat strip
    // and passes any "a chart appeared" assertion; `just fe-e2e-forecast` uses
    // the same 1.5x threshold on the real page.
    const airline = FORECAST_SAMPLES.find((s) => s.id === "airline")!;
    const series = parseSeries(await airline.load(), airline.label);
    const result = backtest(series.values, "seasonal-naive", {
      horizon: airline.horizon,
      windows: 8,
      stride: Math.max(1, Math.round(airline.horizon / 2)),
      mode: "expanding",
      windowLength: 60,
      season: airline.season,
    });
    const maes = result.windows.map((w) => w.metrics.mae);
    expect(result.windows.length).toBe(8);
    expect(Math.max(...maes)).toBeGreaterThan(Math.min(...maes) * 1.5);
    // The single split is one of the windows, so it lies inside the range — the
    // reason the E2E assertion is the *width*, not "the spread straddles it".
    expect(result.single.mae).toBeGreaterThanOrEqual(Math.min(...maes));
    expect(result.single.mae).toBeLessThanOrEqual(Math.max(...maes));
  });

  it("gives the random walk a MASE near 1 at horizon 1 — and more beyond it", async () => {
    // **MASE near 1.0 is a one-step property, and this is where that bites.**
    // The denominator is the *in-sample one-step* naive error by definition, so a
    // multi-step forecast is compared against a one-step benchmark: on a random
    // walk the error grows as sqrt(h), and at horizon 14 the mean MASE is ~2.6
    // with nothing wrong. Reading "1.0 means no better than naive" at a horizon
    // of 14 and concluding the page is broken is the mistake available here, so
    // both halves are pinned and the metric table says which is which.
    const walk = FORECAST_SAMPLES.find((s) => s.id === "random-walk")!;
    const series = parseSeries(await walk.load(), walk.label);
    // Twenty origins three steps apart: at horizon 1 each window's error is a
    // single |dy| draw against the average of all of them, so the ratio is 1 only
    // in expectation and a handful of windows is too noisy to assert on.
    const options = {
      windows: 20,
      stride: 3,
      mode: "expanding" as const,
      windowLength: 60,
      season: 1,
    };
    const meanMase = (horizon: number) => {
      const mases = backtest(series.values, "naive", { ...options, horizon })
        .windows.map((w) => w.metrics.mase)
        .filter((m): m is number => m != null);
      return mases.reduce((a, b) => a + b, 0) / mases.length;
    };

    // One step ahead the forecast and the benchmark are the same estimator, so
    // the ratio sits at 1 — measured 1.08.
    expect(meanMase(1)).toBeGreaterThan(0.7);
    expect(meanMase(1)).toBeLessThan(1.4);
    // Further out, against the same one-step benchmark, it grows — measured
    // 1.77 at 4 and 2.98 at 14. Monotone, which is the shape being pinned.
    expect(meanMase(4)).toBeGreaterThan(meanMase(1));
    expect(meanMase(walk.horizon)).toBeGreaterThan(meanMase(4));
  });

  it("gives the retail series a real weekly period for seasonal naive to find", async () => {
    // Seasonal naive must clearly beat plain naive here, or the season control
    // has nothing to demonstrate on the sample that was built for it.
    const retail = FORECAST_SAMPLES.find((s) => s.id === "retail")!;
    const series = parseSeries(await retail.load(), retail.label);
    const options = {
      horizon: retail.horizon,
      windows: 6,
      stride: 7,
      mode: "expanding" as const,
      windowLength: 60,
      season: retail.season,
    };
    const seasonal = backtest(series.values, "seasonal-naive", options).single.mae;
    const naive = backtest(series.values, "naive", options).single.mae;
    expect(seasonal).toBeLessThan(naive);
  });
});
