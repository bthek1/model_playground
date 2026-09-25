// The bundled series.
//
// Chosen for what the page has to demonstrate rather than for realism, and each
// one is here because a different sample would let a broken page pass:
//
//   airline     a real, unmistakable yearly period, so the season-length control
//               has something to be visibly right and wrong about
//   retail      a **level shift** two thirds of the way through, which is what
//               puts the single-split number inside the backtest spread instead
//               of beside it
//   random walk nothing to learn, so every method lands near MASE 1.0 — which is
//               what makes that number readable
//
// Loaded by dynamic `import()` for the same reason `tabular/samples.ts` is: raw
// CSV text is incompressible and a static import puts every byte in the entry
// chunk, which `npm run check:bundle` fails on.

export interface ForecastSample {
  id: string;
  label: string;
  blurb: string;
  load: () => Promise<string>;
  licence: string;
  source: string;
  /** The season the series actually has, preselected. */
  season: number;
  /** A sensible starting horizon for this series' length. */
  horizon: number;
  synthetic?: boolean;
}

export const FORECAST_SAMPLES: ForecastSample[] = [
  {
    id: "airline",
    label: "Airline passengers",
    blurb:
      "144 monthly totals, 1949–1960. Trend plus an unmistakable yearly period — set the season to 12 and seasonal naive is excellent; set it to 11 and the forecast is confidently off by a phase, with nothing on screen to say so.",
    load: () => import("./data/airline.csv?raw").then((m) => m.default),
    licence:
      "No explicit licence — 144 integers published in a 1976 textbook and redistributed in R, statsmodels and Keras.",
    source: "Box, Jenkins & Reinsel (1976)",
    season: 12,
    horizon: 12,
  },
  {
    id: "retail",
    label: "Daily units (synthetic)",
    blurb:
      "420 days with a weekly period and a level shift two thirds of the way through. The shift is the point: a single split that falls after it reports a flattering number, and the windows that straddle it do not.",
    load: () => import("./data/retail-synthetic.csv?raw").then((m) => m.default),
    licence: "Generated in this repo — see scripts/make-tabular-samples.mjs",
    source: "Synthetic, fixed seed",
    season: 7,
    horizon: 14,
    synthetic: true,
  },
  {
    id: "random-walk",
    label: "Random walk (synthetic)",
    blurb:
      "300 days of pure noise accumulating. There is nothing to learn, so every method lands near MASE 1.0 — which is exactly what that number is for.",
    load: () => import("./data/random-walk-synthetic.csv?raw").then((m) => m.default),
    licence: "Generated in this repo — see scripts/make-tabular-samples.mjs",
    source: "Synthetic, fixed seed",
    season: 7,
    horizon: 14,
    synthetic: true,
  },
];
