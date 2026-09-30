#!/usr/bin/env node
// Rebuilds the bundled sample CSVs in `src/tabular/data/`.
//
// Two of the three are real datasets, fetched from their canonical homes and
// committed so the page works offline and with no backend; the third is
// generated here, and the reason it exists is a requirement rather than a
// convenience — see below.
//
//   penguins.csv   Palmer Archipelago penguins (Horst, Hill & Gorman 2020).
//                  Licence: CC0 1.0 (public domain dedication). Real missing
//                  values, a categorical feature and a three-class target.
//   wine.csv       UCI Wine recognition (Forina et al., 1991).
//                  Licence: CC BY 4.0. Thirteen numeric chemistry columns,
//                  three cultivars, and very nearly linearly separable — the
//                  ladder's floor wins on it, which is the point of having it.
//   credit-risk-synthetic.csv
//                  Generated here with a fixed seed. Real public tables this
//                  small are almost all separable, and a sample every family
//                  gets right demonstrates nothing about any of them — it is
//                  satisfied by a page that fits one model and draws it four
//                  times. So this one is built around an **interaction** a
//                  linear boundary cannot represent, and is labelled synthetic
//                  everywhere it appears.
//
// It also builds `src/forecast/data/`, whose three series are chosen for what
// the forecasting page has to demonstrate rather than for realism:
//
//   airline.csv    International airline passengers, 1949-1960 (Box, Jenkins &
//                  Reinsel). The canonical monthly series — trend plus an
//                  unmistakable yearly period, so the season-length control has
//                  something to be visibly right and wrong about. No explicit
//                  licence is attached to it; it is 144 integers published in a
//                  1976 textbook and redistributed in R, statsmodels and Keras.
//   retail-synthetic.csv
//                  Daily, weekly period, and a **level shift two thirds of the
//                  way through**. The shift is the point: it is what makes the
//                  single-split number land inside the backtest spread instead
//                  of beside it, which is the page's whole argument.
//   random-walk-synthetic.csv
//                  Nothing to learn. Every method lands near MASE 1.0, which is
//                  what makes that number readable as "no better than naive".
//
// Usage:  node scripts/make-tabular-samples.mjs

import { writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "src", "tabular", "data");
const FORECAST_OUT = join(import.meta.dirname, "..", "src", "forecast", "data");

const AIRLINE =
  "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv";
const PENGUINS =
  "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv";
const WINE =
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.text();
}

async function penguins() {
  const text = await fetchText(PENGUINS);
  writeFileSync(join(OUT, "penguins.csv"), text.trimEnd() + "\n");
  return text.trimEnd().split("\n").length - 1;
}

const WINE_COLUMNS = [
  "cultivar",
  "alcohol",
  "malic_acid",
  "ash",
  "alcalinity_of_ash",
  "magnesium",
  "total_phenols",
  "flavanoids",
  "nonflavanoid_phenols",
  "proanthocyanins",
  "colour_intensity",
  "hue",
  "od280_od315",
  "proline",
];
// The file ships the target as 1/2/3. Left as integers it would be inferred
// numeric, and a numeric column is not offered as a classification target — so
// the names go in here, where the mapping can be seen, rather than as a special
// case in the parser.
const CULTIVARS = { 1: "Barolo", 2: "Grignolino", 3: "Barbera" };

async function wine() {
  const text = await fetchText(WINE);
  const rows = text
    .trimEnd()
    .split("\n")
    .map((line) => {
      const parts = line.split(",");
      return [CULTIVARS[Number(parts[0])], ...parts.slice(1)].join(",");
    });
  writeFileSync(join(OUT, "wine.csv"), [WINE_COLUMNS.join(","), ...rows].join("\n") + "\n");
  return rows.length;
}

/** The repo's own PRNG, so the file is reproducible from the seed alone. */
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  const u = 1 - rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/**
 * A synthetic loan book whose risk is an interaction, not a slope.
 *
 * `debt_ratio` is dangerous only on a low income and protective on a high one.
 * That is a product term, so no linear boundary in the raw columns can express
 * it and the ladder's floor is held to roughly the class prior plus the small
 * linear part, while a depth-4 tree finds it in one split pair. Three columns
 * are pure noise, so permutation importance has something to correctly rank
 * near zero.
 */
function creditRisk(rows = 1800, seed = 20260925) {
  const rand = mulberry32(seed);
  const regions = ["north", "south", "coastal", "inland"];
  const out = [
    [
      "age",
      "income",
      "debt_ratio",
      "employment_years",
      "region",
      "has_mortgage",
      "enquiries_6m",
      "postcode_noise",
      "defaulted",
    ].join(","),
  ];

  for (let i = 0; i < rows; i++) {
    const age = Math.round(22 + rand() * 45);
    const income = Math.round(22000 + Math.abs(gaussian(rand)) * 26000);
    const debtRatio = Number(Math.min(0.95, Math.max(0.02, 0.28 + gaussian(rand) * 0.18)).toFixed(3));
    const employment = Number(Math.max(0, Math.min(age - 20, gaussian(rand) * 4 + 6)).toFixed(1));
    const region = regions[Math.floor(rand() * regions.length)];
    const mortgage = rand() < 0.42 ? "yes" : "no";
    const enquiries = Math.floor(rand() * 6);
    const noise = Math.round(rand() * 9999);

    const lowIncome = income < 45000;
    const highDebt = debtRatio > 0.38;
    // The interaction: leveraged on a low income is the risk, leveraged on a
    // high income is not. A linear model has to pick one sign for `debt_ratio`.
    const interaction = lowIncome === highDebt ? 1 : 0;
    const logit =
      -1.15 +
      2.35 * interaction +
      0.55 * (employment < 2 ? 1 : 0) +
      0.22 * enquiries -
      0.45 * (mortgage === "yes" ? 1 : 0);
    const p = 1 / (1 + Math.exp(-logit));
    const defaulted = rand() < p ? "yes" : "no";

    // A handful of genuinely missing incomes, because a sample with no gaps
    // never exercises the missing-value path the parser is careful about.
    const incomeCell = rand() < 0.03 ? "" : String(income);
    out.push(
      [age, incomeCell, debtRatio, employment, region, mortgage, enquiries, noise, defaulted].join(","),
    );
  }
  writeFileSync(join(OUT, "credit-risk-synthetic.csv"), out.join("\n") + "\n");
  return rows;
}

/**
 * The canonical monthly series: international airline passengers, 1949–1960.
 *
 * Trend plus an unmistakable yearly period, which is what makes it the right
 * sample for a page whose season length is a *control*: set it to 12 and
 * seasonal naive is excellent, set it to 11 and the forecast is confidently off
 * by a phase with nothing on screen to say so.
 */
async function airline() {
  const text = await fetchText(AIRLINE);
  const rows = text
    .trimEnd()
    .split("\n")
    .slice(1)
    .map((line) => line.replace(/"/g, ""))
    .filter((line) => line.length > 0)
    // Month resolution only; the page's date parser wants a full date.
    .map((line) => {
      const [month, value] = line.split(",");
      return `${month}-01,${value}`;
    });
  writeFileSync(join(FORECAST_OUT, "airline.csv"), ["date,passengers", ...rows].join("\n") + "\n");
  return rows.length;
}

/**
 * A daily series with a weekly period **and a level shift two thirds of the way
 * through**.
 *
 * The shift is the point. A single train/test split that falls after it reports
 * one number; windows that straddle it report much worse ones — so the
 * single-split metric sits *inside* the spread rather than next to it, which is
 * the page's entire argument made visible. A series without a regime change
 * would give a tight spread and demonstrate nothing.
 */
function retail(days = 420, seed = 20260925) {
  const rand = mulberry32(seed);
  const start = Date.UTC(2024, 0, 1);
  const weekly = [1.18, 0.92, 0.9, 0.95, 1.05, 1.25, 0.75];
  const out = ["date,units"];
  for (let i = 0; i < days; i++) {
    const level = 200 + 0.22 * i + (i > days * 0.66 ? 70 : 0);
    const value = level * weekly[i % 7] * (1 + gaussian(rand) * 0.045);
    const date = new Date(start + i * 86400000).toISOString().slice(0, 10);
    out.push(`${date},${Math.round(value)}`);
  }
  writeFileSync(join(FORECAST_OUT, "retail-synthetic.csv"), out.join("\n") + "\n");
  return days;
}

/**
 * A pure random walk — no trend, no season, nothing to learn.
 *
 * It is here so the page can show MASE doing its job: on a random walk nothing
 * beats "the last value, repeated", so every method lands near 1.0 and the
 * number is readable as "no better than naive" rather than as a small error.
 */
function randomWalk(days = 300, seed = 4242) {
  const rand = mulberry32(seed);
  const start = Date.UTC(2024, 0, 1);
  const out = ["date,price"];
  let v = 100;
  for (let i = 0; i < days; i++) {
    v += gaussian(rand) * 1.4;
    const date = new Date(start + i * 86400000).toISOString().slice(0, 10);
    out.push(`${date},${v.toFixed(2)}`);
  }
  writeFileSync(join(FORECAST_OUT, "random-walk-synthetic.csv"), out.join("\n") + "\n");
  return days;
}

const n1 = await penguins();
const n2 = await wine();
const n3 = creditRisk();
const n4 = await airline();
const n5 = retail();
const n6 = randomWalk();
console.log(
  `tabular: penguins.csv ${n1} · wine.csv ${n2} · credit-risk-synthetic.csv ${n3}\n` +
    `forecast: airline.csv ${n4} · retail-synthetic.csv ${n5} · random-walk-synthetic.csv ${n6}`,
);
