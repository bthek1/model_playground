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
// Usage:  node scripts/make-tabular-samples.mjs

import { writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = join(import.meta.dirname, "..", "src", "tabular", "data");

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

const n1 = await penguins();
const n2 = await wine();
const n3 = creditRisk();
console.log(`penguins.csv ${n1} rows · wine.csv ${n2} rows · credit-risk-synthetic.csv ${n3} rows`);
