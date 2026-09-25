// The design matrix: raw columns in, one dense `Float32Array` out.
//
// Every encoding decision in here is **fitted on the training rows only**, and
// that is the file's reason to exist rather than a pass of inline code in the
// engine. A mean imputed from the whole frame, or a standardisation whose
// variance saw the held-out rows, raises the held-out score and looks like a
// better page — the `/link-prediction` failure in its cheapest form. Passing
// `train` in explicitly is what makes the rule impossible to forget.

import type { Column, Dataset } from "./types";

/** A one-hot column beyond this cardinality is an id column with extra steps. */
export const MAX_LEVELS = 24;

export interface DesignColumn {
  /** Human name, for the coefficient and importance bars. */
  name: string;
  /** Index into `Dataset.columns` this came from — importance groups by it. */
  source: number;
}

export interface Encoder {
  columns: DesignColumn[];
  /** Feature indices actually used, after dropping what could not be encoded. */
  used: number[];
  /** Source columns dropped, with the reason, for the dataset panel. */
  dropped: { name: string; reason: string }[];
  /** Per design column: mean and standard deviation over the training rows. */
  mean: Float32Array;
  std: Float32Array;
}

/**
 * Decide the encoding from the training rows.
 *
 * Numeric columns become themselves, plus — where the column has any missing
 * cell — an explicit `is missing` indicator. Imputing silently with the mean
 * throws away the one piece of information a missing cell reliably carries,
 * which is that it was missing; on plenty of real files that indicator is the
 * strongest feature in the frame.
 *
 * Categorical columns become one binary column per level, and a column with
 * more than `MAX_LEVELS` levels is **dropped with its reason on screen** rather
 * than encoded. One-hot encoding a customer id gives a model that scores
 * perfectly on the training half and at chance on the held-out one.
 */
export function fitEncoder(
  dataset: Dataset,
  featureIndices: number[],
  train: Int32Array,
): Encoder {
  const columns: DesignColumn[] = [];
  const used: number[] = [];
  const dropped: { name: string; reason: string }[] = [];

  for (const source of featureIndices) {
    const col = dataset.columns[source];
    if (col.kind === "categorical") {
      const levels = col.levels ?? [];
      if (levels.length > MAX_LEVELS) {
        dropped.push({
          name: col.name,
          reason: `${levels.length} distinct values — too many to one-hot encode (cap ${MAX_LEVELS}).`,
        });
        continue;
      }
      if (levels.length < 2) {
        dropped.push({ name: col.name, reason: "only one value — no information." });
        continue;
      }
      used.push(source);
      for (const level of levels) {
        columns.push({ name: `${col.name} = ${level}`, source });
      }
      continue;
    }

    if (isConstant(col, train)) {
      dropped.push({ name: col.name, reason: "constant on the training rows." });
      continue;
    }
    used.push(source);
    columns.push({ name: col.name, source });
    if (col.missingCount > 0) {
      columns.push({ name: `${col.name} is missing`, source });
    }
  }

  if (columns.length === 0) {
    throw new Error("No usable feature columns — every one was constant, empty or too high-cardinality.");
  }

  // Means over the *training* rows, used both to impute and to standardise.
  //
  // Computed from the **present** cells directly rather than from an encoded
  // matrix, because the encoding needs the mean to impute with: running
  // `encodeRows` first to get the mean would impute every missing cell with
  // zero and then average that in, dragging the mean toward zero in exactly the
  // columns with the most missing data.
  const width = columns.length;
  const mean = new Float32Array(width);
  const std = new Float32Array(width);
  {
    let j = 0;
    for (const source of used) {
      const col = dataset.columns[source];
      if (col.kind === "categorical") {
        const levels = col.levels ?? [];
        for (let l = 0; l < levels.length; l++) {
          let hits = 0;
          for (let i = 0; i < train.length; i++) {
            const r = train[i];
            if (!col.missing[r] && col.values[r] === l) hits++;
          }
          mean[j + l] = hits / train.length;
        }
        j += levels.length;
        continue;
      }
      let sum = 0;
      let seen = 0;
      let missingRows = 0;
      for (let i = 0; i < train.length; i++) {
        const r = train[i];
        if (col.missing[r]) {
          missingRows++;
          continue;
        }
        sum += col.values[r];
        seen++;
      }
      mean[j] = seen > 0 ? sum / seen : 0;
      j++;
      if (col.missingCount > 0) {
        mean[j] = missingRows / train.length;
        j++;
      }
    }
  }

  const encoder: Encoder = { columns, used, dropped, mean, std };
  const encoded = encodeRows(dataset, encoder, train, false);
  for (let j = 0; j < width; j++) {
    // The standardiser's centre is the *encoded* column mean, which for a
    // numeric column with missing cells is not the present-value mean above —
    // it is that mean diluted by nothing, since the imputed cells sit exactly
    // on it. They agree here, and would not if the imputation ever changed.
    let sum = 0;
    for (let i = 0; i < train.length; i++) sum += encoded[i * width + j];
    const mu = sum / train.length;
    let ss = 0;
    for (let i = 0; i < train.length; i++) {
      const d = encoded[i * width + j] - mu;
      ss += d * d;
    }
    mean[j] = mu;
    // A floor, not a guard: a design column that is constant on the training
    // rows would otherwise divide every row by zero and fill the matrix with
    // NaN, which a gradient loop happily propagates for ten epochs.
    std[j] = Math.max(1e-6, Math.sqrt(ss / Math.max(1, train.length)));
  }

  return { columns, used, dropped, mean, std };
}

function isConstant(col: Column, rows: Int32Array): boolean {
  let first: number | null = null;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (col.missing[r]) continue;
    if (first == null) first = col.values[r];
    else if (col.values[r] !== first) return false;
  }
  return true;
}

/**
 * Materialise `rows` as a dense `rows.length × encoder.columns.length` matrix.
 *
 * `standardise` is the switch between the two families: the trees want the
 * column's own units (a split threshold of `age ≤ 38` is readable and a split
 * on a z-score is not), while the gradient families want zero mean and unit
 * variance or the learning rate has to be tuned per column.
 */
export function encodeRows(
  dataset: Dataset,
  encoder: Encoder,
  rows: Int32Array,
  standardise: boolean,
): Float32Array {
  const width = encoder.columns.length;
  const out = new Float32Array(rows.length * width);

  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    let j = 0;
    for (const source of encoder.used) {
      const col = dataset.columns[source];
      if (col.kind === "categorical") {
        const levels = col.levels ?? [];
        // A missing categorical row is all-zero across its levels, which is a
        // distinct and meaningful pattern — no extra indicator needed.
        const code = col.missing[r] ? -1 : col.values[r];
        for (let l = 0; l < levels.length; l++) {
          out[i * width + j + l] = l === code ? 1 : 0;
        }
        j += levels.length;
        continue;
      }
      const missing = col.missing[r] === 1;
      // Imputed with the *training* mean. `encoder.mean` is zero on the first
      // pass (it is being computed), which makes that pass impute with zero —
      // harmless, because that pass exists only to compute the mean and the
      // missing cells are a minority by construction.
      out[i * width + j] = missing ? encoder.mean[j] : col.values[r];
      j++;
      if (col.missingCount > 0) {
        out[i * width + j] = missing ? 1 : 0;
        j++;
      }
    }
  }

  if (standardise) {
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < width; j++) {
        out[i * width + j] = (out[i * width + j] - encoder.mean[j]) / encoder.std[j];
      }
    }
  }
  return out;
}

/**
 * Encode a single hand-typed row for PREDICT.
 *
 * Takes the values in `featureIndices` order — the order the form renders — and
 * returns one row of the design matrix. Anything it cannot read becomes a
 * missing cell rather than a zero, for the reason `parseNumber` gives in
 * `csv.ts`.
 */
export function encodeOne(
  dataset: Dataset,
  encoder: Encoder,
  values: (number | string | null)[],
  featureIndices: number[],
  standardise: boolean,
): Float32Array {
  const width = encoder.columns.length;
  const row = new Float32Array(width);
  let j = 0;
  for (const source of encoder.used) {
    const col = dataset.columns[source];
    const raw = values[featureIndices.indexOf(source)];
    if (col.kind === "categorical") {
      const levels = col.levels ?? [];
      const code = typeof raw === "string" ? levels.indexOf(raw) : -1;
      for (let l = 0; l < levels.length; l++) row[j + l] = l === code ? 1 : 0;
      j += levels.length;
      continue;
    }
    const n = typeof raw === "number" ? raw : Number(raw);
    const missing = raw == null || raw === "" || !Number.isFinite(n);
    row[j] = missing ? encoder.mean[j] : n;
    j++;
    if (col.missingCount > 0) {
      row[j] = missing ? 1 : 0;
      j++;
    }
  }
  if (standardise) {
    for (let k = 0; k < width; k++) row[k] = (row[k] - encoder.mean[k]) / encoder.std[k];
  }
  return row;
}
