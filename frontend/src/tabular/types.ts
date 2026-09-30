// The Tabular category's vocabulary: a columnar dataset, the fit specification,
// and the worker protocol — expressed over the shared `ModelRequest` /
// `ModelResponse` envelope from `model/types.ts`, not beside it.
//
// The shape of the data is the whole design decision here. Every other category
// hands the worker something small (a string, one image, a clip) and downloads
// something large; this one downloads nothing and hands the worker the load
// itself. A million-row CSV as an array of row objects is hundreds of megabytes
// of JS objects; as one `Float32Array` per column it is four bytes a cell. So
// there is no row type in this file, deliberately — nothing anywhere in
// `src/tabular/` ever materialises `{ age: 31, city: "Wellington" }`.

import type { ModelRequest, ModelResponse } from "@/model/types";

/**
 * How a column's numbers are to be read.
 *
 * `categorical` values are integer *codes* into `levels`, stored as floats like
 * everything else — one array type for the whole dataset is worth more than the
 * two bytes a `Uint16Array` would save on some columns.
 */
export type ColumnKind = "numeric" | "categorical";

export interface Column {
  name: string;
  kind: ColumnKind;
  /** `rowCount` values. Codes into `levels` when `kind` is categorical. */
  values: Float32Array;
  /** Distinct strings, in first-seen order. Present iff categorical. */
  levels?: string[];
  /**
   * 1 where the cell was empty in the file.
   *
   * An explicit mask rather than a sentinel, and this is not fastidiousness:
   * `NaN` and `0` are both real values in real data, so any sentinel silently
   * becomes one of them on some file. `values[i]` is undefined-but-finite
   * (zero) where `missing[i]` is 1; nothing may read it without checking.
   */
  missing: Uint8Array;
  /** Cached `sum(missing)`, so the UI need not walk the column to draw a badge. */
  missingCount: number;
}

export interface Dataset {
  /** Where it came from — a file name, or a bundled sample's title. */
  name: string;
  columns: Column[];
  /** Rows actually held. */
  rowCount: number;
  /** Rows in the source. Greater than `rowCount` when the cap bit. */
  sourceRowCount: number;
  /** True when `sourceRowCount > rowCount` — the UI must say so. */
  sampled: boolean;
}

/** A row the parser refused, with the line number the user can go and look at. */
export interface ParseIssue {
  /** 1-based line in the file, counting the header. */
  line: number;
  message: string;
}

export interface ParseResult {
  dataset: Dataset;
  /**
   * Rows rejected, with their line numbers. Ragged rows are *rejected*, never
   * padded: a best-effort recovery produces a model fitted on shifted columns,
   * which trains happily and is wrong everywhere.
   */
  issues: ParseIssue[];
}

// --- The fit -----------------------------------------------------------------

/**
 * The model ladder. Four families, and the MLP is on it precisely because it
 * loses: "deep learning does not win on tabular data" is a claim people do not
 * believe until they watch it happen on their own file, and it only lands if
 * the MLP is genuinely fitted rather than described.
 */
export type Family =
  | "logistic"
  | "forest"
  | "boosting"
  | "mlp"
  // Regression-only rungs. `ridge` is the closed-form solve — the one place the
  // GPU/CPU split is exact rather than rhetorical — and `quantile` fits a band
  // instead of a number, which is both more honest and the better thing to draw.
  | "ridge"
  | "quantile";

/** What the page is asking of the target column. */
export type Objective = "classification" | "regression";

export interface Hyperparams {
  // Trees (forest, boosting)
  maxDepth: number;
  minLeaf: number;
  nTrees: number;
  /** Fraction of features considered at each split. Bagging's other half. */
  featureFraction: number;
  /** Boosting's shrinkage. */
  shrinkage: number;
  // Gradient families (logistic, mlp)
  epochs: number;
  learningRate: number;
  batchSize: number;
  /** MLP hidden width. One hidden layer — a second buys nothing here. */
  hidden: number;
  /** Ridge's L2 penalty. It is what makes the normal equations solvable. */
  lambda: number;
}

export interface FitSpec {
  family: Family;
  objective: Objective;
  /** Index into `Dataset.columns`. */
  targetIndex: number;
  /** Indices into `Dataset.columns`. Never contains `targetIndex`. */
  featureIndices: number[];
  hp: Hyperparams;
  /** Drives the split, the bagging and the permutation shuffles. On screen. */
  seed: number;
  /** Held-out fraction, 0–1. */
  testFraction: number;
  /**
   * Fit on `log1p(target)` and back-transform before scoring (regression only).
   * Cannot be re-derived from a fit on the raw target, so flipping it spends —
   * see `transform.ts`.
   */
  logTarget?: boolean;
  /** Quantiles to fit alongside the mean (regression only). */
  quantiles?: number[];
}

// --- Results -----------------------------------------------------------------

export interface ConfusionMatrix {
  /** `counts[actual * k + predicted]`. */
  counts: Int32Array;
  labels: string[];
}

export interface ClassificationMetrics {
  accuracy: number;
  /** Macro-averaged over classes. */
  precision: number;
  recall: number;
  f1: number;
  confusion: ConfusionMatrix;
  /**
   * What predicting the training set's most common class scores on this same
   * held-out split. It travels *inside* the metrics so the two can never come
   * from different runs — `/graph-classification` settled that, and the reason
   * is that a class prior is the easiest thing in any dataset to learn, so a
   * broken model and a working one both produce a plausible-looking number.
   */
  baselineAccuracy: number;
  /** The label that baseline predicts, for the sentence beside it. */
  baselineLabel: string;
}

export interface RegressionMetrics {
  rmse: number;
  mae: number;
  r2: number;
  /** RMSE of predicting the *training* mean on the held-out rows. */
  baselineRmse: number;
  baselineMae: number;
  /** The units these numbers are in — `transform.ts` owns this string. */
  units: string;
}

export interface FeatureImportance {
  name: string;
  /** Mean score drop when this column is shuffled. Higher matters more. */
  drop: number;
}

export interface FitResult {
  spec: FitSpec;
  /** Rows used to fit, and rows held out. */
  trainRows: number;
  testRows: number;
  /**
   * Rows excluded because the **target** was missing.
   *
   * Dropped rather than imputed or refused. A row with no answer cannot be
   * trained on and cannot be scored — but refusing the whole file over two
   * blank cells is unusable (the Palmer penguins sample has exactly that), and
   * imputing a target is inventing the thing being predicted. A missing
   * *categorical* target is the dangerous case: its code is 0, which is a real
   * class, so the row would silently join whichever class happened to be
   * encountered first.
   */
  droppedRows: number;
  /** Wall-clock of the fit itself, excluding the design-matrix build. */
  fitMs: number;
  /** Where the arithmetic ran, and why — rendered on the page. */
  compute: "gpu" | "cpu";
  classification?: ClassificationMetrics;
  /** Scored in the target's own units — always, whatever the fit was on. */
  regression?: RegressionMetrics;
  /**
   * The same scores in the space the model was actually fitted in, present only
   * when that differs (the log-transform toggle).
   *
   * They are a *separate field* rather than a replacement because the page's
   * whole subject is that the two are not comparable: putting an RMSE in log
   * units where an RMSE in dollars is expected is the mistake being
   * demonstrated, so both are carried and both are labelled with the units
   * `transform.ts` gave them.
   */
  logSpaceRegression?: RegressionMetrics;
  /**
   * Held-out class probabilities, `testRows × classes`, row-major. The
   * threshold slider re-derives precision, recall and the matrix from these on
   * the main thread — it re-reads a fit in hand and must never re-fit.
   */
  probabilities?: Float32Array;
  /** Held-out predictions (regression), in the target's own units. */
  predictions?: Float32Array;
  /** Held-out truth, aligned with the two arrays above. */
  actuals?: Float32Array;
  /**
   * Held-out class codes, and the *training* half's class codes.
   *
   * Both travel with the result because the threshold slider re-derives the
   * whole metric block on the main thread, and the baseline has to come from
   * the training half of this same split — computing it from whatever is to
   * hand is how a null model and a score end up describing different runs.
   */
  testLabels?: Uint8Array;
  trainLabels?: Uint8Array;
  /** Quantile predictions, `quantiles.length × testRows`. */
  quantilePredictions?: Float32Array;
  /** Empirical coverage of the outer quantile band on the held-out rows. */
  quantileCoverage?: number;
  /** Class labels in code order (classification only). */
  labels?: string[];
  importance: FeatureImportance[];
  /** Ridge only: one coefficient per design column, plus the intercept. */
  coefficients?: { name: string; weight: number }[];
  /**
   * Ridge only: whether the normal equations were solvable without λ. A design
   * with a duplicated column is rank-deficient, and reporting that beats
   * returning the plausible-looking garbage an unregularised solve produces.
   */
  rankDeficient?: boolean;
  /** Per-iteration loss, for the fit curve. Sparse — one point per report. */
  curve: { step: number; loss: number }[];
}

/** Progress *inside* the fit: determinate, because the user just set the total. */
export interface FitPartial {
  /** Iterations finished — epochs for the gradient families, trees otherwise. */
  done: number;
  /** Iterations asked for. Known up front, which is why this is not a spinner. */
  total: number;
  /** "Encoding", "Fitting", "Scoring", "Permuting" — named for the user. */
  phase: string;
  /** Training loss at `done`, where the family has one. */
  loss: number | null;
}

/** A single hand-typed row, predicted after the fit. */
export interface PredictRequest {
  /** One value per `featureIndices` entry, in that order. */
  values: (number | string | null)[];
}

export interface PredictResult {
  /** Class probabilities, or a single-element array for regression. */
  scores: number[];
  labels: string[];
  objective: Objective;
}

// --- Worker protocol ---------------------------------------------------------
//
// `load` hands over the dataset. It is the one load in the app that spends
// nothing — no bytes cross the network and the rows are already in the tab —
// but it still goes through Machine A, because the page needs somewhere to put
// "the worker has the data" and inventing a status for it would cost the one
// vocabulary that makes every other page readable.
//
// The **fit is a `run`**, which is what puts its iteration metrics on `partial`:
// progress inside one request, correlated to its id. Machine A stays `ready`
// throughout and `running` stays an inflight count. No new envelope, no new
// status — see `docs/standards/model-page-pattern.md` §7.

export interface TabularLoad {
  dataset: Dataset;
}

export type TabularRun = { spec: FitSpec } | { predict: PredictRequest };

export type TabularRequest =
  | ModelRequest<TabularLoad, TabularRun>
  /**
   * Stop the fit in flight. Outside the shared envelope on purpose: it carries
   * no id and expects no reply, because it does not open a request — it sets a
   * flag the fit loop polls, exactly as `FitCallbacks.shouldStop` does in
   * `webgpu/linearModel.ts`. The run it interrupts still settles, with whatever
   * the loop had reached.
   */
  | { type: "stop" };

export type TabularResponse = ModelResponse<
  FitResult | PredictResult,
  FitPartial
>;
