// Scores, and the null model that makes each of them readable.
//
// Every function here returns its baseline **inside** the same object as the
// score. That is not tidiness: `/graph-classification` settled it the hard way.
// A class prior is the easiest thing in any dataset to learn, so a broken
// readout and a working model both produce a confident, plausible number —
// PROTEINS is 663/450, so "0.598 accuracy" is simultaneously a decent-looking
// result and proof that the model ignored its input. An accuracy without its
// null model is not a number. Carrying the two together is what stops them
// coming from different runs, which is the way the pairing actually breaks.

import type {
  ClassificationMetrics,
  ConfusionMatrix,
  RegressionMetrics,
} from "./types";

/** Argmax of one row of a probability matrix. */
export function argmaxRow(
  probs: Float32Array,
  row: number,
  classes: number,
): number {
  let best = 0;
  let bestP = -Infinity;
  for (let c = 0; c < classes; c++) {
    const p = probs[row * classes + c];
    if (p > bestP) {
      bestP = p;
      best = c;
    }
  }
  return best;
}

/**
 * Turn held-out probabilities into predicted class codes.
 *
 * `threshold` applies to **binary problems only**, where it is the probability
 * the positive class (code 1) must clear. The default 0.5 is a convention, not
 * a decision, which is exactly why the page puts a slider on it — and why this
 * function takes probabilities rather than labels, so moving the slider
 * re-derives on the main thread and never re-fits.
 */
export function predictedClasses(
  probs: Float32Array,
  rows: number,
  classes: number,
  threshold = 0.5,
): Uint8Array {
  const out = new Uint8Array(rows);
  if (classes === 2) {
    for (let i = 0; i < rows; i++) out[i] = probs[i * 2 + 1] >= threshold ? 1 : 0;
    return out;
  }
  for (let i = 0; i < rows; i++) out[i] = argmaxRow(probs, i, classes);
  return out;
}

export function confusionMatrix(
  actual: Uint8Array,
  predicted: Uint8Array,
  labels: string[],
): ConfusionMatrix {
  const k = labels.length;
  const counts = new Int32Array(k * k);
  for (let i = 0; i < actual.length; i++) {
    counts[actual[i] * k + predicted[i]]++;
  }
  return { counts, labels };
}

/**
 * Accuracy, macro precision/recall/F1, the matrix, and the majority baseline.
 *
 * `trainLabels` is what the baseline is computed from — the *training* half's
 * most common class, scored on the *held-out* rows. Taking the majority from
 * the held-out half instead would give the baseline a look at the answers and
 * flatter it, which on an imbalanced file is enough to make a real model look
 * worse than doing nothing.
 */
export function classificationMetrics(
  actual: Uint8Array,
  predicted: Uint8Array,
  labels: string[],
  trainLabels: Uint8Array,
): ClassificationMetrics {
  const k = labels.length;
  const confusion = confusionMatrix(actual, predicted, labels);
  const counts = confusion.counts;

  let correct = 0;
  for (let c = 0; c < k; c++) correct += counts[c * k + c];
  const accuracy = actual.length === 0 ? 0 : correct / actual.length;

  let precision = 0;
  let recall = 0;
  let f1 = 0;
  let present = 0;
  for (let c = 0; c < k; c++) {
    const tp = counts[c * k + c];
    let predictedC = 0;
    let actualC = 0;
    for (let j = 0; j < k; j++) {
      predictedC += counts[j * k + c];
      actualC += counts[c * k + j];
    }
    // Macro-averaged over the classes that actually occur in the held-out half.
    // Including an absent class as a zero drags every macro score down by a
    // factor that has nothing to do with the model.
    if (actualC === 0) continue;
    present++;
    const p = predictedC === 0 ? 0 : tp / predictedC;
    const r = tp / actualC;
    precision += p;
    recall += r;
    f1 += p + r === 0 ? 0 : (2 * p * r) / (p + r);
  }
  const denom = Math.max(1, present);

  const trainCounts = new Int32Array(k);
  for (let i = 0; i < trainLabels.length; i++) trainCounts[trainLabels[i]]++;
  let majority = 0;
  for (let c = 1; c < k; c++) if (trainCounts[c] > trainCounts[majority]) majority = c;
  let baselineHits = 0;
  for (let i = 0; i < actual.length; i++) if (actual[i] === majority) baselineHits++;

  return {
    accuracy,
    precision: precision / denom,
    recall: recall / denom,
    f1: f1 / denom,
    confusion,
    baselineAccuracy: actual.length === 0 ? 0 : baselineHits / actual.length,
    baselineLabel: labels[majority] ?? "—",
  };
}

/**
 * Re-derive the whole metric block at a new threshold, from probabilities
 * already in hand.
 *
 * This is the page's most useful control and it must never re-fit — moving a
 * slider is a *re-read*, per the model-page-pattern rule that a control which
 * re-derives from a result in hand spends nothing. Everything it needs is in
 * the `FitResult` the worker already returned.
 */
export function metricsAtThreshold(
  probabilities: Float32Array,
  actual: Uint8Array,
  labels: string[],
  trainLabels: Uint8Array,
  threshold: number,
): ClassificationMetrics {
  const k = labels.length;
  const rows = actual.length;
  const predicted = predictedClasses(probabilities, rows, k, threshold);
  return classificationMetrics(actual, predicted, labels, trainLabels);
}

// --- Regression ---------------------------------------------------------------

/**
 * RMSE, MAE, R² — and the null model, which for regression is **predicting the
 * training mean**.
 *
 * R² is that comparison already (it is `1 − SSE/SST` against the *test* mean),
 * but the page still shows the baseline's own RMSE and MAE beside the model's,
 * for two reasons. It puts the null model in the same units as the score, which
 * R² is not; and R² computed against the test mean flatters a model on a split
 * whose held-out half happens to be less variable than the training half. Both
 * numbers come out of the same call so they cannot describe different splits.
 *
 * `units` is carried, not derived: `transform.ts` owns it precisely so an RMSE
 * in log space can never be rendered beside one in the target's units without
 * both being labelled.
 */
export function regressionMetrics(
  actual: Float32Array,
  predicted: Float32Array,
  trainTargets: Float32Array,
  units: string,
): RegressionMetrics {
  const n = actual.length;
  if (n === 0) {
    return { rmse: 0, mae: 0, r2: 0, baselineRmse: 0, baselineMae: 0, units };
  }

  let trainMean = 0;
  for (let i = 0; i < trainTargets.length; i++) trainMean += trainTargets[i];
  trainMean /= Math.max(1, trainTargets.length);

  let testMean = 0;
  for (let i = 0; i < n; i++) testMean += actual[i];
  testMean /= n;

  let sse = 0;
  let sae = 0;
  let sst = 0;
  let baseSse = 0;
  let baseSae = 0;
  for (let i = 0; i < n; i++) {
    const e = predicted[i] - actual[i];
    sse += e * e;
    sae += Math.abs(e);
    const t = actual[i] - testMean;
    sst += t * t;
    const b = trainMean - actual[i];
    baseSse += b * b;
    baseSae += Math.abs(b);
  }

  return {
    rmse: Math.sqrt(sse / n),
    mae: sae / n,
    // A constant target makes SST zero, and 1 − 0/0 is NaN. A model that
    // predicts a constant target exactly has explained everything there was.
    r2: sst === 0 ? (sse === 0 ? 1 : 0) : 1 - sse / sst,
    baselineRmse: Math.sqrt(baseSse / n),
    baselineMae: baseSae / n,
    units,
  };
}
