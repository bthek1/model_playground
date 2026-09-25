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

import type { ClassificationMetrics, ConfusionMatrix } from "./types";

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
