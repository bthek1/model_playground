// The ladder, as catalogue data — the SELECT slot's contents.
//
// This is the Tabular counterpart of `text/catalogue.ts` and `vision/models.ts`,
// with one field those files do not have and one they do have that this cannot.
// There is no `bytes` and no `params`, because nothing is downloaded: the size
// guardrail every other category needs is vacuous here, and the thing the user
// actually needs to know before pressing FIT is **how long it will take on this
// many rows** and **where the arithmetic will run**.

import type { Family, Hyperparams, Objective } from "./types";

export interface FamilyInfo {
  id: Family;
  label: string;
  /** One sentence: what it is, and what it is on the ladder for. */
  blurb: string;
  /** Where the heavy arithmetic goes, and the honest reason. */
  compute: "gpu" | "cpu";
  computeNote: string;
  /** Which questions it can answer. */
  objectives: Objective[];
  defaults: Hyperparams;
  /** Hyperparameters this family actually reads — the rest are hidden. */
  knobs: (keyof Hyperparams)[];
}

const TREE_DEFAULTS: Hyperparams = {
  maxDepth: 6,
  minLeaf: 5,
  nTrees: 60,
  featureFraction: 0.7,
  shrinkage: 0.1,
  epochs: 20,
  learningRate: 0.2,
  batchSize: 64,
  hidden: 32,
  lambda: 1,
};

export const FAMILIES: FamilyInfo[] = [
  {
    id: "logistic",
    label: "Logistic regression",
    blurb:
      "The floor. A linear decision boundary, fitted by gradient descent — if nothing on the ladder beats it, the columns carry no interaction worth modelling.",
    compute: "gpu",
    computeNote:
      "Two matmuls per step, run as WGSL compute shaders on your GPU. This is what the hardware is for.",
    objectives: ["classification"],
    defaults: { ...TREE_DEFAULTS, epochs: 30, learningRate: 0.3, batchSize: 64 },
    knobs: ["epochs", "learningRate", "batchSize"],
  },
  {
    id: "forest",
    label: "Random forest",
    blurb:
      "The zero-tuning baseline. Bagged decision trees with a feature subsample at every split; it is usually within a point or two of the best thing here and needs nothing set.",
    compute: "cpu",
    computeNote:
      "Plain TypeScript in a Web Worker. Recursive splitting is branch-heavy and does not vectorise — a GPU would make this slower, not faster.",
    objectives: ["classification", "regression"],
    defaults: { ...TREE_DEFAULTS, nTrees: 60, maxDepth: 8, featureFraction: 0.7 },
    knobs: ["nTrees", "maxDepth", "minLeaf", "featureFraction"],
  },
  {
    id: "boosting",
    label: "Gradient boosting",
    blurb:
      "The workhorse, and on most real tables the winner. Histogram-binned trees fitted to the previous round's residuals — the browser's answer to XGBoost and LightGBM.",
    compute: "cpu",
    computeNote:
      "Plain TypeScript in a Web Worker, histogram-binned like HistGradientBoosting. Same reason as the forest: there is no matmul in a recursive split.",
    objectives: ["classification", "regression"],
    defaults: { ...TREE_DEFAULTS, nTrees: 120, maxDepth: 4, shrinkage: 0.1 },
    knobs: ["nTrees", "maxDepth", "minLeaf", "shrinkage", "featureFraction"],
  },
  {
    id: "mlp",
    label: "Neural network (MLP)",
    blurb:
      "The deep baseline, and it is here because it loses. One hidden layer with ReLU, fitted the same way and scored on the same split — watch it come third on your own data.",
    compute: "gpu",
    computeNote:
      "Four matmuls per step (two forward, two backward) as WGSL on your GPU. The arithmetic is genuinely GPU-shaped; the result still usually is not the best on this page.",
    objectives: ["classification"],
    defaults: { ...TREE_DEFAULTS, epochs: 40, learningRate: 0.1, batchSize: 64, hidden: 48 },
    knobs: ["hidden", "epochs", "learningRate", "batchSize"],
  },
];

/**
 * The regression rungs.
 *
 * Separate entries rather than an `objective` flag on the four above, because
 * two of them are genuinely different models — and because the defaults are
 * **measured on this page's own samples**, not inherited. A depth that suits a
 * Gini split is not automatically right for variance reduction, and reuse that
 * looks like a decision is often an inheritance (`/link-prediction` paid for
 * that one at 0.19 of AUC).
 */
export const REGRESSION_FAMILIES: FamilyInfo[] = [
  {
    id: "ridge",
    label: "Ridge regression",
    blurb:
      "The floor, and the only model here with a closed form: one solve, no iterations, no learning rate. The λ penalty is not a tuning knob bolted on — it is what makes the normal equations solvable at all.",
    compute: "gpu",
    computeNote:
      "XᵀX and Xᵀy are O(n·d²) in the row count and run as WGSL on your GPU; the d×d factorisation is microseconds and stays on the CPU. That is the honest split, and here it is arithmetic rather than a claim.",
    objectives: ["regression"],
    defaults: { ...TREE_DEFAULTS, lambda: 1 },
    knobs: ["lambda"],
  },
  {
    id: "forest",
    label: "Random forest",
    blurb:
      "Bagged regression trees: the split criterion becomes variance reduction and the leaf becomes a mean. Zero tuning, and usually within a little of the best thing here.",
    compute: "cpu",
    computeNote:
      "Plain TypeScript in a Web Worker. Recursive splitting is branch-heavy and does not vectorise — a GPU would make this slower, not faster.",
    objectives: ["regression"],
    defaults: { ...TREE_DEFAULTS, nTrees: 60, maxDepth: 8, minLeaf: 5 },
    knobs: ["nTrees", "maxDepth", "minLeaf", "featureFraction"],
  },
  {
    id: "boosting",
    label: "Gradient boosting",
    blurb:
      "Trees fitted to the previous round's residuals. On most real tables this is the winner, and it is the browser's answer to XGBoost and LightGBM.",
    compute: "cpu",
    computeNote:
      "Plain TypeScript in a Web Worker, histogram-binned like HistGradientBoosting. Same reason as the forest: there is no matmul in a recursive split.",
    objectives: ["regression"],
    defaults: { ...TREE_DEFAULTS, nTrees: 150, maxDepth: 4, shrinkage: 0.08 },
    knobs: ["nTrees", "maxDepth", "minLeaf", "shrinkage", "featureFraction"],
  },
  {
    id: "quantile",
    label: "Quantile regression",
    blurb:
      "Fits an interval rather than a number — a low, a median and a high line, by minimising the pinball loss. A band is more honest than a point estimate, and the median is not the mean on a skewed target.",
    compute: "gpu",
    computeNote:
      "One matmul per batch for all three quantiles at once, as WGSL on your GPU. Three separate fits would be three times the arithmetic for the same lines.",
    objectives: ["regression"],
    defaults: { ...TREE_DEFAULTS, epochs: 60, learningRate: 0.08, batchSize: 64 },
    knobs: ["epochs", "learningRate", "batchSize"],
  },
];

export function familyInfo(id: Family, objective: Objective = "classification"): FamilyInfo {
  const pool = objective === "regression" ? REGRESSION_FAMILIES : FAMILIES;
  const found = pool.find((f) => f.id === id) ?? [...FAMILIES, ...REGRESSION_FAMILIES].find((f) => f.id === id);
  if (!found) throw new Error(`Unknown model family: ${id}`);
  return found;
}

/** Families that can answer this question, in ladder order. */
export function familiesFor(objective: Objective): FamilyInfo[] {
  return objective === "regression" ? REGRESSION_FAMILIES : FAMILIES;
}

/** Human labels and units for the hyperparameter controls. */
export const KNOB_LABELS: Record<
  keyof Hyperparams,
  { label: string; min: number; max: number; step: number; hint: string }
> = {
  maxDepth: {
    label: "Max depth",
    min: 1,
    max: 12,
    step: 1,
    hint: "How many questions deep each tree may go. Past ~8 the fit slows sharply and the held-out score stops moving.",
  },
  minLeaf: {
    label: "Min rows per leaf",
    min: 1,
    max: 50,
    step: 1,
    hint: "A leaf built from three rows is a memory of three rows.",
  },
  nTrees: {
    label: "Trees",
    min: 5,
    max: 300,
    step: 5,
    hint: "The forest averages them; the booster adds them. Both are roughly linear in fit time.",
  },
  featureFraction: {
    label: "Features per split",
    min: 0.1,
    max: 1,
    step: 0.05,
    hint: "Bagging's other half — below 1 the trees decorrelate and the average improves.",
  },
  shrinkage: {
    label: "Learning rate (shrinkage)",
    min: 0.01,
    max: 1,
    step: 0.01,
    hint: "How much of each tree is kept. Smaller needs more trees and usually lands better.",
  },
  epochs: {
    label: "Epochs",
    min: 1,
    max: 200,
    step: 1,
    hint: "Full passes over the training rows.",
  },
  learningRate: {
    label: "Learning rate",
    min: 0.001,
    max: 1,
    step: 0.001,
    hint: "Step size for gradient descent, as a fraction of the target's own spread — so the same number behaves the same way on a column of dollars and a column of ratios.",
  },
  batchSize: {
    label: "Batch size",
    min: 8,
    max: 512,
    step: 8,
    hint: "Rows per gradient step. Larger batches make each GPU matmul worth dispatching.",
  },
  hidden: {
    label: "Hidden units",
    min: 4,
    max: 256,
    step: 4,
    hint: "Width of the single hidden layer.",
  },
  lambda: {
    label: "Ridge penalty (λ)",
    min: 0,
    max: 100,
    step: 0.5,
    hint: "Shrinks the coefficients — and makes XᵀX positive definite, which is what lets the solve succeed at all. Drag it to 0 on a design with two identical columns and the page will say the fit is rank-deficient.",
  },
};
