// Decision trees, a random forest, and gradient boosting — plain TypeScript,
// no GPU, and the page says why.
//
// Recursive splitting is branch-heavy and does not vectorise: every node asks a
// different question of a different subset of rows, which is the opposite of
// what a compute shader is for. Putting this in WGSL would make it slower and
// more impressive-sounding, and §5 of the tabular roadmap is explicit that it
// is the wrong instinct. The linear and MLP halves of the ladder *do* go to the
// GPU, through the same `MatmulFn` seam `webgpu/linearModel.ts` uses — the
// split between the two is a teaching point, not an implementation accident.
//
// Everything here is histogram-based, as HistGradientBoosting and LightGBM are:
// features are pre-binned once into at most `MAX_BINS` quantile buckets, and
// split finding is then one pass per node per feature over 64 accumulators
// instead of a sort. The bin edges are computed from the **training rows only**,
// for the reason `design.ts` gives at length.

import { mulberry32 } from "@/lib/random";

/** Enough resolution that binning is not the thing limiting a split. */
export const MAX_BINS = 64;

export interface Binning {
  /** `rows × features`, one bin index per cell. */
  bins: Uint8Array;
  /** Per feature, ascending bin upper bounds. Length is that feature's bins−1. */
  thresholds: Float32Array[];
  features: number;
}

/**
 * Quantile-bin a design matrix.
 *
 * Quantiles rather than equal width: a column with one outlier at 10^6 gets 63
 * empty buckets and one useful one under equal width, and every split it could
 * have made disappears.
 */
export function binFeatures(
  x: Float32Array,
  rows: number,
  features: number,
  maxBins = MAX_BINS,
): Binning {
  const bins = new Uint8Array(rows * features);
  const thresholds: Float32Array[] = [];
  const column = new Float32Array(rows);

  for (let f = 0; f < features; f++) {
    for (let i = 0; i < rows; i++) column[i] = x[i * features + f];
    const sorted = Float32Array.from(column).sort();
    // Distinct values first: a binary indicator has two, and asking for 63 cut
    // points on it produces 62 duplicates that each split nothing.
    const distinct: number[] = [];
    for (let i = 0; i < sorted.length; i++) {
      if (i === 0 || sorted[i] !== sorted[i - 1]) distinct.push(sorted[i]);
    }
    let cuts: Float32Array;
    if (distinct.length <= maxBins) {
      // Every distinct value gets its own bin; the cut sits on the value below.
      cuts = Float32Array.from(distinct.slice(0, distinct.length - 1));
    } else {
      const out: number[] = [];
      for (let b = 1; b < maxBins; b++) {
        const v = sorted[Math.floor((b * sorted.length) / maxBins)];
        if (out.length === 0 || v > out[out.length - 1]) out.push(v);
      }
      cuts = Float32Array.from(out);
    }
    thresholds.push(cuts);
    for (let i = 0; i < rows; i++) {
      bins[i * features + f] = binOf(column[i], cuts);
    }
  }
  return { bins, thresholds, features };
}

/** Bin index for `v`: the number of cut points strictly below it. */
function binOf(v: number, cuts: Float32Array): number {
  let lo = 0;
  let hi = cuts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (v <= cuts[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** Apply an existing binning to unseen rows — the held-out half, or one row. */
export function applyBinning(
  binning: Binning,
  x: Float32Array,
  rows: number,
): Uint8Array {
  const { features, thresholds } = binning;
  const out = new Uint8Array(rows * features);
  for (let i = 0; i < rows; i++) {
    for (let f = 0; f < features; f++) {
      out[i * features + f] = binOf(x[i * features + f], thresholds[f]);
    }
  }
  return out;
}

/**
 * A fitted tree, flat.
 *
 * Node 0 is the root. `feature[n] < 0` marks a leaf, whose prediction is
 * `value.subarray(n * stride, (n + 1) * stride)` — one number for a regression
 * or boosting tree, one per class for a classification tree.
 */
export interface Tree {
  feature: Int32Array;
  /** Bin index: a row goes left when its bin is ≤ this. */
  bin: Int32Array;
  /** The same split in the column's own units, for a readable diagram. */
  threshold: Float32Array;
  left: Int32Array;
  right: Int32Array;
  value: Float32Array;
  stride: number;
  nodes: number;
  depth: number;
}

export interface TreeParams {
  maxDepth: number;
  minLeaf: number;
  /** Fraction of features considered at each split — bagging's other half. */
  featureFraction: number;
  /** Newton trees only: the L2 penalty on leaf values. */
  lambda?: number;
}

interface Builder {
  stride: number;
  /** Accumulate a row into `acc` at `slot`. */
  add: (acc: Float64Array, slot: number, row: number) => void;
  /** Score of a node given its totals — higher is better. */
  score: (acc: Float64Array, offset: number) => number;
  /** Leaf prediction from a node's totals, written into `out`. */
  leaf: (acc: Float64Array, offset: number, out: Float32Array, at: number) => void;
  /** Accumulator slots per bin. */
  slots: number;
  /** Rows reaching a node, read from its totals — the min-leaf test. */
  count: (acc: Float64Array, offset: number) => number;
}

/**
 * The generic histogram tree builder.
 *
 * Both concrete builders below hand it a `Builder`: the classification one
 * accumulates class counts and scores by Gini, the Newton one accumulates
 * gradients and hessians and scores by the usual `g²/(h+λ)`. Sharing the
 * recursion, the caps and the feature subsample is the point — §Phase 1 of #49
 * asks the regression trees to reuse this file rather than copy it, and there
 * is nothing left to copy.
 */
function buildTree(
  binning: Binning,
  rows: Int32Array,
  params: TreeParams,
  builder: Builder,
  rand: () => number,
): Tree {
  const { features, bins } = binning;
  const maxNodes = Math.max(3, 2 * (1 << Math.min(params.maxDepth, 20)) + 1);

  const feature = new Int32Array(maxNodes).fill(-1);
  const binAt = new Int32Array(maxNodes).fill(-1);
  const threshold = new Float32Array(maxNodes);
  const left = new Int32Array(maxNodes).fill(-1);
  const right = new Int32Array(maxNodes).fill(-1);
  const value = new Float32Array(maxNodes * builder.stride);
  let nodes = 0;
  let deepest = 0;

  const slots = builder.slots;
  const hist = new Float64Array(MAX_BINS * slots);
  const totals = new Float64Array(slots);
  const acc = new Float64Array(slots);

  const candidates = new Int32Array(features);
  for (let f = 0; f < features; f++) candidates[f] = f;

  /** Recursively build, returning the node index. `nodeRows` is owned here. */
  function grow(nodeRows: Int32Array, depth: number): number {
    const node = nodes++;
    deepest = Math.max(deepest, depth);

    totals.fill(0);
    for (let i = 0; i < nodeRows.length; i++) builder.add(totals, 0, nodeRows[i]);
    builder.leaf(totals, 0, value, node * builder.stride);

    if (depth >= params.maxDepth || nodeRows.length < 2 * params.minLeaf) {
      return node;
    }

    const parentScore = builder.score(totals, 0);
    let bestGain = 0;
    let bestFeature = -1;
    let bestBin = -1;

    // The feature subsample. Fisher–Yates on a prefix, so the draw costs
    // `nTry` swaps rather than a shuffle of the whole list at every node.
    const nTry = Math.max(
      1,
      Math.min(features, Math.round(features * params.featureFraction)),
    );
    for (let t = 0; t < nTry; t++) {
      const j = t + Math.floor(rand() * (features - t));
      const tmp = candidates[t];
      candidates[t] = candidates[j];
      candidates[j] = tmp;
      const f = candidates[t];

      hist.fill(0);
      for (let i = 0; i < nodeRows.length; i++) {
        const r = nodeRows[i];
        builder.add(hist, bins[r * features + f] * slots, r);
      }

      acc.fill(0);
      const nBins = binning.thresholds[f].length + 1;
      for (let b = 0; b < nBins - 1; b++) {
        for (let s = 0; s < slots; s++) acc[s] += hist[b * slots + s];
        const leftCount = builder.count(acc, 0);
        const rightCount = builder.count(totals, 0) - leftCount;
        if (leftCount < params.minLeaf || rightCount < params.minLeaf) continue;
        // The right child's totals are the parent's minus the left's, which is
        // what makes this one pass rather than two.
        const rightAcc = new Float64Array(slots);
        for (let s = 0; s < slots; s++) rightAcc[s] = totals[s] - acc[s];
        const gain =
          builder.score(acc, 0) + builder.score(rightAcc, 0) - parentScore;
        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = f;
          bestBin = b;
        }
      }
    }

    if (bestFeature < 0) return node;

    const leftRows: number[] = [];
    const rightRows: number[] = [];
    for (let i = 0; i < nodeRows.length; i++) {
      const r = nodeRows[i];
      if (bins[r * features + bestFeature] <= bestBin) leftRows.push(r);
      else rightRows.push(r);
    }

    feature[node] = bestFeature;
    binAt[node] = bestBin;
    threshold[node] = binning.thresholds[bestFeature][bestBin];
    left[node] = grow(Int32Array.from(leftRows), depth + 1);
    right[node] = grow(Int32Array.from(rightRows), depth + 1);
    return node;
  }

  grow(rows, 0);

  return {
    feature: feature.slice(0, nodes),
    bin: binAt.slice(0, nodes),
    threshold: threshold.slice(0, nodes),
    left: left.slice(0, nodes),
    right: right.slice(0, nodes),
    value: value.slice(0, nodes * builder.stride),
    stride: builder.stride,
    nodes,
    depth: deepest,
  };
}

/** Walk one row (already binned) to its leaf, returning the leaf's offset. */
export function leafOffset(tree: Tree, rowBins: Uint8Array, base: number): number {
  let node = 0;
  while (tree.feature[node] >= 0) {
    node = rowBins[base + tree.feature[node]] <= tree.bin[node]
      ? tree.left[node]
      : tree.right[node];
  }
  return node * tree.stride;
}

// --- Classification: Gini ----------------------------------------------------

/**
 * A classification tree over class counts, split by Gini impurity.
 *
 * The score is the *negative* weighted impurity, so "higher is better" holds
 * for both builders and `gain = left + right − parent` reads the same way.
 */
export function fitGiniTree(
  binning: Binning,
  rows: Int32Array,
  y: Uint8Array,
  numClasses: number,
  params: TreeParams,
  rand: () => number,
): Tree {
  const builder: Builder = {
    stride: numClasses,
    slots: numClasses,
    add: (acc, slot, row) => {
      acc[slot + y[row]] += 1;
    },
    count: (acc, offset) => {
      let n = 0;
      for (let c = 0; c < numClasses; c++) n += acc[offset + c];
      return n;
    },
    score: (acc, offset) => {
      let n = 0;
      for (let c = 0; c < numClasses; c++) n += acc[offset + c];
      if (n === 0) return 0;
      let sumSq = 0;
      for (let c = 0; c < numClasses; c++) {
        const p = acc[offset + c] / n;
        sumSq += p * p;
      }
      // −n · gini = −n(1 − Σp²). Weighted by n so the sum over children is
      // comparable with the parent's.
      return -n * (1 - sumSq);
    },
    leaf: (acc, offset, out, at) => {
      let n = 0;
      for (let c = 0; c < numClasses; c++) n += acc[offset + c];
      for (let c = 0; c < numClasses; c++) {
        out[at + c] = n === 0 ? 1 / numClasses : acc[offset + c] / n;
      }
    },
  };
  return buildTree(binning, rows, params, builder, rand);
}

// --- Regression and boosting: Newton -----------------------------------------

/**
 * A tree fitted to first and second derivatives.
 *
 * One builder covers three things, which is why the regression arm of #49 added
 * no recursion of its own: a plain regression tree is `g = −y, h = 1` (the leaf
 * becomes the mean and the gain becomes variance reduction exactly), a squared-
 * loss boosting stage is `g = pred − y, h = 1`, and a logistic boosting stage is
 * `g = p − y, h = p(1 − p)`.
 */
export function fitNewtonTree(
  binning: Binning,
  rows: Int32Array,
  grad: Float32Array,
  hess: Float32Array,
  params: TreeParams,
  rand: () => number,
): Tree {
  const lambda = params.lambda ?? 1;
  const builder: Builder = {
    stride: 1,
    // [Σg, Σh, n] — the count is carried explicitly because `Σh` is not a row
    // count once the hessian stops being 1, and `minLeaf` means rows.
    slots: 3,
    add: (acc, slot, row) => {
      acc[slot] += grad[row];
      acc[slot + 1] += hess[row];
      acc[slot + 2] += 1;
    },
    count: (acc, offset) => acc[offset + 2],
    score: (acc, offset) => {
      const g = acc[offset];
      const h = acc[offset + 1];
      return (g * g) / (h + lambda);
    },
    leaf: (acc, offset, out, at) => {
      out[at] = -acc[offset] / (acc[offset + 1] + lambda);
    },
  };
  return buildTree(binning, rows, params, builder, rand);
}

// --- Ensembles ---------------------------------------------------------------

export interface Forest {
  trees: Tree[];
  numClasses: number;
  /** Regression only: the constant every tree's output is added to. */
  base: number;
  kind: "forest-classify" | "forest-regress" | "boost-classify" | "boost-regress";
  shrinkage: number;
}

/** Bootstrap sample of `rows`, with replacement — the bagging half. */
function bootstrap(rows: Int32Array, rand: () => number): Int32Array {
  const out = new Int32Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    out[i] = rows[Math.floor(rand() * rows.length)];
  }
  return out;
}

export interface EnsembleCallbacks {
  /** Called after each tree with `(done, total, loss | null)`. */
  onProgress?: (done: number, total: number, loss: number | null) => void;
  /** Polled between trees; true stops the fit and keeps what was built. */
  shouldStop?: () => boolean;
}

export function fitForestClassifier(
  binning: Binning,
  rows: Int32Array,
  y: Uint8Array,
  numClasses: number,
  nTrees: number,
  params: TreeParams,
  seed: number,
  cb: EnsembleCallbacks = {},
): Forest {
  const rand = mulberry32(seed);
  const trees: Tree[] = [];
  for (let t = 0; t < nTrees; t++) {
    if (cb.shouldStop?.()) break;
    trees.push(fitGiniTree(binning, bootstrap(rows, rand), y, numClasses, params, rand));
    cb.onProgress?.(t + 1, nTrees, null);
  }
  return { trees, numClasses, base: 0, kind: "forest-classify", shrinkage: 1 };
}

export function fitForestRegressor(
  binning: Binning,
  rows: Int32Array,
  y: Float32Array,
  nTrees: number,
  params: TreeParams,
  seed: number,
  cb: EnsembleCallbacks = {},
): Forest {
  const rand = mulberry32(seed);
  const grad = new Float32Array(y.length);
  const hess = new Float32Array(y.length).fill(1);
  for (let i = 0; i < y.length; i++) grad[i] = -y[i];
  const trees: Tree[] = [];
  for (let t = 0; t < nTrees; t++) {
    if (cb.shouldStop?.()) break;
    trees.push(
      fitNewtonTree(binning, bootstrap(rows, rand), grad, hess, { ...params, lambda: 0 }, rand),
    );
    cb.onProgress?.(t + 1, nTrees, null);
  }
  return { trees, numClasses: 1, base: 0, kind: "forest-regress", shrinkage: 1 };
}

export function fitBoostedRegressor(
  binning: Binning,
  rows: Int32Array,
  y: Float32Array,
  nTrees: number,
  shrinkage: number,
  params: TreeParams,
  seed: number,
  cb: EnsembleCallbacks = {},
): Forest {
  const rand = mulberry32(seed);
  let base = 0;
  for (let i = 0; i < rows.length; i++) base += y[rows[i]];
  base /= rows.length;

  const pred = new Float32Array(y.length).fill(base);
  const grad = new Float32Array(y.length);
  const hess = new Float32Array(y.length).fill(1);
  const trees: Tree[] = [];

  for (let t = 0; t < nTrees; t++) {
    if (cb.shouldStop?.()) break;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      grad[r] = pred[r] - y[r];
    }
    const tree = fitNewtonTree(binning, rows, grad, hess, params, rand);
    trees.push(tree);
    let loss = 0;
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      pred[r] += shrinkage * tree.value[leafOffset(tree, binning.bins, r * binning.features)];
      const d = pred[r] - y[r];
      loss += d * d;
    }
    cb.onProgress?.(t + 1, nTrees, Math.sqrt(loss / rows.length));
  }
  return { trees, numClasses: 1, base, kind: "boost-regress", shrinkage };
}

/**
 * Multiclass gradient boosting: `numClasses` trees per round, softmax loss.
 *
 * Binary problems still build two trees a round rather than one. It costs a
 * factor of two on the cheapest case and removes a whole second code path whose
 * only test would be "does binary still work" — and the two-tree form is what
 * makes the leaf values and the loss curve line up with the multiclass case the
 * page also has to render.
 */
export function fitBoostedClassifier(
  binning: Binning,
  rows: Int32Array,
  y: Uint8Array,
  numClasses: number,
  nTrees: number,
  shrinkage: number,
  params: TreeParams,
  seed: number,
  cb: EnsembleCallbacks = {},
): Forest {
  const rand = mulberry32(seed);
  const n = y.length;
  const scores = new Float32Array(n * numClasses);
  const grad = new Float32Array(n);
  const hess = new Float32Array(n);
  const trees: Tree[] = [];
  const probs = new Float32Array(numClasses);

  for (let t = 0; t < nTrees; t++) {
    if (cb.shouldStop?.()) break;
    let loss = 0;
    // One softmax pass per round, reused by every class's tree.
    const cached = new Float32Array(rows.length * numClasses);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      let max = -Infinity;
      for (let c = 0; c < numClasses; c++) max = Math.max(max, scores[r * numClasses + c]);
      let sum = 0;
      for (let c = 0; c < numClasses; c++) {
        const e = Math.exp(scores[r * numClasses + c] - max);
        probs[c] = e;
        sum += e;
      }
      for (let c = 0; c < numClasses; c++) cached[i * numClasses + c] = probs[c] / sum;
      loss -= Math.log(Math.max(cached[i * numClasses + y[r]], 1e-12));
    }

    for (let c = 0; c < numClasses; c++) {
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        const p = cached[i * numClasses + c];
        grad[r] = p - (y[r] === c ? 1 : 0);
        hess[r] = Math.max(1e-6, p * (1 - p));
      }
      const tree = fitNewtonTree(binning, rows, grad, hess, params, rand);
      trees.push(tree);
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        scores[r * numClasses + c] +=
          shrinkage * tree.value[leafOffset(tree, binning.bins, r * binning.features)];
      }
    }
    cb.onProgress?.(t + 1, nTrees, loss / rows.length);
  }
  return { trees, numClasses, base: 0, kind: "boost-classify", shrinkage };
}

// --- Prediction --------------------------------------------------------------

/** Class probabilities for `rows` binned rows, `rows × numClasses` row-major. */
export function predictProba(
  forest: Forest,
  rowBins: Uint8Array,
  rows: number,
  features: number,
): Float32Array {
  const { numClasses, trees, shrinkage } = forest;
  const out = new Float32Array(rows * numClasses);

  if (forest.kind === "forest-classify") {
    if (trees.length === 0) return out.fill(1 / numClasses);
    for (let i = 0; i < rows; i++) {
      for (const tree of trees) {
        const off = leafOffset(tree, rowBins, i * features);
        for (let c = 0; c < numClasses; c++) out[i * numClasses + c] += tree.value[off + c];
      }
      for (let c = 0; c < numClasses; c++) out[i * numClasses + c] /= trees.length;
    }
    return out;
  }

  // Boosting: additive scores per class, then one softmax.
  const scores = new Float32Array(numClasses);
  const rounds = Math.floor(trees.length / numClasses);
  for (let i = 0; i < rows; i++) {
    scores.fill(0);
    for (let t = 0; t < rounds; t++) {
      for (let c = 0; c < numClasses; c++) {
        const tree = trees[t * numClasses + c];
        scores[c] += shrinkage * tree.value[leafOffset(tree, rowBins, i * features)];
      }
    }
    let max = -Infinity;
    for (let c = 0; c < numClasses; c++) max = Math.max(max, scores[c]);
    let sum = 0;
    for (let c = 0; c < numClasses; c++) {
      const e = Math.exp(scores[c] - max);
      out[i * numClasses + c] = e;
      sum += e;
    }
    for (let c = 0; c < numClasses; c++) out[i * numClasses + c] /= sum;
  }
  return out;
}

/** Point predictions for a regression forest or booster. */
export function predictValue(
  forest: Forest,
  rowBins: Uint8Array,
  rows: number,
  features: number,
): Float32Array {
  const out = new Float32Array(rows);
  const { trees, shrinkage, base, kind } = forest;
  for (let i = 0; i < rows; i++) {
    let sum = kind === "boost-regress" ? base : 0;
    for (const tree of trees) {
      sum += shrinkage * tree.value[leafOffset(tree, rowBins, i * features)];
    }
    if (kind === "forest-regress" && trees.length > 0) sum /= trees.length;
    out[i] = sum;
  }
  return out;
}
