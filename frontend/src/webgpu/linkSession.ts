// The worker-side owner of the /link-prediction route's dataset and runs.
//
// The sibling of graphSession.ts, and deliberately a sibling rather than a mode
// on it: the two pages hold different graphs. `GraphSession` propagates over the
// whole of Cora, because node classification hides *labels*, not edges. This one
// hides edges, so the graph it trains on is missing 15 % of its citations and
// the one it samples negatives against is not.
//
// Everything expensive is still kept inside the worker — the 15.5 MB feature
// matrix, and the layout — but the **embeddings do** cross postMessage at the
// end of a run. 2708 × 32 floats is 346 KB, and having them on the page is what
// makes clicking two nodes to see their score a local dot product instead of a
// round trip through a worker that may be busy training.
//
// One choice worth stating because either would have been defensible: the layout
// is computed from the **training** graph. A layout is a drawing, not a model
// input, so laying out the full graph would leak nothing — but it would pull the
// endpoints of every held-out citation together, and then a correct prediction
// and a flattering picture look the same. Laying out what the model was actually
// given means a dashed line drawn across a gap is a claim about something the
// layout did not already know.

import { loadCora, type CoraGraph } from "@/lib/cora";
import { splitEdges, type EdgeList, type EdgeSplit } from "@/lib/edgeSplit";
import { forceLayout, type GraphLayout } from "@/lib/graphLayout";
import { mulberry32 } from "@/lib/random";

import { detectWebGPU } from "./capabilities";
import { AttentionPropagator } from "./gat";
import {
  archScales,
  cpuMatmulFn,
  type GnnArch,
  type GnnInput,
  type GnnOps,
  GnnTrainer,
  isScaledArch,
  makeCpuAggregate,
  makeNeighbourSmoothness,
  prepareInput,
  type Propagator,
  scaledGatherPropagator,
} from "./gnn";
import type { GraphBackend } from "./graphSession";
import { GraphAggregator, gpuMatmul } from "./gnnRuntime";
import {
  edgeScores,
  fitLinkPredictor,
  type LinkMetrics,
} from "./linkPredictor";

/** What the page needs to draw the graph and read a score off it. */
export interface LinkSummary {
  nNodes: number;
  /** Directed entry count of the **training** graph. */
  nEdges: number;
  /** Undirected citations held out for validation and for test. */
  nValEdges: number;
  nTestEdges: number;
  /** Undirected citations the model may aggregate over. */
  nTrainEdges: number;
  /** Edges kept despite being drawn, because removing them would isolate. */
  rescued: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  labels: Uint8Array;
  x: Float32Array;
  y: Float32Array;
  layoutMs: number;
  backend: GraphBackend;
}

export interface LinkTrainRequest {
  arch: GnnArch;
  layers: number;
  hidden: number;
  /** Width of the embedding the decoder scores pairs from. */
  embedding: number;
  learningRate: number;
  weightDecay: number;
  dropout: number;
  epochs: number;
  /** How many of the top-scoring non-edges to bring back for drawing. */
  topK: number;
  seed?: number;
}

export interface LinkTrainResult {
  /** Null when the run was stopped before its first epoch — a normal outcome. */
  metrics: LinkMetrics | null;
  /** The highest-scoring pairs that are **not** citations, best first. */
  candidates: EdgeList;
  candidateScores: Float32Array;
  /** Final embeddings, so the page can score a clicked pair itself. */
  embedding: Float32Array;
  embeddingDim: number;
  backend: GraphBackend;
  elapsedMs: number;
}

interface LoadedLink {
  graph: CoraGraph;
  input: GnnInput;
  split: EdgeSplit;
  layout: GraphLayout;
  layoutMs: number;
}

/**
 * Held-out fractions are a property of the *split*, so changing one has to
 * rebuild the graph and the layout with it. That is why it is a load parameter
 * rather than a training one, and why the page calls it out as the control that
 * costs a reload.
 */
export interface LinkLoadOptions {
  testFrac?: number;
  valFrac?: number;
  seed?: number;
}

export class LinkSession {
  private loaded: LoadedLink | null = null;
  private loadedKey = "";
  private backend: GraphBackend | null = null;

  /**
   * Fetch, decode, split and lay out. Idempotent **for the same split**: asking
   * for a different held-out fraction is a different graph, so it re-splits and
   * re-lays-out, and asking for the same one again does neither.
   */
  async load(options: LinkLoadOptions = {}): Promise<LinkSummary> {
    const key = JSON.stringify([
      options.testFrac ?? 0.1,
      options.valFrac ?? 0.05,
      options.seed ?? 7,
    ]);
    if (!this.loaded || this.loadedKey !== key) {
      this.loaded = await this.prepare(options);
      this.loadedKey = key;
    }
    this.backend ??= (await detectWebGPU()).status === "ready" ? "webgpu" : "cpu";

    const { graph, split, layout, layoutMs } = this.loaded;
    return {
      nNodes: graph.nNodes,
      nEdges: split.colIdx.length,
      nTrainEdges: split.trainPos.length / 2,
      nValEdges: split.valPos.length / 2,
      nTestEdges: split.testPos.length / 2,
      rescued: split.rescued,
      // Copies: these are transferred to the page and the session keeps its own.
      rowPtr: split.rowPtr.slice(),
      colIdx: split.colIdx.slice(),
      labels: graph.labels.slice(),
      x: layout.x.slice(),
      y: layout.y.slice(),
      layoutMs,
      backend: this.backend,
    };
  }

  private async prepare(options: LinkLoadOptions): Promise<LoadedLink> {
    const graph = await loadCora();
    const split = splitEdges(graph.rowPtr, graph.colIdx, graph.nNodes, options);
    const started = now();
    const layout = forceLayout(split.rowPtr, split.colIdx, graph.nNodes);
    const layoutMs = now() - started;
    return {
      graph,
      input: prepareInput(graph.features, graph.nNodes, graph.nFeat),
      split,
      layout,
      layoutMs,
    };
  }

  async train(
    request: LinkTrainRequest,
    onEpoch?: (metrics: LinkMetrics) => void,
    shouldStop?: () => boolean,
  ): Promise<LinkTrainResult> {
    if (!this.loaded) throw new Error("load the graph before training");
    const { graph, input, split } = this.loaded;
    const backend = this.backend ?? "cpu";
    const matmul = backend === "webgpu" ? gpuMatmul : cpuMatmulFn;

    // Every one of these reads the **training** CSR and the degrees rebuilt with
    // it. Handing any of them the full graph would put the held-out citations
    // back into the model's reach — through the gather for the propagator, and
    // through D^-1/2 for the scales.
    let aggregator: GraphAggregator | null = null;
    let propagator: (nFeat: number) => Propagator;

    if (isScaledArch(request.arch)) {
      const { alpha, beta } = archScales(request.arch, split.degree);
      if (backend === "webgpu") {
        aggregator = await GraphAggregator.create(
          split.rowPtr,
          split.colIdx,
          graph.nNodes,
          alpha,
          beta,
        );
        const gather = aggregator.aggregate;
        propagator = () => scaledGatherPropagator(gather);
      } else {
        const gather = makeCpuAggregate(
          split.rowPtr,
          split.colIdx,
          graph.nNodes,
          alpha,
          beta,
        );
        propagator = () => scaledGatherPropagator(gather);
      }
    } else {
      const rand = mulberry32((request.seed ?? 42) ^ 0x9e3779b9);
      propagator = (nFeat) =>
        new AttentionPropagator(
          split.rowPtr,
          split.colIdx,
          graph.nNodes,
          nFeat,
          rand,
        );
    }

    const ops: GnnOps = {
      matmul,
      propagator,
      smoothness: makeNeighbourSmoothness(split.rowPtr, split.colIdx),
    };

    const started = now();
    try {
      const trainer = new GnnTrainer(
        ops,
        {
          nNodes: graph.nNodes,
          nFeat: graph.nFeat,
          // The network's output width. There is no classifier here — the last
          // layer *is* the embedding the decoder scores pairs from.
          nClasses: request.embedding,
          hidden: request.hidden,
          layers: request.layers,
        },
        request.seed,
      );

      const metrics = await fitLinkPredictor(
        trainer,
        input,
        split,
        {
          arch: request.arch,
          learningRate: request.learningRate,
          weightDecay: request.weightDecay,
          dropout: request.dropout,
          epochs: request.epochs,
          // Negatives are rejected against the full graph, so a held-out
          // citation is never handed to the model as a non-edge.
          fullRowPtr: graph.rowPtr,
          fullColIdx: graph.colIdx,
          seed: request.seed,
        },
        { onEpoch, shouldStop },
      );

      const pass = await trainer.forward(0, false);
      const embedding = pass.logits;
      const dim = trainer.dims[trainer.dims.length - 1];

      // Scored once, at the end. Per epoch this would be 3.7 M dot products
      // against a picture nobody is reading yet.
      const { pairs, scores } = topCandidates(
        embedding,
        dim,
        graph.nNodes,
        graph.rowPtr,
        graph.colIdx,
        request.topK,
      );

      return {
        metrics,
        candidates: pairs,
        candidateScores: scores,
        embedding: embedding.slice(),
        embeddingDim: dim,
        backend,
        elapsedMs: now() - started,
      };
    } finally {
      aggregator?.dispose();
    }
  }
}

/**
 * The `k` highest-scoring pairs that are not already citations.
 *
 * Every pair is considered — 3.7 M of them on Cora — but only `k` are kept, in
 * an insertion-sorted list rather than a sort of the whole set: `k` is a handful
 * of dozens, and materialising 3.7 M scores to sort them would cost 30 MB to
 * throw away all but 50.
 *
 * Pairs already in the graph are skipped rather than scored and filtered. The
 * model ranks real citations highly — that is what a working model *is* — so
 * leaving them in would fill the entire list with edges the user can already
 * see, and the page would look like it had predicted nothing.
 */
export function topCandidates(
  z: Float32Array,
  dim: number,
  nNodes: number,
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  k: number,
): { pairs: EdgeList; scores: Float32Array } {
  if (k <= 0) return { pairs: new Uint32Array(0), scores: new Float32Array(0) };

  const bestU = new Uint32Array(k);
  const bestV = new Uint32Array(k);
  const bestS = new Float32Array(k).fill(-Infinity);
  let held = 0;
  let floor = -Infinity;

  for (let u = 0; u < nNodes; u++) {
    const a = u * dim;
    let at = rowPtr[u];
    const end = rowPtr[u + 1];
    for (let v = u + 1; v < nNodes; v++) {
      // The row is ascending, so walking it alongside v is a merge rather than
      // a search per pair.
      while (at < end && colIdx[at] < v) at++;
      if (at < end && colIdx[at] === v) continue;

      const b = v * dim;
      let score = 0;
      for (let f = 0; f < dim; f++) score += z[a + f] * z[b + f];
      if (held === k && score <= floor) continue;

      // Insertion into a descending list of at most k entries.
      let i = held < k ? held++ : k - 1;
      while (i > 0 && bestS[i - 1] < score) {
        bestS[i] = bestS[i - 1];
        bestU[i] = bestU[i - 1];
        bestV[i] = bestV[i - 1];
        i--;
      }
      bestS[i] = score;
      bestU[i] = u;
      bestV[i] = v;
      if (held === k) floor = bestS[k - 1];
    }
  }

  const pairs = new Uint32Array(held * 2);
  for (let i = 0; i < held; i++) {
    pairs[2 * i] = bestU[i];
    pairs[2 * i + 1] = bestV[i];
  }
  return { pairs, scores: bestS.slice(0, held) };
}

/** Score one pair from embeddings the page already holds. */
export function scorePair(
  z: Float32Array,
  dim: number,
  u: number,
  v: number,
): number {
  return edgeScores(z, dim, Uint32Array.from([u, v]))[0];
}

const now = () =>
  typeof performance !== "undefined" ? performance.now() : Date.now();
