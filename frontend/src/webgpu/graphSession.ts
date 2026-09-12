// The worker-side owner of the /graph route's dataset and training runs.
//
// It lives in its own module rather than inline in worker.ts for the usual
// reason: everything here is a plain message handler over injected pieces, so it
// can be reasoned about — and most of it unit-tested — without a Worker or a GPU.
//
// Two things are deliberately kept **inside** the worker and never posted:
//
//   - **The feature matrix.** Cora's features are 2708x1433 dense, which is
//     15.5 MB; the page has no use for them, and structured-cloning them once
//     per run would cost more than an epoch does.
//   - **The layout.** It is the most expensive thing on the page — about a
//     second — and it depends only on the graph, so it is computed on load and
//     reused for every subsequent run. Changing the architecture or the depth
//     re-trains; it must never re-lay-out, or the user loses the mental map they
//     were reading the result against.

import {
  loadCora,
  makeSplit,
  type CoraGraph,
  type GraphSplit,
  type SplitOptions,
} from "@/lib/cora";
import { forceLayout, type GraphLayout } from "@/lib/graphLayout";

import { mulberry32 } from "@/lib/random";

import { detectWebGPU } from "./capabilities";
import { AttentionPropagator } from "./gat";
import {
  archScales,
  cpuMatmulFn,
  fitGnn,
  type GnnArch,
  type GnnInput,
  type GnnMetrics,
  type GnnOps,
  GnnTrainer,
  isScaledArch,
  makeCpuAggregate,
  makeNeighbourSmoothness,
  prepareInput,
  type Propagator,
  scaledGatherPropagator,
} from "./gnn";
import { GraphAggregator, gpuMatmul } from "./gnnRuntime";

/** Which of the two compute paths a run actually used. */
export type GraphBackend = "webgpu" | "cpu";

/** What the page needs in order to draw the graph. Everything else stays here. */
export interface GraphSummary {
  nNodes: number;
  nFeat: number;
  nClasses: number;
  /** Directed entry count — each undirected citation appears twice. */
  nEdges: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  labels: Uint8Array;
  trainMask: Uint8Array;
  x: Float32Array;
  y: Float32Array;
  /** Milliseconds the layout took, which is the whole of the load. */
  layoutMs: number;
  /** What a training run would use, probed before anything is dispatched. */
  backend: GraphBackend;
}

export interface GraphTrainRequest {
  arch: GnnArch;
  /** Number of weight matrices; the depth slider's value. */
  layers: number;
  hidden: number;
  learningRate: number;
  weightDecay: number;
  dropout: number;
  epochs: number;
  seed?: number;
}

export interface GraphTrainResult {
  /**
   * Null when the run was stopped before its first epoch finished. Cancelling is
   * a normal outcome, not a failure — throwing here would put "training produced
   * no metrics" in the error slot every time someone pressed Stop quickly.
   */
  metrics: GnnMetrics | null;
  predictions: Uint8Array;
  backend: GraphBackend;
  elapsedMs: number;
}

interface LoadedGraph {
  graph: CoraGraph;
  input: GnnInput;
  split: GraphSplit;
  layout: GraphLayout;
  layoutMs: number;
}

/**
 * Owns the loaded dataset for the lifetime of the worker. One page, one worker,
 * one graph — so a second `load` is a no-op rather than a second layout.
 */
export class GraphSession {
  private loaded: LoadedGraph | null = null;
  private backend: GraphBackend | null = null;

  /**
   * `split` defaults to the semi-supervised setting Cora is always quoted in —
   * 20 labelled nodes per class, 500 validation, 1000 test. It is an option so a
   * test can drive the session with a graph small enough to reason about.
   */
  constructor(private readonly split: SplitOptions = {}) {}

  /**
   * Fetch, decode, split and lay out the graph. Idempotent: calling it again
   * returns the same summary without recomputing the layout.
   */
  async load(): Promise<GraphSummary> {
    this.loaded ??= await this.prepare();
    this.backend ??= (await detectWebGPU()).status === "ready" ? "webgpu" : "cpu";
    const { graph, split, layout, layoutMs } = this.loaded;

    return {
      nNodes: graph.nNodes,
      nFeat: graph.nFeat,
      nClasses: graph.nClasses,
      nEdges: graph.colIdx.length,
      // Copies, because these are transferred to the page and the session keeps
      // using its own.
      rowPtr: graph.rowPtr.slice(),
      colIdx: graph.colIdx.slice(),
      labels: graph.labels.slice(),
      trainMask: split.trainMask.slice(),
      x: layout.x.slice(),
      y: layout.y.slice(),
      layoutMs,
      backend: this.backend,
    };
  }

  private async prepare(): Promise<LoadedGraph> {
    const graph = await loadCora();
    const started = now();
    const layout = forceLayout(graph.rowPtr, graph.colIdx, graph.nNodes);
    const layoutMs = now() - started;
    return {
      graph,
      input: prepareInput(graph.features, graph.nNodes, graph.nFeat),
      split: makeSplit(graph.labels, graph.nClasses, this.split),
      layout,
      layoutMs,
    };
  }

  /**
   * Train one model. The aggregator is built per run because the scale vectors
   * are a property of the architecture, and disposed at the end because it holds
   * device buffers — but the graph it uploads is the same one every time.
   */
  async train(
    request: GraphTrainRequest,
    onEpoch?: (metrics: GnnMetrics, predictions: Uint8Array) => void,
    shouldStop?: () => boolean,
  ): Promise<GraphTrainResult> {
    if (!this.loaded) throw new Error("load the graph before training");
    const { graph, input, split } = this.loaded;
    const backend = this.backend ?? "cpu";
    const smoothness = makeNeighbourSmoothness(graph.rowPtr, graph.colIdx);

    // The CPU path is not a stub: it is the same reference implementation the
    // unit tests check the WGSL kernel against, so a machine without WebGPU runs
    // the identical arithmetic, more slowly. There is no third option — a
    // hand-written shader has no ONNX Runtime to fall back to.
    const matmul = backend === "webgpu" ? gpuMatmul : cpuMatmulFn;

    let aggregator: GraphAggregator | null = null;
    let propagator: (nFeat: number) => Propagator;

    if (isScaledArch(request.arch)) {
      const { alpha, beta } = archScales(request.arch, graph.degree);
      if (backend === "webgpu") {
        aggregator = await GraphAggregator.create(
          graph.rowPtr,
          graph.colIdx,
          graph.nNodes,
          alpha,
          beta,
        );
        const gather = aggregator.aggregate;
        propagator = () => scaledGatherPropagator(gather);
      } else {
        const gather = makeCpuAggregate(
          graph.rowPtr,
          graph.colIdx,
          graph.nNodes,
          alpha,
          beta,
        );
        propagator = () => scaledGatherPropagator(gather);
      }
    } else {
      // GAT's coefficients are learned per edge, so it does not go through the
      // aggregation shader at all; its projections still use `matmul` above,
      // which is the expensive half. See gat.ts for the full argument.
      const rand = mulberry32((request.seed ?? 42) ^ 0x9e3779b9);
      propagator = (nFeat) =>
        new AttentionPropagator(
          graph.rowPtr,
          graph.colIdx,
          graph.nNodes,
          nFeat,
          rand,
        );
    }

    const ops: GnnOps = { matmul, propagator, smoothness };

    const started = now();
    try {
      const trainer = new GnnTrainer(
        ops,
        {
          nNodes: graph.nNodes,
          nFeat: graph.nFeat,
          nClasses: graph.nClasses,
          hidden: request.hidden,
          layers: request.layers,
        },
        request.seed,
      );

      let lastPredictions: Uint8Array<ArrayBufferLike> = new Uint8Array(
        graph.nNodes,
      );
      const metrics = await fitGnn(
        trainer,
        input,
        graph.labels,
        split,
        {
          arch: request.arch,
          learningRate: request.learningRate,
          weightDecay: request.weightDecay,
          dropout: request.dropout,
          epochs: request.epochs,
        },
        {
          onEpoch: (m, predictions) => {
            lastPredictions = predictions;
            onEpoch?.(m, predictions);
          },
          shouldStop,
        },
      );

      return {
        metrics,
        predictions: lastPredictions,
        backend,
        elapsedMs: now() - started,
      };
    } finally {
      aggregator?.dispose();
    }
  }
}

const now = () =>
  typeof performance !== "undefined" ? performance.now() : Date.now();
