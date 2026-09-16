// The worker-side owner of the /graph-classification route's dataset and runs.
//
// The third sibling of GraphSession, and the only one with a **real LOAD state**:
// /graph and /link-prediction both read a 161 KB binary bundled with the app,
// while this fetches 2.06 MB from the Hub. That download — and the parse and
// union-building behind it — is what the LOAD slot is actually reporting, and
// why `lib/proteinsCache.ts` exists: the second visit skips all three.
//
// What does *not* differ is the compute. The union of all 1113 graphs is 43 471
// nodes and 162 088 directed entries, so it is one CSR uploaded once and trained
// full-batch, through exactly the aggregation kernel, matmul and propagators the
// other two routes use. There is no mini-batching here and no second code path.
//
// Layouts are the one thing computed lazily. /graph lays out its 2708 nodes once
// because the page draws all of them; here the page draws a gallery of a few
// dozen small graphs, and laying out all 1113 up front would be a second of work
// for pictures nobody has asked for. `layoutFor` computes one graph's coordinates
// on demand and remembers it.

import {
  buildUnion,
  fetchProteins,
  majorityBaseline,
  makeGraphSplit,
  type GraphSplitIndices,
  type ProteinSplitOptions,
  type ProteinUnion,
} from "@/lib/proteins";
import { loadCachedUnion, saveCachedUnion } from "@/lib/proteinsCache";
import { forceLayout } from "@/lib/graphLayout";
import { mulberry32 } from "@/lib/random";

import { detectWebGPU } from "./capabilities";
import { AttentionPropagator } from "./gat";
import {
  archScales,
  cpuMatmulFn,
  type GnnArch,
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
import type { GraphBackend } from "./graphSession";
import {
  fitGraphClassifier,
  type GraphClassMetrics,
  type ReadoutMode,
} from "./graphPool";

/** What the page needs to describe the dataset it just loaded. */
export interface ProteinSummary {
  nGraphs: number;
  nNodes: number;
  nFeat: number;
  nClasses: number;
  /** Directed entry count across the whole union. */
  nEdges: number;
  nTrain: number;
  nVal: number;
  nTest: number;
  /** One label per graph, so the gallery can say what each protein really is. */
  labels: Uint8Array;
  /** Node ranges, so the gallery can size each graph without the whole union. */
  graphPtr: Uint32Array;
  /** The test graphs, which are the ones the gallery draws. */
  testIdx: Uint32Array;
  /** What a model that ignores its input would score on that test split. */
  baselineAcc: number;
  /** True when the union came out of IndexedDB rather than the network. */
  fromCache: boolean;
  loadMs: number;
  backend: GraphBackend;
}

export interface ProteinTrainRequest {
  arch: GnnArch;
  readout: ReadoutMode;
  layers: number;
  hidden: number;
  learningRate: number;
  weightDecay: number;
  dropout: number;
  epochs: number;
  seed?: number;
}

export interface ProteinTrainResult {
  /** Null when the run was stopped before its first epoch — a normal outcome. */
  metrics: GraphClassMetrics | null;
  /** The class each graph ended up predicted to be. */
  predicted: Uint8Array;
  backend: GraphBackend;
  elapsedMs: number;
}

/** One graph's drawable form: local CSR plus unit-square coordinates. */
export interface GraphLayoutPayload {
  index: number;
  nNodes: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  x: Float32Array;
  y: Float32Array;
}

interface LoadedProteins {
  union: ProteinUnion;
  split: GraphSplitIndices;
  baselineAcc: number;
  fromCache: boolean;
  loadMs: number;
}

export class ProteinSession {
  private loaded: LoadedProteins | null = null;
  private backend: GraphBackend | null = null;
  private readonly layouts = new Map<number, GraphLayoutPayload>();

  constructor(private readonly splitOptions: ProteinSplitOptions = {}) {}

  /**
   * Fetch (or read the cache), build the union, split by graph. Idempotent: a
   * second call returns the same summary without a second download.
   */
  async load(): Promise<ProteinSummary> {
    this.loaded ??= await this.prepare();
    this.backend ??= (await detectWebGPU()).status === "ready" ? "webgpu" : "cpu";

    const { union, split, baselineAcc, fromCache, loadMs } = this.loaded;
    return {
      nGraphs: union.nGraphs,
      nNodes: union.nNodes,
      nFeat: union.nFeat,
      nClasses: union.nClasses,
      nEdges: union.colIdx.length,
      nTrain: split.train.length,
      nVal: split.val.length,
      nTest: split.test.length,
      // Copies, because these are transferred to the page.
      labels: union.labels.slice(),
      graphPtr: union.graphPtr.slice(),
      testIdx: split.test.slice(),
      baselineAcc,
      fromCache,
      loadMs,
      backend: this.backend,
    };
  }

  private async prepare(): Promise<LoadedProteins> {
    const started = now();
    const cached = await loadCachedUnion();
    const union = cached ?? buildUnion(await fetchProteins());
    if (!cached) {
      // Best-effort, and deliberately not awaited for correctness: a quota
      // failure must not fail the load.
      void saveCachedUnion(union);
    }

    const split = makeGraphSplit(union.labels, union.nClasses, this.splitOptions);
    const { accuracy } = majorityBaseline(
      union.labels,
      split.train,
      split.test,
      union.nClasses,
    );
    return {
      union,
      split,
      baselineAcc: accuracy,
      fromCache: cached !== null,
      loadMs: now() - started,
    };
  }

  /**
   * One graph's drawable form, laid out on demand and remembered.
   *
   * The indices are rebased to the graph's own 0…n-1 so the canvas never sees
   * the union's global numbering — a gallery tile that drew global indices would
   * point at whatever node happened to sit at that offset in the first graph.
   */
  layoutFor(index: number): GraphLayoutPayload {
    const existing = this.layouts.get(index);
    if (existing) return existing;
    if (!this.loaded) throw new Error("load the dataset before laying it out");

    const { union } = this.loaded;
    if (index < 0 || index >= union.nGraphs) {
      throw new Error(`no graph ${index}: the dataset has ${union.nGraphs}`);
    }
    const base = union.graphPtr[index];
    const nNodes = union.graphPtr[index + 1] - base;

    const rowPtr = new Uint32Array(nNodes + 1);
    for (let i = 0; i < nNodes; i++) {
      rowPtr[i + 1] =
        rowPtr[i] + (union.rowPtr[base + i + 1] - union.rowPtr[base + i]);
    }
    const colIdx = new Uint32Array(rowPtr[nNodes]);
    let at = 0;
    for (let i = 0; i < nNodes; i++) {
      for (let e = union.rowPtr[base + i]; e < union.rowPtr[base + i + 1]; e++) {
        colIdx[at++] = union.colIdx[e] - base;
      }
    }

    const layout = forceLayout(rowPtr, colIdx, nNodes);
    const payload: GraphLayoutPayload = {
      index,
      nNodes,
      rowPtr,
      colIdx,
      x: layout.x,
      y: layout.y,
    };
    this.layouts.set(index, payload);
    return payload;
  }

  async train(
    request: ProteinTrainRequest,
    onEpoch?: (metrics: GraphClassMetrics, predicted: Uint8Array) => void,
    shouldStop?: () => boolean,
  ): Promise<ProteinTrainResult> {
    if (!this.loaded) throw new Error("load the dataset before training");
    const { union, split, baselineAcc } = this.loaded;
    const backend = this.backend ?? "cpu";
    const matmul = backend === "webgpu" ? gpuMatmul : cpuMatmulFn;

    let aggregator: GraphAggregator | null = null;
    let propagator: (nFeat: number) => Propagator;

    if (isScaledArch(request.arch)) {
      const { alpha, beta } = archScales(request.arch, union.degree);
      if (backend === "webgpu") {
        aggregator = await GraphAggregator.create(
          union.rowPtr,
          union.colIdx,
          union.nNodes,
          alpha,
          beta,
        );
        const gather = aggregator.aggregate;
        propagator = () => scaledGatherPropagator(gather);
      } else {
        const gather = makeCpuAggregate(
          union.rowPtr,
          union.colIdx,
          union.nNodes,
          alpha,
          beta,
        );
        propagator = () => scaledGatherPropagator(gather);
      }
    } else {
      const rand = mulberry32((request.seed ?? 42) ^ 0x9e3779b9);
      propagator = (nFeat) =>
        new AttentionPropagator(
          union.rowPtr,
          union.colIdx,
          union.nNodes,
          nFeat,
          rand,
        );
    }

    const ops: GnnOps = {
      matmul,
      propagator,
      smoothness: makeNeighbourSmoothness(union.rowPtr, union.colIdx),
    };

    const started = now();
    try {
      const trainer = new GnnTrainer(
        ops,
        {
          nNodes: union.nNodes,
          nFeat: union.nFeat,
          // The output width is the class count: the readout pools the *logits*,
          // which is the same function as pooling then classifying. See
          // graphPool.ts.
          nClasses: union.nClasses,
          hidden: request.hidden,
          layers: request.layers,
        },
        request.seed,
      );
      trainer.setInput(
        prepareInput(union.features, union.nNodes, union.nFeat),
      );

      let lastPredicted: Uint8Array<ArrayBufferLike> = new Uint8Array(
        union.nGraphs,
      );
      const metrics = await fitGraphClassifier(
        trainer,
        union.graphPtr,
        union.labels,
        split,
        baselineAcc,
        {
          arch: request.arch,
          readout: request.readout,
          learningRate: request.learningRate,
          weightDecay: request.weightDecay,
          dropout: request.dropout,
          epochs: request.epochs,
        },
        {
          onEpoch: (m, predicted) => {
            lastPredicted = predicted;
            onEpoch?.(m, predicted);
          },
          shouldStop,
        },
      );

      return {
        metrics,
        predicted: lastPredicted,
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
