// The fit worker's message-handling core, factored out of `fit.worker.ts` so it
// can be unit-tested with a CPU matmul and no Worker at all — the same shape as
// `text/engine.ts`, `vision/engine.ts` and `audio/pipelineEngine.ts`.
//
// Two of the three obligations those engines owe do not apply here, and saying
// why is more useful than quietly omitting them:
//
//   one model live at a time   There is no model to leak. A fitted forest is
//                              kilobytes of typed arrays and holds no GPU
//                              session; the device itself is memoised in
//                              `webgpu/device.ts` and is shared by every kernel
//                              in this realm, so destroying it here would break
//                              the rest of the app to free nothing.
//   warm-up on load            There are no shaders to compile until a fit asks
//                              for one, and `load` is handed data rather than
//                              weights. The warm-up that matters — the first
//                              WGSL dispatch — happens inside the first fit,
//                              which is why the FIT band reports a phase.
//
// The third — **never block the main thread** — is the only reason this file is
// in a worker at all, and it is the one that binds hardest: a 200 000-row CSV
// parsed and a gradient booster fitted on the main thread freezes the tab for
// seconds with no way to say so.

import { encodeOne, encodeRows, fitEncoder, type Encoder } from "./design";
import { permutationImportance } from "./importance";
import { fitLogistic, type LinearModel, type MatmulFn } from "./linear";
import { classificationMetrics, predictedClasses } from "./metrics";
import { fitMlp, type MlpModel } from "./mlp";
import { splitRows } from "./split";
import {
  applyBinning,
  binFeatures,
  fitBoostedClassifier,
  fitForestClassifier,
  predictProba,
  type Binning,
  type Forest,
} from "./trees";
import type {
  Dataset,
  FitPartial,
  FitResult,
  FitSpec,
  PredictRequest,
  PredictResult,
  TabularRequest,
  TabularResponse,
} from "./types";

export type Post = (message: TabularResponse) => void;

/** Injected so a test fits on the CPU reference and the worker fits on WGSL. */
export interface EngineDeps {
  /** Resolves to the GPU matmul, or rejects if there is no device. */
  gpuMatmul: () => Promise<MatmulFn>;
  /** The reference implementation, and the fallback when there is no GPU. */
  cpuMatmul: MatmulFn;
}

interface Fitted {
  spec: FitSpec;
  encoder: Encoder;
  binning: Binning | null;
  forest: Forest | null;
  linear: LinearModel | null;
  mlp: MlpModel | null;
  labels: string[];
}

export function createFitHandler(post: Post, deps: EngineDeps) {
  let dataset: Dataset | null = null;
  let fitted: Fitted | null = null;
  // Set by the out-of-band `stop` message and polled by every fit loop. Cleared
  // when a run starts, so a stop that arrives after a fit has already finished
  // cannot cancel the next one.
  let stopped = false;

  async function resolveMatmul(wantsGpu: boolean): Promise<{
    matmul: MatmulFn;
    compute: "gpu" | "cpu";
  }> {
    if (!wantsGpu) return { matmul: deps.cpuMatmul, compute: "cpu" };
    try {
      return { matmul: await deps.gpuMatmul(), compute: "gpu" };
    } catch {
      // A machine with no WebGPU still gets the whole ladder. The page says
      // which half ran where, so falling back silently would make the one
      // sentence this category is about — "this bit is on your GPU and this bit
      // deliberately is not" — quietly untrue.
      return { matmul: deps.cpuMatmul, compute: "cpu" };
    }
  }

  // The fit's iteration counter, rate-limited to ~20 posts a second.
  //
  // `useLinearTraining` rAF-batches its metric flush for this reason and the
  // batching has moved one step earlier: `useModelWorker` sets React state
  // directly from each `partial`, so a booster posting per tree would re-render
  // the page a hundred times in eight seconds — and throttling here also skips
  // the `postMessage` rather than merely skipping the render. A phase change or
  // the final iteration always gets through, or the band would stall on
  // "Fitting" at 118 of 120.
  let lastPost = 0;
  let lastPhase = "";
  function progress(id: number, partial: FitPartial) {
    const now = performance.now();
    const boundary = partial.phase !== lastPhase || partial.done >= partial.total;
    if (!boundary && now - lastPost < 50) return;
    lastPost = now;
    lastPhase = partial.phase;
    post({ type: "partial", id, partial });
  }

  async function fit(id: number, spec: FitSpec): Promise<FitResult> {
    if (!dataset) throw new Error("No dataset loaded.");
    stopped = false;
    const shouldStop = () => stopped;
    const started = performance.now();
    const target = dataset.columns[spec.targetIndex];
    if (!target) throw new Error("Pick a target column.");
    if (target.kind !== "categorical") {
      throw new Error(
        `"${target.name}" is numeric — classification needs a column with a small set of distinct values.`,
      );
    }
    const labels = target.levels ?? [];
    if (labels.length < 2) throw new Error("The target column has only one value.");

    progress(id, { done: 0, total: 1, phase: "Encoding", loss: null });

    // The split comes first, and everything fitted from the data is fitted on
    // its training half — see `design.ts`. Stratified on the target so a rare
    // class is present on both sides.
    const split = splitRows(dataset.rowCount, spec.testFraction, spec.seed, target.values);
    const encoder = fitEncoder(dataset, spec.featureIndices, split.train);

    const yTrain = new Uint8Array(split.train.length);
    for (let i = 0; i < split.train.length; i++) yTrain[i] = target.values[split.train[i]];
    const yTest = new Uint8Array(split.test.length);
    for (let i = 0; i < split.test.length; i++) yTest[i] = target.values[split.test[i]];

    const info = spec.family === "logistic" || spec.family === "mlp";
    const { matmul, compute } = await resolveMatmul(info);

    const curve: { step: number; loss: number }[] = [];
    const record = (done: number, total: number, loss: number | null) => {
      if (loss != null && Number.isFinite(loss)) curve.push({ step: done, loss });
      progress(id, { done, total, phase: "Fitting", loss });
    };

    let binning: Binning | null = null;
    let forest: Forest | null = null;
    let linear: LinearModel | null = null;
    let mlp: MlpModel | null = null;
    let probabilities: Float32Array;

    if (spec.family === "forest" || spec.family === "boosting") {
      // Trees read the columns in their own units — a split at `age ≤ 38` is
      // readable and a split at `z ≤ 0.4` is not — so no standardisation.
      const xTrain = encodeRows(dataset, encoder, split.train, false);
      binning = binFeatures(xTrain, split.train.length, encoder.columns.length);
      const rows = new Int32Array(split.train.length);
      for (let i = 0; i < rows.length; i++) rows[i] = i;
      forest =
        spec.family === "forest"
          ? fitForestClassifier(binning, rows, yTrain, labels.length, spec.hp.nTrees, treeParams(spec), spec.seed, {
              onProgress: record,
              shouldStop,
            })
          : fitBoostedClassifier(binning, rows, yTrain, labels.length, spec.hp.nTrees, spec.hp.shrinkage, treeParams(spec), spec.seed, {
              onProgress: record,
              shouldStop,
            });
      const xTest = encodeRows(dataset, encoder, split.test, false);
      const testBins = applyBinning(binning, xTest, split.test.length);
      probabilities = predictProba(forest, testBins, split.test.length, encoder.columns.length);
    } else {
      const xTrain = encodeRows(dataset, encoder, split.train, true);
      const xTest = encodeRows(dataset, encoder, split.test, true);
      if (spec.family === "logistic") {
        linear = await fitLogistic(matmul, xTrain, yTrain, split.train.length, encoder.columns.length, labels.length, {
          epochs: spec.hp.epochs,
          learningRate: spec.hp.learningRate,
          batchSize: spec.hp.batchSize,
          seed: spec.seed,
          onProgress: record,
          shouldStop,
        });
        probabilities = await linear.predict(xTest, split.test.length);
      } else {
        mlp = await fitMlp(matmul, xTrain, yTrain, split.train.length, encoder.columns.length, labels.length, {
          hidden: spec.hp.hidden,
          epochs: spec.hp.epochs,
          learningRate: spec.hp.learningRate,
          batchSize: spec.hp.batchSize,
          seed: spec.seed,
          onProgress: record,
          shouldStop,
        });
        probabilities = await mlp.predict(xTest, split.test.length);
      }
    }

    const fitMs = performance.now() - started;
    progress(id, { done: 1, total: 1, phase: "Scoring", loss: null });

    const predicted = predictedClasses(probabilities, split.test.length, labels.length);
    const metrics = classificationMetrics(yTest, predicted, labels, yTrain);

    fitted = { spec, encoder, binning, forest, linear, mlp, labels };

    // Permutation importance, on the held-out half. Measuring it on the
    // training half would report what the model memorised rather than what it
    // uses, and on a deep tree those are very different rankings.
    const xTestRaw = encodeRows(dataset, encoder, split.test, spec.family === "logistic" || spec.family === "mlp");
    const sourceNames = new Map<number, string>();
    for (const source of encoder.used) sourceNames.set(source, dataset.columns[source].name);
    const scoreFn = async (matrix: Float32Array): Promise<number> => {
      const probs = await scoreMatrix(matrix, split.test.length);
      const p = predictedClasses(probs, split.test.length, labels.length);
      let hits = 0;
      for (let i = 0; i < p.length; i++) if (p[i] === yTest[i]) hits++;
      return hits / Math.max(1, p.length);
    };
    const scoreMatrix = async (matrix: Float32Array, n: number): Promise<Float32Array> => {
      if (forest && binning) {
        return predictProba(forest, applyBinning(binning, matrix, n), n, encoder.columns.length);
      }
      if (linear) return linear.predict(matrix, n);
      if (mlp) return mlp.predict(matrix, n);
      return new Float32Array(n * labels.length);
    };

    const importance = await permutationImportance(
      xTestRaw,
      split.test.length,
      encoder.columns,
      sourceNames,
      scoreFn,
      {
        seed: spec.seed,
        shouldStop,
        onProgress: (done, total) =>
          progress(id, { done, total, phase: "Permuting columns", loss: null }),
      },
    );

    return {
      spec,
      trainRows: split.train.length,
      testRows: split.test.length,
      fitMs,
      compute,
      classification: metrics,
      probabilities,
      testLabels: yTest,
      trainLabels: yTrain,
      labels,
      importance,
      curve,
    };
  }

  async function predict(req: PredictRequest): Promise<PredictResult> {
    if (!dataset || !fitted) throw new Error("Fit a model first.");
    const { encoder, spec, labels } = fitted;
    const standardise = spec.family === "logistic" || spec.family === "mlp";
    const row = encodeOne(dataset, encoder, req.values, spec.featureIndices, standardise);
    let probs: Float32Array;
    if (fitted.forest && fitted.binning) {
      probs = predictProba(fitted.forest, applyBinning(fitted.binning, row, 1), 1, encoder.columns.length);
    } else if (fitted.linear) {
      probs = await fitted.linear.predict(row, 1);
    } else if (fitted.mlp) {
      probs = await fitted.mlp.predict(row, 1);
    } else {
      throw new Error("Fit a model first.");
    }
    return { scores: Array.from(probs), labels, objective: "classification" };
  }

  // Messages are handled one at a time. The main thread posts `load` and the
  // first `run` in the same tick — the FIT button does both — and a fit that
  // began before the dataset landed would fail on an empty frame.
  let chain: Promise<void> = Promise.resolve();

  return function handle(message: TabularRequest): Promise<void> {
    // `stop` jumps the queue, and has to: chained behind the fit it is meant to
    // interrupt it would be delivered after that fit had already finished,
    // which is a Stop button that does nothing except cancel the *next* run.
    if (message.type === "stop") {
      stopped = true;
      return Promise.resolve();
    }
    chain = chain.then(() => dispatch(message));
    return chain;
  };

  async function dispatch(message: TabularRequest): Promise<void> {
    if (message.type === "stop") return;
    if (message.type === "load") {
      try {
        dataset = message.dataset;
        fitted = null;
        // `backend` is what the *page* shows as the compute chip, and at load
        // time no family has been chosen, so it reports the honest thing: the
        // data is here and nothing has run. The per-fit answer travels on the
        // result, where it can name which half ran where.
        post({ type: "ready", model: message.dataset.name, backend: "wasm" });
      } catch (error) {
        post({ type: "error", error: describe(error) });
      }
      return;
    }
    const { id } = message;
    try {
      const result = "spec" in message ? await fit(id, message.spec) : await predict(message.predict);
      post({ type: "result", id, result });
    } catch (error) {
      post({ type: "error", id, error: describe(error) });
    }
  };
}

function treeParams(spec: FitSpec) {
  return {
    maxDepth: spec.hp.maxDepth,
    minLeaf: spec.hp.minLeaf,
    featureFraction: spec.hp.featureFraction,
  };
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
