// Main-thread client for the WebGPU worker. Handles worker creation and
// request/response correlation so callers get plain promises.

import { registerAllocationPort } from "./allocations";
import type {
  GraphSummary,
  GraphTrainRequest,
  GraphTrainResult,
} from "./graphSession";
import type {
  LinkLoadOptions,
  LinkSummary,
  LinkTrainRequest,
  LinkTrainResult,
} from "./linkSession";
import type { LinkMetrics } from "./linkPredictor";
import type { GraphClassMetrics } from "./graphPool";
import type {
  GraphLayoutPayload,
  ProteinSummary,
  ProteinTrainRequest,
  ProteinTrainResult,
} from "./proteinSession";
import type { GnnMetrics } from "./gnn";
import type {
  TrainMetrics,
  TrainRequest,
  TrainResult,
} from "./linearModel";
import type {
  MatmulJob,
  MatmulResult,
  TensorOpJob,
  TensorOpResult,
} from "./types";

interface WorkerResponse<T> {
  id: number;
  ok: boolean;
  result?: T;
  error?: string;
}

export function createWebGPUWorker(): Worker {
  const worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });

  // The worker owns its own GPUDevice, and therefore its own allocation ledger:
  // every WGSL kernel in the app allocates in *that* realm, invisible from here.
  // This port is how the system panel learns what the GPU is actually holding.
  // It stays silent until the panel asks — see webgpu/allocations.ts.
  if (typeof MessageChannel !== "undefined") {
    const channel = new MessageChannel();
    worker.postMessage({ type: "telemetryPort" }, [channel.port2]);
    registerAllocationPort(channel.port1);
  }

  return worker;
}

let nextRequestId = 0;

function call<T>(
  worker: Worker,
  message: Record<string, unknown>,
  transfer: Transferable[] = [],
): Promise<T> {
  const id = ++nextRequestId;
  return new Promise<T>((resolve, reject) => {
    const handler = (event: MessageEvent<WorkerResponse<T>>) => {
      if (event.data?.id !== id) return;
      worker.removeEventListener("message", handler);
      if (event.data.ok) resolve(event.data.result as T);
      else reject(new Error(event.data.error ?? "Worker error"));
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ ...message, id }, transfer);
  });
}


export function runMatmulInWorker(
  worker: Worker,
  job: MatmulJob,
): Promise<MatmulResult> {
  // Transfer the input buffers to avoid a copy; `job`'s arrays are consumed.
  return call<MatmulResult>(worker, { type: "matmul", job }, [
    job.a.buffer,
    job.b.buffer,
  ]);
}

export function runTensorOpInWorker(
  worker: Worker,
  job: TensorOpJob,
): Promise<TensorOpResult> {
  // Transfer the operand buffers to avoid a copy; `job`'s arrays are consumed.
  const transfer: Transferable[] = [job.a.buffer];
  if (job.b) transfer.push(job.b.buffer);
  return call<TensorOpResult>(worker, { type: "tensorOp", job }, transfer);
}

export interface TrainingHandle {
  /** Resolves when training finishes (or is cancelled); rejects on error. */
  promise: Promise<TrainResult>;
  /** Ask the worker to stop after the current step. */
  cancel: () => void;
}

/** A snapshot of the model parameters, streamed at each epoch boundary. */
export interface WeightSnapshot {
  epoch: number;
  weights: Float32Array;
  bias: Float32Array;
}

/**
 * Start a streaming linear-model training run in the worker. `onProgress` fires
 * for every emitted metric and `onWeights` for each epoch-boundary parameter
 * snapshot; the returned promise resolves with the final weights and test
 * accuracy. The dataset buffers are transferred (consumed).
 */
export function trainLinearInWorker(
  worker: Worker,
  req: TrainRequest,
  onProgress: (metrics: TrainMetrics) => void,
  onWeights?: (snapshot: WeightSnapshot) => void,
): TrainingHandle {
  const id = ++nextRequestId;
  const promise = new Promise<TrainResult>((resolve, reject) => {
    const handler = (
      event: MessageEvent<{
        id: number;
        event?: "progress" | "weights";
        metrics?: TrainMetrics;
        epoch?: number;
        weights?: Float32Array;
        bias?: Float32Array;
        ok?: boolean;
        result?: TrainResult;
        error?: string;
      }>,
    ) => {
      const data = event.data;
      if (data?.id !== id) return;
      if (data.event === "progress") {
        if (data.metrics) onProgress(data.metrics);
        return;
      }
      if (data.event === "weights") {
        if (data.weights && data.bias) {
          onWeights?.({
            epoch: data.epoch ?? 0,
            weights: data.weights,
            bias: data.bias,
          });
        }
        return;
      }
      worker.removeEventListener("message", handler);
      if (data.ok) resolve(data.result as TrainResult);
      else reject(new Error(data.error ?? "Training failed"));
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ type: "train", id, req }, [
      req.data.xTrain.buffer,
      req.data.yTrain.buffer,
      req.data.xTest.buffer,
      req.data.yTest.buffer,
    ]);
  });

  const cancel = () => worker.postMessage({ type: "trainCancel", id });
  return { promise, cancel };
}


/** Load the bundled graph, lay it out, and report what a run would compute on. */
export function loadGraphInWorker(worker: Worker): Promise<GraphSummary> {
  return call<GraphSummary>(worker, { type: "graphLoad" });
}

export interface GraphTrainingHandle {
  /** Resolves when training finishes (or is cancelled); rejects on error. */
  promise: Promise<GraphTrainResult>;
  /** Ask the worker to stop after the current epoch. */
  cancel: () => void;
}

/**
 * Start a streaming GNN training run. `onEpoch` fires once per epoch with that
 * epoch's metrics and the class each node is currently predicted to be — which
 * is what makes the colours settle on screen while the model trains.
 *
 * Nothing is transferred on the way in: the worker already holds the dataset,
 * and the request is a handful of hyperparameters.
 */
export function trainGraphInWorker(
  worker: Worker,
  req: GraphTrainRequest,
  onEpoch: (metrics: GnnMetrics, predictions: Uint8Array) => void,
): GraphTrainingHandle {
  const id = ++nextRequestId;
  const promise = new Promise<GraphTrainResult>((resolve, reject) => {
    const handler = (
      event: MessageEvent<{
        id: number;
        event?: "progress";
        metrics?: GnnMetrics;
        predictions?: Uint8Array;
        ok?: boolean;
        result?: GraphTrainResult;
        error?: string;
      }>,
    ) => {
      const data = event.data;
      if (data?.id !== id) return;
      if (data.event === "progress") {
        if (data.metrics && data.predictions) {
          onEpoch(data.metrics, data.predictions);
        }
        return;
      }
      worker.removeEventListener("message", handler);
      if (data.ok) resolve(data.result as GraphTrainResult);
      else reject(new Error(data.error ?? "Training failed"));
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ type: "graphTrain", id, req });
  });

  const cancel = () => worker.postMessage({ type: "graphCancel", id });
  return { promise, cancel };
}

export function loadLinkGraphInWorker(
  worker: Worker,
  options: LinkLoadOptions = {},
): Promise<LinkSummary> {
  return call<LinkSummary>(worker, { type: "linkLoad", options });
}

export interface LinkTrainingHandle {
  /** Resolves when training finishes (or is cancelled); rejects on error. */
  promise: Promise<LinkTrainResult>;
  /** Ask the worker to stop after the current epoch. */
  cancel: () => void;
}

/**
 * Start a streaming link-prediction run. `onEpoch` fires with that epoch's
 * metrics only — unlike `/graph`, there is no per-epoch picture to send. The
 * candidates and the embeddings are computed once, after the last epoch, because
 * scoring 3.7 M pairs per epoch would be work done against a drawing nobody is
 * reading yet.
 */
export function trainLinkInWorker(
  worker: Worker,
  req: LinkTrainRequest,
  onEpoch: (metrics: LinkMetrics) => void,
): LinkTrainingHandle {
  const id = ++nextRequestId;
  const promise = new Promise<LinkTrainResult>((resolve, reject) => {
    const handler = (
      event: MessageEvent<{
        id: number;
        event?: "progress";
        metrics?: LinkMetrics;
        ok?: boolean;
        result?: LinkTrainResult;
        error?: string;
      }>,
    ) => {
      const data = event.data;
      if (data?.id !== id) return;
      if (data.event === "progress") {
        if (data.metrics) onEpoch(data.metrics);
        return;
      }
      worker.removeEventListener("message", handler);
      if (data.ok) resolve(data.result as LinkTrainResult);
      else reject(new Error(data.error ?? "Training failed"));
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ type: "linkTrain", id, req });
  });

  const cancel = () => worker.postMessage({ type: "linkCancel", id });
  return { promise, cancel };
}

export function loadProteinsInWorker(worker: Worker): Promise<ProteinSummary> {
  return call<ProteinSummary>(worker, { type: "proteinLoad" });
}

/** One graph's drawable form. The gallery asks for the tiles it is showing. */
export function layoutProteinInWorker(
  worker: Worker,
  graph: number,
): Promise<GraphLayoutPayload> {
  return call<GraphLayoutPayload>(worker, { type: "proteinLayout", graph });
}

export interface ProteinTrainingHandle {
  /** Resolves when training finishes (or is cancelled); rejects on error. */
  promise: Promise<ProteinTrainResult>;
  /** Ask the worker to stop after the current epoch. */
  cancel: () => void;
}

/**
 * Start a streaming graph-classification run. `onEpoch` fires with that epoch's
 * metrics and the class each of the 1113 graphs is currently predicted to be —
 * which is what repaints the gallery while the model trains.
 */
export function trainProteinsInWorker(
  worker: Worker,
  req: ProteinTrainRequest,
  onEpoch: (metrics: GraphClassMetrics, predicted: Uint8Array) => void,
): ProteinTrainingHandle {
  const id = ++nextRequestId;
  const promise = new Promise<ProteinTrainResult>((resolve, reject) => {
    const handler = (
      event: MessageEvent<{
        id: number;
        event?: "progress";
        metrics?: GraphClassMetrics;
        predicted?: Uint8Array;
        ok?: boolean;
        result?: ProteinTrainResult;
        error?: string;
      }>,
    ) => {
      const data = event.data;
      if (data?.id !== id) return;
      if (data.event === "progress") {
        if (data.metrics && data.predicted) onEpoch(data.metrics, data.predicted);
        return;
      }
      worker.removeEventListener("message", handler);
      if (data.ok) resolve(data.result as ProteinTrainResult);
      else reject(new Error(data.error ?? "Training failed"));
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ type: "proteinTrain", id, req });
  });

  const cancel = () => worker.postMessage({ type: "proteinCancel", id });
  return { promise, cancel };
}
