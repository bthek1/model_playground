// Main-thread client for the WebGPU worker. Handles worker creation and
// request/response correlation so callers get plain promises.

import { registerAllocationPort } from "./allocations";
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
