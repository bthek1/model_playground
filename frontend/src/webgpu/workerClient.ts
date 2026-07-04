// Main-thread client for the WebGPU worker. Handles worker creation and
// request/response correlation so callers get plain promises.

import type {
  MatmulJob,
  MatmulResult,
  TensorOpJob,
  TensorOpResult,
  WebGPUCapabilities,
} from "./types";

interface WorkerResponse<T> {
  id: number;
  ok: boolean;
  result?: T;
  error?: string;
}

export function createWebGPUWorker(): Worker {
  return new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
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

export function detectWebGPUInWorker(
  worker: Worker,
): Promise<WebGPUCapabilities> {
  return call<WebGPUCapabilities>(worker, { type: "detect" });
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
