// Dedicated Web Worker that owns a GPUDevice and runs compute jobs off the main
// thread, keeping the UI responsive during heavy inference. Instantiate it via
// createWebGPUWorker() in workerClient.ts.
//
// We avoid `/// <reference lib="webworker" />` because it collides with the DOM
// lib used by the rest of the app; instead we narrow `self` to just what we use.

import { detectWebGPU } from "./capabilities";
import { runMatmul } from "./runtime";
import { runTensorOp } from "./tensorops";
import type { MatmulJob, TensorOpJob } from "./types";

type WorkerRequest =
  | { type: "detect"; id: number }
  | { type: "matmul"; id: number; job: MatmulJob }
  | { type: "tensorOp"; id: number; job: TensorOpJob };

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

ctx.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "detect") {
      ctx.postMessage({ id: msg.id, ok: true, result: await detectWebGPU() });
      return;
    }
    if (msg.type === "matmul") {
      const result = await runMatmul(msg.job);
      ctx.postMessage({ id: msg.id, ok: true, result }, [result.data.buffer]);
      return;
    }
    if (msg.type === "tensorOp") {
      const result = await runTensorOp(msg.job);
      ctx.postMessage({ id: msg.id, ok: true, result }, [result.data.buffer]);
    }
  } catch (error) {
    ctx.postMessage({
      id: msg.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
