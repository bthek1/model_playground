// Dedicated Web Worker that owns a GPUDevice and runs compute jobs off the main
// thread, keeping the UI responsive during heavy inference. Instantiate it via
// createWebGPUWorker() in workerClient.ts.
//
// We avoid `/// <reference lib="webworker" />` because it collides with the DOM
// lib used by the rest of the app; instead we narrow `self` to just what we use.

import { detectWebGPU } from "./capabilities";
import { LinearTrainer, type MatmulFn, type TrainRequest } from "./linearModel";
import { runMatmul } from "./runtime";
import { runTensorOp } from "./tensorops";
import type { MatmulJob, TensorOpJob } from "./types";

type WorkerRequest =
  | { type: "detect"; id: number }
  | { type: "matmul"; id: number; job: MatmulJob }
  | { type: "tensorOp"; id: number; job: TensorOpJob }
  | { type: "train"; id: number; req: TrainRequest }
  | { type: "trainCancel"; id: number };

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: unknown, transfer?: Transferable[]) => void;
};

// GPU-backed matmul for the trainer: the two heavy matmuls per step run on the
// device this worker owns.
const gpuMatmul: MatmulFn = async (a, b, m, k, n) =>
  (await runMatmul({ a, b, m, k, n })).data;

// Ids whose training the main thread has asked to stop. Because onmessage yields
// at every awaited matmul, a `trainCancel` message is processed between steps.
const cancelled = new Set<number>();

ctx.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "trainCancel") {
    cancelled.add(msg.id);
    return;
  }

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
      return;
    }
    if (msg.type === "train") {
      const { shape, hp, data, seed } = msg.req;
      const trainer = new LinearTrainer(gpuMatmul, shape, seed);
      await trainer.fit(data, hp, {
        onMetrics: (metrics) => {
          ctx.postMessage({ id: msg.id, event: "progress", metrics });
          // At epoch boundaries stream a copy of the current weights/bias so the
          // UI can render the model evolving. The live buffers are still in use,
          // so we snapshot with slice() before transferring the copies.
          if (metrics.testAcc !== null) {
            const weights = trainer.weights.slice();
            const bias = trainer.bias.slice();
            ctx.postMessage(
              { id: msg.id, event: "weights", epoch: metrics.epoch, weights, bias },
              [weights.buffer, bias.buffer],
            );
          }
        },
        shouldStop: () => cancelled.has(msg.id),
      });
      const testAcc = await trainer.accuracy(
        data.xTest,
        data.yTest,
        data.testSize,
      );
      ctx.postMessage(
        {
          id: msg.id,
          ok: true,
          result: { testAcc, weights: trainer.weights, bias: trainer.bias },
        },
        [trainer.weights.buffer, trainer.bias.buffer],
      );
      cancelled.delete(msg.id);
    }
  } catch (error) {
    cancelled.delete(msg.id);
    ctx.postMessage({
      id: msg.id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
