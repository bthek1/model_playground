// Dedicated Web Worker that owns a GPUDevice and runs compute jobs off the main
// thread, keeping the UI responsive during heavy inference. Instantiate it via
// createWebGPUWorker() in workerClient.ts.
//
// We avoid `/// <reference lib="webworker" />` because it collides with the DOM
// lib used by the rest of the app; instead we narrow `self` to just what we use.

import { attachAllocationPort } from "./allocations";
import { detectWebGPU } from "./capabilities";
import { GraphSession, type GraphTrainRequest } from "./graphSession";
import {
  LinkSession,
  type LinkLoadOptions,
  type LinkTrainRequest,
} from "./linkSession";
import { ProteinSession, type ProteinTrainRequest } from "./proteinSession";
import { LinearTrainer, type MatmulFn, type TrainRequest } from "./linearModel";
import { runMatmul } from "./runtime";
import { runTensorOp } from "./tensorops";
import type { MatmulJob, TensorOpJob } from "./types";

type WorkerRequest =
  | { type: "telemetryPort" }
  | { type: "detect"; id: number }
  | { type: "matmul"; id: number; job: MatmulJob }
  | { type: "tensorOp"; id: number; job: TensorOpJob }
  | { type: "train"; id: number; req: TrainRequest }
  | { type: "trainCancel"; id: number }
  | { type: "graphLoad"; id: number }
  | { type: "graphTrain"; id: number; req: GraphTrainRequest }
  | { type: "graphCancel"; id: number }
  | { type: "linkLoad"; id: number; options: LinkLoadOptions }
  | { type: "linkTrain"; id: number; req: LinkTrainRequest }
  | { type: "linkCancel"; id: number }
  | { type: "proteinLoad"; id: number }
  | { type: "proteinLayout"; id: number; graph: number }
  | { type: "proteinTrain"; id: number; req: ProteinTrainRequest }
  | { type: "proteinCancel"; id: number };

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

// The /graph route's dataset and layout, loaded once and reused for every run in
// this worker. See graphSession.ts for why neither ever crosses postMessage.
const graph = new GraphSession();

// /link-prediction's graph is a *different* graph: the same Cora with 15 % of
// its citations held out. Its own session, for that reason — see linkSession.ts.
const link = new LinkSession();

// /graph-classification's dataset: 1113 protein graphs as one disjoint union.
// The only one of the three that is fetched rather than bundled.
const proteins = new ProteinSession();

ctx.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "telemetryPort") {
    // The page's end of the allocation ledger. Publishes nothing until asked.
    const port = event.ports?.[0];
    if (port) attachAllocationPort(port);
    return;
  }
  if (
    msg.type === "trainCancel" ||
    msg.type === "graphCancel" ||
    msg.type === "linkCancel" ||
    msg.type === "proteinCancel"
  ) {
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
    if (msg.type === "graphLoad") {
      const summary = await graph.load();
      ctx.postMessage({ id: msg.id, ok: true, result: summary }, [
        summary.rowPtr.buffer,
        summary.colIdx.buffer,
        summary.labels.buffer,
        summary.trainMask.buffer,
        summary.x.buffer,
        summary.y.buffer,
      ]);
      return;
    }
    if (msg.type === "graphTrain") {
      const result = await graph.train(
        msg.req,
        (metrics, predictions) => {
          // A copy per epoch: `predictions` belongs to the training loop, which
          // keeps using it, so transferring it would detach the buffer mid-run.
          const snapshot = predictions.slice();
          ctx.postMessage(
            { id: msg.id, event: "progress", metrics, predictions: snapshot },
            [snapshot.buffer],
          );
        },
        () => cancelled.has(msg.id),
      );
      cancelled.delete(msg.id);
      ctx.postMessage({ id: msg.id, ok: true, result }, [
        result.predictions.buffer,
      ]);
      return;
    }
    if (msg.type === "linkLoad") {
      const summary = await link.load(msg.options);
      ctx.postMessage({ id: msg.id, ok: true, result: summary }, [
        summary.rowPtr.buffer,
        summary.colIdx.buffer,
        summary.labels.buffer,
        summary.x.buffer,
        summary.y.buffer,
      ]);
      return;
    }
    if (msg.type === "linkTrain") {
      const result = await link.train(
        msg.req,
        (metrics) => {
          // Metrics only: the candidates and the embeddings are computed once,
          // after the last epoch, so there is nothing per-epoch to transfer.
          ctx.postMessage({ id: msg.id, event: "progress", metrics });
        },
        () => cancelled.has(msg.id),
      );
      cancelled.delete(msg.id);
      ctx.postMessage({ id: msg.id, ok: true, result }, [
        result.candidates.buffer,
        result.candidateScores.buffer,
        result.embedding.buffer,
      ]);
      return;
    }
    if (msg.type === "proteinLoad") {
      const summary = await proteins.load();
      ctx.postMessage({ id: msg.id, ok: true, result: summary }, [
        summary.labels.buffer,
        summary.graphPtr.buffer,
        summary.testIdx.buffer,
      ]);
      return;
    }
    if (msg.type === "proteinLayout") {
      // One graph's drawable form, laid out on demand: the gallery shows a few
      // dozen of 1113, and laying out the rest would be work for pictures
      // nobody has asked for.
      const layout = proteins.layoutFor(msg.graph);
      // Copies, not the session's own arrays: `layoutFor` remembers what it
      // computed, and transferring its buffers would detach them — the second
      // request for the same graph would get a zero-length layout back.
      const snapshot = {
        index: layout.index,
        nNodes: layout.nNodes,
        rowPtr: layout.rowPtr.slice(),
        colIdx: layout.colIdx.slice(),
        x: layout.x.slice(),
        y: layout.y.slice(),
      };
      ctx.postMessage({ id: msg.id, ok: true, result: snapshot }, [
        snapshot.rowPtr.buffer,
        snapshot.colIdx.buffer,
        snapshot.x.buffer,
        snapshot.y.buffer,
      ]);
      return;
    }
    if (msg.type === "proteinTrain") {
      const result = await proteins.train(
        msg.req,
        (metrics, predicted) => {
          // A copy per epoch: `predicted` belongs to the training loop, which
          // keeps using it, so transferring it would detach the buffer mid-run.
          const snapshot = predicted.slice();
          ctx.postMessage(
            { id: msg.id, event: "progress", metrics, predicted: snapshot },
            [snapshot.buffer],
          );
        },
        () => cancelled.has(msg.id),
      );
      cancelled.delete(msg.id);
      ctx.postMessage({ id: msg.id, ok: true, result }, [
        result.predicted.buffer,
      ]);
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
