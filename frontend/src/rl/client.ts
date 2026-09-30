// Main-thread client for the RL worker — `trainGraphInWorker`'s shape: a
// `{ promise, cancel }` handle plus a progress callback, so a page drives a
// streaming run the same way `/graph` does. What it adds is `control`, for the
// two knobs that change a run in progress without restarting it.

import type {
  RlControl,
  RlProgress,
  RlTrainRequest,
  RlTrainResult,
  RlWorkerRequest,
  RlWorkerResponse,
} from "./types";

export function createRlWorker(): Worker {
  return new Worker(new URL("./rl.worker.ts", import.meta.url), { type: "module" });
}

export interface RlTrainingHandle {
  /** Resolves when the run finishes or is stopped; rejects on error. */
  promise: Promise<RlTrainResult>;
  /** Ask the worker to stop. It jumps the queue; see session.ts. */
  cancel: () => void;
  /** Change ε or the speed of the run in progress. Not a re-run. */
  control: (control: RlControl) => void;
}

let nextRequestId = 0;

export function trainRlInWorker(
  worker: Worker,
  req: RlTrainRequest,
  onProgress: (progress: RlProgress) => void,
  speed: number | null = null,
): RlTrainingHandle {
  const id = ++nextRequestId;
  const send = (message: RlWorkerRequest) => worker.postMessage(message);

  const promise = new Promise<RlTrainResult>((resolve, reject) => {
    const handler = (event: MessageEvent<RlWorkerResponse>) => {
      const data = event.data;
      if (data?.id !== id) return;
      if ("event" in data) {
        onProgress(data.progress);
        return;
      }
      worker.removeEventListener("message", handler);
      if (data.ok) resolve(data.result);
      else reject(new Error(data.error));
    };
    worker.addEventListener("message", handler);
    send({ type: "train", id, req, speed });
  });

  return {
    promise,
    cancel: () => send({ type: "cancel", id }),
    control: (control) => send({ type: "control", id, control }),
  };
}
