// The RL worker. A thin wrapper around `session.ts`, which holds all the logic
// and is unit-tested there — the same split as `tabular/fit.worker.ts`.
//
// The environment and the learner live together in here: an RL step is
// environment, then policy, then update, and splitting them across a message
// boundary would cost more than the arithmetic. (No `/// <reference
// lib="webworker" />` — it collides with the app's DOM lib, as the other
// workers note.)

import { createRlSession } from "./session";
import type { RlWorkerRequest, RlWorkerResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<RlWorkerRequest>) => void) | null;
  postMessage: (message: RlWorkerResponse, transfer?: Transferable[]) => void;
};

const handle = createRlSession((message, transfer) => ctx.postMessage(message, transfer ?? []));

ctx.onmessage = (event) => {
  void handle(event.data);
};
