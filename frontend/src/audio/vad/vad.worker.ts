// Web Worker for voice activity detection. The ONNX session and the frame loop
// run here: a one-minute take is ~1900 sequential model calls, and although each
// is sub-millisecond, the loop as a whole would visibly stall the UI thread.
//
// Message handling lives in `vadEngine.ts` (unit-tested there); this file only
// wires it to `self` and supplies the real session factory. As in the other
// workers we avoid `/// <reference lib="webworker" />` — it collides with the
// app's DOM lib — and narrow `self` to what we use.

import { loadVad } from "./session";
import { VAD_MODELS } from "./types";
import type { VadRequest, VadResponse } from "./types";
import { createVadHandler } from "./vadEngine";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<VadRequest>) => void) | null;
  postMessage: (message: VadResponse, transfer?: Transferable[]) => void;
};

const handle = createVadHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  (model, onProgress) => {
    const entry = VAD_MODELS.find((m) => m.id === model);
    if (!entry) throw new Error(`Unknown VAD model: ${model}`);
    return loadVad(
      model,
      entry.repo,
      entry.file,
      ({ file, loaded, total, progress }) =>
        onProgress({ status: "progress", file, loaded, total, progress }),
    );
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
