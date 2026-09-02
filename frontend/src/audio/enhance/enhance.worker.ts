// Web Worker for speech enhancement. All the DSP (STFT, ERB, deep filtering,
// overlap-add) plus the ONNX session run here — a minute of 48 kHz audio is
// hundreds of megaflops and would stall the UI thread outright.
//
// Message handling lives in `enhanceEngine.ts` (unit-tested there); this file
// only wires it to `self` and supplies the real session factory. As in
// `pipeline.worker.ts` we avoid `/// <reference lib="webworker" />` — it
// collides with the app's DOM lib — and narrow `self` to what we use.

import { createEnhanceHandler } from "./enhanceEngine";
import { loadDeepFilterNet } from "./session";
import type { EnhanceRequest, EnhanceResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<EnhanceRequest>) => void) | null;
  postMessage: (message: EnhanceResponse, transfer?: Transferable[]) => void;
};

const handle = createEnhanceHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  (repo, onProgress, backend) =>
    loadDeepFilterNet(
      repo,
      ({ file, loaded, total, progress }) =>
        onProgress({ status: "progress", file, loaded, total, progress }),
      backend,
    ),
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
