// Dedicated Web Worker for Automatic Speech Recognition. Loads a Transformers.js
// ASR pipeline (on WebGPU or WASM) and runs transcription off the UI thread. The
// message-handling logic lives in `asrEngine.ts` (unit-tested there); this file
// only wires it to `self` and supplies the real pipeline factory.
//
// As in `webgpu/worker.ts`, we avoid `/// <reference lib="webworker" />` (it
// collides with the app's DOM lib) and narrow `self` to what we use.

import { pipeline } from "@huggingface/transformers";

import { createAsrHandler, type AsrPipeline } from "./asrEngine";
import type { AsrRequest, AsrResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<AsrRequest>) => void) | null;
  postMessage: (message: AsrResponse, transfer?: Transferable[]) => void;
};

const handle = createAsrHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const pipe = await pipeline("automatic-speech-recognition", model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8",
      progress_callback: opts.progress_callback,
    });
    return pipe as unknown as AsrPipeline;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
