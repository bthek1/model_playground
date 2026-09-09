// Generic Web Worker for discriminative Transformers.js vision pipelines
// (classification, detection, segmentation, depth, zero-shot, features). The
// pipeline `task` comes from the `load` message, so one worker serves them all.
// Message-handling logic lives in `engine.ts` (unit-tested there); this file only
// wires it to `self`, supplies the real pipeline factory, and rebuilds the
// `RawImage` the pipeline expects from the payload that crossed the wire.
//
// As in `audio/pipeline.worker.ts`, we avoid `/// <reference lib="webworker" />`
// (it collides with the app's DOM lib) and narrow `self` to what we use.

import { pipeline } from "@huggingface/transformers";

import { createVisionHandler, type CallableVisionPipeline } from "./engine";
import { fromPayload } from "./image";
import type { VisionRequest, VisionResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<VisionRequest>) => void) | null;
  postMessage: (message: VisionResponse, transfer?: Transferable[]) => void;
};

const handle = createVisionHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (task, model, opts) => {
    const pipe = await pipeline(task as Parameters<typeof pipeline>[0], model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8",
      progress_callback: opts.progress_callback,
    });
    // The payload → RawImage conversion lives here, not in the engine: it is the
    // one step that needs the Transformers.js runtime.
    const call: CallableVisionPipeline = (payload, ...args) =>
      (pipe as unknown as (...a: unknown[]) => Promise<unknown>)(
        fromPayload(payload),
        ...args,
      );
    call.dispose = () => (pipe as unknown as { dispose: () => Promise<void> }).dispose();
    return call;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
