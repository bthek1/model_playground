// Generic Web Worker for discriminative Transformers.js **text** pipelines. The
// pipeline `task` comes from the `load` message, so one worker serves the whole
// modality. Message-handling logic lives in `engine.ts` (unit-tested there);
// this file only wires it to `self` and supplies the real pipeline factory.
//
// As in `vision/vision.worker.ts`, we avoid `/// <reference lib="webworker" />`
// (it collides with the app's DOM lib) and narrow `self` to what we use.

import { pipeline } from "@huggingface/transformers";

import { createTextHandler, type CallableTextPipeline } from "./engine";
import type { TextRequest, TextResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<TextRequest>) => void) | null;
  postMessage: (message: TextResponse, transfer?: Transferable[]) => void;
};

const handle = createTextHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (task, model, opts) => {
    const pipe = await pipeline(task as Parameters<typeof pipeline>[0], model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8",
      progress_callback: opts.progress_callback,
    });
    const call: CallableTextPipeline = (input, ...args) =>
      (pipe as unknown as (...a: unknown[]) => Promise<unknown>)(input, ...args);
    call.dispose = () =>
      (pipe as unknown as { dispose: () => Promise<void> }).dispose();
    return call;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
