// Generic Web Worker for discriminative Transformers.js pipelines (audio
// classification, zero-shot audio classification, …). The pipeline `task` comes
// from the `load` message, so one worker serves many tasks. Message-handling
// logic lives in `pipelineEngine.ts` (unit-tested there); this file only wires it
// to `self` and supplies the real pipeline factory.
//
// As in `webgpu/worker.ts`, we avoid `/// <reference lib="webworker" />` (it
// collides with the app's DOM lib) and narrow `self` to what we use.

import { pipeline } from "@huggingface/transformers";

import {
  createPipelineHandler,
  type CallablePipeline,
} from "./pipelineEngine";
import type { PipelineRequest, PipelineResponse } from "./pipelineTypes";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<PipelineRequest>) => void) | null;
  postMessage: (message: PipelineResponse, transfer?: Transferable[]) => void;
};

const handle = createPipelineHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (task, model, opts) => {
    const pipe = await pipeline(
      task as Parameters<typeof pipeline>[0],
      model,
      {
        device: opts.device as "webgpu" | "wasm",
        dtype: opts.dtype as "fp16" | "q8",
        progress_callback: opts.progress_callback,
      },
    );
    return pipe as unknown as CallablePipeline;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
