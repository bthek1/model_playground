// The document-QA worker: wires the engine to `self` and supplies the real
// model. The only file in `multimodal/docvqa/` that imports the runtime.
//
// A plain `pipeline("document-question-answering")` — the rare case in this
// category where §8 applies unchanged. The pipeline owns Donut's prompt format
// (`<s_docvqa><s_question>…</s_question><s_answer>`) and the extraction regex,
// which is exactly why this route does not roll its own engine: reimplementing
// either would be reimplementing the thing that already works.
//
// Two details that are silent when wrong:
//
//   the answer may be null   The pipeline returns `[{ answer: null }]` when its
//                            `<s_answer>` match misses. Coercing that to `""`
//                            here would erase the difference between "found
//                            nothing" and "found an empty span" before the page
//                            can say which.
//   batch size is 1          The pipeline throws on an array of more than one
//                            image, so the payload is always a single page.

import { pipeline, type DocumentQuestionAnsweringPipeline } from "@huggingface/transformers";

import { fromPayload } from "@/vision/image";
import { createDocVqaHandler } from "./engine";
import type { DocVqaRequest, DocVqaResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<DocVqaRequest>) => void) | null;
  postMessage: (message: DocVqaResponse, transfer?: Transferable[]) => void;
};

const handle = createDocVqaHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const pipe = (await pipeline("document-question-answering", model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8" | "fp32",
      progress_callback: opts.progress_callback,
    })) as DocumentQuestionAnsweringPipeline;

    return {
      answer: async (payload, question, maxNewTokens) => {
        const out = await pipe(fromPayload(payload), question, {
          max_new_tokens: maxNewTokens,
        });
        const first = Array.isArray(out) ? out[0] : out;
        return (first as { answer: string | null } | undefined)?.answer ?? null;
      },
      dispose: () => pipe.dispose(),
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
