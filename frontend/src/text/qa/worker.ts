// The QA worker: wires the engine to `self` and supplies the real reader. This
// is the only file in the module that imports `@huggingface/transformers`,
// which is what keeps `engine.ts`, `select.ts` and `offsets.ts` testable
// without a download.
//
// Three details taken from the runtime rather than assumed:
//
//   the tokenizer call   `tokenizer(question, { text_pair: context })` is how a
//                        BERT-family pair is encoded, and it is what
//                        `QuestionAnsweringPipeline` does. Encoding the two
//                        separately and concatenating produces a sequence with
//                        the wrong segment ids and no separator.
//   truncation           left on, and the resulting silence is reported: the
//                        model's window is 512 tokens and a longer passage is
//                        cut without an error. `answerFrom` derives it from the
//                        alignment and `QaAnswer.truncated` carries it to the
//                        page.
//   `piece()`            decodes **one id at a time with special tokens kept**,
//                        which is the only way to recover the `##` continuation
//                        prefix — `decode` of a run strips it and re-spaces the
//                        punctuation, which is exactly the string that is not in
//                        the passage.

import {
  AutoModelForQuestionAnswering,
  AutoTokenizer,
} from "@huggingface/transformers";

import { createQaHandler, type QaEncoding } from "./engine";
import type { QaRequest, QaResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<QaRequest>) => void) | null;
  postMessage: (message: QaResponse, transfer?: Transferable[]) => void;
};

/** An ORT tensor as it arrives from the model, before we flatten it. */
type AnyTensor = { data: ArrayLike<number>; dims: number[] };

/** First (and only) batch row of a `[1, seq]` logit tensor, as a plain array. */
function row(tensor: AnyTensor): number[] {
  const width = tensor.dims[tensor.dims.length - 1];
  return Array.from({ length: width }, (_, i) => Number(tensor.data[i]));
}

const handle = createQaHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const load = {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8" | "fp32",
      progress_callback: opts.progress_callback,
    };
    const [tokenizer, reader] = await Promise.all([
      AutoTokenizer.from_pretrained(model, {
        progress_callback: opts.progress_callback,
      }),
      AutoModelForQuestionAnswering.from_pretrained(model, load),
    ]);

    const tok = tokenizer as unknown as {
      (text: string, opts: Record<string, unknown>): Record<string, AnyTensor>;
      decode: (ids: number[], opts: Record<string, unknown>) => string;
      sep_token_id: number;
      all_special_ids: number[];
    };

    return {
      sepTokenId: tok.sep_token_id,
      specialIds: tok.all_special_ids,
      piece: (id: number) => tok.decode([id], { skip_special_tokens: false }),
      decode: (ids: readonly number[]) =>
        tok.decode([...ids], { skip_special_tokens: true }),
      ask: async (question: string, context: string): Promise<QaEncoding> => {
        const inputs = tok(question, {
          text_pair: context,
          padding: true,
          truncation: true,
        });
        const out = (await (
          reader as unknown as (
            i: Record<string, AnyTensor>,
          ) => Promise<Record<string, AnyTensor>>
        )(inputs)) as Record<string, AnyTensor>;
        return {
          ids: row(inputs.input_ids),
          mask: row(inputs.attention_mask),
          startLogits: row(out.start_logits),
          endLogits: row(out.end_logits),
        };
      },
      dispose: async () => {
        await (reader as unknown as { dispose: () => Promise<void> }).dispose();
      },
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
