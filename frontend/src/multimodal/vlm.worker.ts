// The image-text-to-text worker: wires the engine to `self` and supplies the
// real model. The only file in `multimodal/` that imports the runtime.
//
// `AutoModelForImageTextToText` + `AutoProcessor`, **not a pipeline** — 4.2.0
// has no `image-text-to-text` task, see `types.ts`.
//
// Four details that are silent when wrong:
//
//   the chat template   `apply_chat_template` is not decoration. Each checkpoint
//                       has its own image placeholder token and its own turn
//                       markers, and a hand-built prompt string produces output
//                       that is subtly degraded rather than obviously broken —
//                       fluent, confident and not an answer to the question.
//                       Same class of bug as Florence-2's `construct_prompts`,
//                       which this repo already paid for once.
//   the image slots     `{ type: "image" }` is a *slot*, filled positionally
//                       from the array passed to the processor. The content
//                       array's image entries and the image list must line up;
//                       a template rendered with no image slot puts the picture
//                       nowhere and the model answers from the text alone.
//                       With N frames that becomes N slots in the same order —
//                       one short and every later frame is attributed to the
//                       wrong moment, with nothing to see but a confident
//                       sentence. `imageSlots` is that count, derived from the
//                       list rather than passed alongside it, so the two cannot
//                       disagree.
//   the prompt echo     `generate` returns the full sequence, prompt included.
//                       Decoding all of it hands the user back their own
//                       question with the answer glued to the end, so the
//                       prompt's tokens are sliced off first.
//   add_generation_prompt  without it the template stops at the end of the
//                       user's turn and the model continues the *question*
//                       rather than starting an answer.

import {
  AutoProcessor,
  AutoModelForImageTextToText,
  TextStreamer,
} from "@huggingface/transformers";

import { fromPayload } from "@/vision/image";
import { createVlmHandler, type VlmOutput } from "./engine";
import type { VlmRequest, VlmResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<VlmRequest>) => void) | null;
  postMessage: (message: VlmResponse, transfer?: Transferable[]) => void;
};

/** The processor's call signature, which the published types do not describe. */
interface VlmProcessor {
  (text: string, images: unknown[]): Promise<Record<string, unknown>>;
  apply_chat_template: (
    messages: unknown[],
    opts: { add_generation_prompt: boolean },
  ) => string;
  batch_decode: (ids: unknown, opts: { skip_special_tokens: boolean }) => string[];
  tokenizer?: unknown;
}

interface VlmModel {
  generate: (args: Record<string, unknown>) => Promise<unknown>;
  dispose: () => Promise<void>;
}

const handle = createVlmHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const load = {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "q4f16" | "q4",
      progress_callback: opts.progress_callback,
    };

    const [processor, net] = await Promise.all([
      AutoProcessor.from_pretrained(model),
      AutoModelForImageTextToText.from_pretrained(model, load),
    ]);
    const proc = processor as unknown as VlmProcessor;
    const gen = net as unknown as VlmModel;

    return {
      generate: async (
        payloads,
        prompt,
        maxNewTokens,
        onPartial,
      ): Promise<VlmOutput> => {
        const images = payloads.map(fromPayload);

        // Everything up to the first generated token: templating, tokenizing,
        // and the vision encoder. On a 500M model this is seconds, and it is
        // the pause the user would otherwise wait through with nothing on
        // screen.
        onPartial({ stage: "encoding" });
        const encodeStart = now();

        const text = proc.apply_chat_template(
          [
            {
              role: "user",
              content: [...imageSlots(images.length), { type: "text", text: prompt }],
            },
          ],
          { add_generation_prompt: true },
        );
        const inputs = await proc(text, images);

        // The prompt's own length, so the answer can be sliced out of the
        // returned sequence below.
        const promptLength = inputLength(inputs);
        const encodeMs = now() - encodeStart;

        let streamed = "";
        let tokens = 0;
        const streamer = new TextStreamer(
          proc.tokenizer as ConstructorParameters<typeof TextStreamer>[0],
          {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (chunk: string) => {
              streamed += chunk;
              tokens += 1;
              onPartial({ stage: "generating", text: streamed });
            },
          },
        );

        const ids = await gen.generate({
          ...inputs,
          max_new_tokens: maxNewTokens,
          do_sample: false,
          streamer,
        });

        // `generate` returns prompt + answer. Decoding the whole thing hands the
        // user back their own question; slice the prompt off first.
        const full = proc.batch_decode(sliceAnswer(ids, promptLength), {
          skip_special_tokens: true,
        });
        const answer = (full[0] ?? streamed).trim();

        return { text: answer, encodeMs, tokens: tokens || countOf(ids, promptLength) };
      },
      dispose: () => gen.dispose(),
    };
  },
);

/**
 * One `{ type: "image" }` slot per picture, in order.
 *
 * Derived from the image list's own length at the call site rather than taken
 * as a separate argument: the count and the array are the same fact, and the
 * only way this goes wrong is if they are allowed to be two.
 */
function imageSlots(count: number): { type: "image" }[] {
  return Array.from({ length: count }, () => ({ type: "image" as const }));
}

/** Token count of the templated prompt, read off whichever input carries it. */
function inputLength(inputs: Record<string, unknown>): number {
  const ids = inputs.input_ids as { dims?: number[] } | undefined;
  return ids?.dims?.[ids.dims.length - 1] ?? 0;
}

/**
 * Drop the prompt's tokens from the generated sequence.
 *
 * `generate` hands back a Tensor whose `slice` takes per-dimension ranges; the
 * last dimension is the sequence. If the shape is not what we expect, return the
 * sequence untouched rather than throwing — a prompt echoed into the answer is a
 * cosmetic failure, and losing the answer entirely is not.
 */
function sliceAnswer(ids: unknown, promptLength: number): unknown {
  const tensor = ids as {
    dims?: number[];
    slice?: (...ranges: (null | [number, number])[]) => unknown;
  };
  if (!tensor?.slice || !tensor.dims || promptLength <= 0) return ids;
  const seq = tensor.dims[tensor.dims.length - 1];
  if (seq <= promptLength) return ids;
  try {
    return tensor.slice(null, [promptLength, seq]);
  } catch {
    return ids;
  }
}

/** Generated-token count, for the runs where the streamer reported none. */
function countOf(ids: unknown, promptLength: number): number {
  const dims = (ids as { dims?: number[] })?.dims;
  if (!dims?.length) return 0;
  return Math.max(0, dims[dims.length - 1] - promptLength);
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

ctx.onmessage = (event) => {
  void handle(event.data);
};
