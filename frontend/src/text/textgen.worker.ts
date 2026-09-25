// The Web Worker for **text generation** — the one NLP task that does not ride
// `pipeline.worker.ts`, because it streams. The ASR precedent: the task that
// owns a loop owns its worker.
//
// Message-handling logic lives in `textgenEngine.ts` (unit-tested there); this
// file wires it to `self` and supplies the real `pipeline()`-backed generator.
//
// As in the other workers, `/// <reference lib="webworker" />` is avoided (it
// collides with the app's DOM lib) and `self` is narrowed to what we use.

import { pipeline, TextStreamer } from "@huggingface/transformers";

import { createTextGenHandler, type Generator } from "./textgenEngine";
import type {
  Decoding,
  TextGenPartial,
  TextGenRequest,
  TextGenResponse,
} from "./textgenTypes";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<TextGenRequest>) => void) | null;
  postMessage: (message: TextGenResponse, transfer?: Transferable[]) => void;
};

/** What the text-generation pipeline returns per input. */
interface RawGeneration {
  generated_text?: string | { role: string; content: string }[];
}

/** The pipeline's call signature plus the tokenizer the streamer needs. */
interface GenPipeline {
  (input: unknown, opts: Record<string, unknown>): Promise<RawGeneration[]>;
  tokenizer: ConstructorParameters<typeof TextStreamer>[0] & {
    apply_chat_template?: (
      messages: unknown[],
      opts: Record<string, unknown>,
    ) => string;
  };
  dispose: () => Promise<void>;
}

/**
 * Translate our `Decoding` into the pipeline's generation options.
 *
 * Two deliberate shapes. `do_sample: false` is **greedy**, and the sampling
 * knobs are omitted entirely rather than passed alongside it — Transformers.js
 * warns when a sampling parameter is set on a greedy run, and a warning the user
 * cannot see is worse than an option that is simply absent. `top_k: 0` means "no
 * cap", so it is dropped rather than sent as a zero the generator would read as
 * a cap of nothing.
 */
function generationArgs(d: Decoding, streamer: TextStreamer) {
  const base: Record<string, unknown> = {
    max_new_tokens: d.maxNewTokens,
    do_sample: d.doSample,
    repetition_penalty: d.repetitionPenalty,
    streamer,
  };
  if (!d.doSample) return base;
  return {
    ...base,
    temperature: d.temperature,
    top_p: d.topP,
    ...(d.topK > 0 ? { top_k: d.topK } : {}),
  };
}

const handle = createTextGenHandler(
  (message) => ctx.postMessage(message),
  async (model, opts) => {
    const pipe = (await pipeline("text-generation", model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "q4f16",
      progress_callback: opts.progress_callback,
      // A repo whose only usable quantized build lives under a legacy
      // `decoder_model_merged_*` name is reachable only by naming it:
      // `MODEL_SESSION_CONFIG[DecoderOnly]` is
      // `{ model: options.model_file_name ?? 'model' }`.
      ...(opts.modelFile ? { model_file_name: opts.modelFile } : {}),
    })) as unknown as GenPipeline;

    const generator: Generator = {
      async generate(prompt, decoding, chat, onPartial) {
        let streamed = "";
        let tokens = 0;
        const streamer = new TextStreamer(pipe.tokenizer, {
          // The prompt is skipped here *and* the result is read out of the
          // pipeline's own return value below, so the user never gets their
          // own question back — the mistake `/image-text-to-text` documents.
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (chunk: string) => {
            streamed += chunk;
            tokens += 1;
            onPartial({ text: streamed, tokens } satisfies TextGenPartial);
          },
        });

        // An instruct model is asked a question; a bare LM is given a
        // beginning to continue. Sending the raw prompt to an instruct model
        // produces a fluent non-answer with nothing failing — the
        // `apply_chat_template` lesson from #30, one modality over — so the
        // page states which mode it is in and the catalogue says which the
        // checkpoint wants.
        const input =
          chat && pipe.tokenizer.apply_chat_template
            ? pipe.tokenizer.apply_chat_template(
                [{ role: "user", content: prompt }],
                { tokenize: false, add_generation_prompt: true },
              )
            : prompt;

        const started = performance.now();
        const out = await pipe(input, generationArgs(decoding, streamer));
        const ms = performance.now() - started;

        // `generated_text` is prompt + continuation for a plain string input,
        // so the prompt is sliced off. The streamed text is the fallback rather
        // than the primary: a streamer sees only what the callback fired for,
        // and a run that produced no callback at all should still return the
        // model's answer.
        const full = out?.[0]?.generated_text;
        const text =
          typeof full === "string"
            ? full.startsWith(input)
              ? full.slice(input.length)
              : full
            : streamed;

        return { text: text.trim(), tokens: tokens || 0, ms };
      },
      dispose: () => pipe.dispose(),
    };
    return generator;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
