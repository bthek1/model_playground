// The streaming worker's message-handling core, factored out of
// `textgen.worker.ts` so it can be unit-tested with a fake model — no download,
// no real Worker, and no `@huggingface/transformers` import. `multimodal/engine.ts`
// is the closest counterpart, and this is deliberately the simpler of the two:
// there is no image to encode, so a run has one phase rather than two.
//
// It owes the same three behaviours as every engine in this app: **one model
// live at a time** with the reference nulled *before* the dispose, **one warm-up
// before `ready`** that never fails the load, and **never blocking the main
// thread**.
//
// The warm-up is capped at a couple of tokens. An autoregressive decoder warmed
// up to completion would put seconds of generation on the critical path to
// `ready` — enough to compile the shaders is the entire point, and that is two
// tokens.

import {
  pickBackendForF16,
  vlmLoadOpts,
  type DtypeSpec,
} from "@/model/backend";

import type {
  Decoding,
  TextGenPartial,
  TextGenProgress,
  TextGenRequest,
  TextGenResponse,
  TextGenResult,
} from "./textgenTypes";

/** A loaded generator, however it is driven underneath. */
export interface Generator {
  generate: (
    prompt: string,
    decoding: Decoding,
    chat: boolean,
    onPartial: (partial: TextGenPartial) => void,
  ) => Promise<TextGenResult>;
  dispose?: () => Promise<void>;
}

export interface GeneratorOpts {
  device: string;
  dtype: DtypeSpec;
  modelFile?: string;
  progress_callback?: (p: TextGenProgress) => void;
}

export type GeneratorFactory = (
  model: string,
  opts: GeneratorOpts,
) => Promise<Generator>;

/** Two tokens — enough to compile the decode loop's kernels, and no more. */
const WARMUP_DECODING: Decoding = {
  doSample: false,
  temperature: 1,
  topP: 1,
  topK: 0,
  repetitionPenalty: 1,
  maxNewTokens: 2,
};

/**
 * The precision a generative decoder loads at.
 *
 * **`vlmLoadOpts()`, referenced rather than copied.** This is the same family of
 * model as a VLM's language half and the arithmetic is identical: fp16 is not a
 * viable download for it (SmolLM2-360M is 691 MB at fp16 against 273 MB at
 * q4f16). A `textgenLoadOpts` would be a second copy of one answer, which is
 * what the note beside `asrLoadOpts` warns about — and there is deliberately no
 * `textLoadOpts` in this category for the same reason. An entry that needs
 * something else pins it per-entry through `dtypes`, applied below.
 */
const generatorLoadOpts = vlmLoadOpts;

export function createTextGenHandler(
  post: (message: TextGenResponse) => void,
  factory: GeneratorFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let generator: Generator | null = null;

  return async function handle(msg: TextGenRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time, and the order matters: null the reference
        // *first*, so a teardown that throws cannot leave a stale model live.
        // These are among the larger downloads in the app.
        const previous = generator;
        generator = null;
        await disposeQuietly(previous);

        // **`pickBackendForF16`, not `pickBackend`.** The default precision
        // here is `q4f16`, and an adapter without `shader-f16` loads those
        // weights, reports ready, and fails on the first operator of every run.
        // This is the first page in the app where the picker's probe and the
        // worker's own choice could disagree — every earlier f16 entry was
        // WebGPU-only, so the row was disabled and the worker never asked.
        const backend = await pickBackendForF16();
        const opts = generatorLoadOpts(backend);
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        generator = await factory(msg.model, {
          device: opts.device,
          dtype,
          modelFile: msg.modelFile,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await generator.generate(
              "Hello",
              WARMUP_DECODING,
              false,
              () => {},
            );
          } catch {
            /* ignore — the first real run pays the compile cost instead */
          }
        }
        post({ type: "ready", model: msg.model, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    // msg.type === "run"
    try {
      if (!generator) throw new Error("No model loaded");
      const result = await generator.generate(
        msg.prompt,
        msg.decoding,
        msg.chat,
        // Correlated to the request id, so a partial from a superseded run can
        // be dropped on the main thread rather than repainting a finished one.
        (partial) => post({ type: "partial", id: msg.id, partial }),
      );
      post({ type: "result", id: msg.id, result });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(generator: Generator | null): Promise<void> {
  try {
    await generator?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
