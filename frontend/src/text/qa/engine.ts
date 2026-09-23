// The extractive-QA engine: tokenize the pair, run the reader, choose the span,
// and map it back onto the **passage's own characters**.
//
// Factored out of the worker so it is testable with a fake reader — no
// download, no real Worker, no `@huggingface/transformers` import — which is
// the same seam `vision/zeroshot/engine.ts` uses.
//
// It owes the three behaviours every engine in this repo owes: **one model live
// at a time** (null the reference *first*, then dispose), **warm-up before
// `ready`** (one throwaway inference, announced as `{ status: "warmup" }`, never
// failing the load), and **never block the main thread**.
//
// The arithmetic it owns beyond the model is deliberately small and pure:
// `select.ts` transcribes the pipeline's span choice so the answer is the one
// the pipeline would have given, and `offsets.ts` recovers the character range
// the pipeline throws away. Everything here is the wiring between them.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";

import { truncatedAfter, wordPieceOffsets } from "../offsets";
import { contextRange, selectAnswerSpan } from "./select";
import type { QaAnswer, QaProgress, QaRequest, QaResponse } from "./types";

/** One forward pass's raw output, plus the sequence that produced it. */
export interface QaEncoding {
  /** `[CLS] question [SEP] context [SEP]`, possibly padded. */
  ids: number[];
  /** 0 for padding. */
  mask: number[];
  startLogits: number[];
  endLogits: number[];
}

/**
 * The tokenizer + model pair, as the worker supplies them. Everything that
 * imports the runtime lives behind this interface.
 */
export interface QaReader {
  /** Tokenize `(question, context)` as a pair and run the model on it. */
  ask(question: string, context: string): Promise<QaEncoding>;
  /** One id → its token string, `##` continuation prefix intact. */
  piece(id: number): string;
  /** Decode a run of ids the way the pipeline would, for `QaAnswer.decoded`. */
  decode(ids: readonly number[]): string;
  sepTokenId: number;
  specialIds: readonly number[];
  dispose?: () => Promise<void>;
}

export interface QaReaderOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: QaProgress) => void;
}

export type QaReaderFactory = (
  model: string,
  opts: QaReaderOpts,
) => Promise<QaReader>;

/**
 * The throwaway pair the reader runs once at load, so the first real question
 * does not pay to compile the shaders. Short on purpose — an encoder's cost is
 * quadratic in sequence length and the point is to touch every kernel.
 */
const WARMUP = {
  question: "What is this?",
  context: "This is a warm-up sentence.",
};

/**
 * Turn one forward pass into an answer.
 *
 * Exported and pure so the two things that can silently go wrong here — the
 * span choice and the character alignment — are testable together without a
 * model, which is the only level at which their interaction is visible.
 */
export function answerFrom(
  encoding: QaEncoding,
  context: string,
  reader: Pick<QaReader, "piece" | "decode" | "sepTokenId" | "specialIds">,
): QaAnswer {
  const { ids, mask, startLogits, endLogits } = encoding;
  const choice = selectAnswerSpan({
    startLogits,
    endLogits,
    ids,
    mask,
    specialIds: reader.specialIds,
    sepTokenId: reader.sepTokenId,
  });
  if (!choice) {
    throw new Error(
      "The model produced no answerable span — the passage may be empty.",
    );
  }

  const decoded = reader.decode(
    ids.slice(choice.startToken, choice.endToken + 1),
  );

  // The context's own tokens, aligned against the context's own characters. The
  // range comes from `select.ts` so the two cannot drift apart.
  const range = contextRange(ids, reader.sepTokenId, reader.specialIds);
  const offsets = wordPieceOffsets(
    context,
    ids.slice(range.start, range.end).map((id) => reader.piece(id)),
  );

  // No alignment, no highlight. The answer string still stands, and the page
  // says the passage could not be marked rather than marking the wrong words.
  if (!offsets) {
    return { text: "", decoded, score: choice.score, start: null, end: null, truncated: false };
  }

  const first = offsets[choice.startToken - range.start];
  const last = offsets[choice.endToken - range.start];
  if (!first || !last) {
    return { text: "", decoded, score: choice.score, start: null, end: null, truncated: false };
  }

  return {
    text: context.slice(first.start, last.end),
    decoded,
    score: choice.score,
    start: first.start,
    end: last.end,
    truncated: truncatedAfter(context, offsets),
  };
}

export function createQaHandler(
  post: (message: QaResponse, transfer?: Transferable[]) => void,
  factory: QaReaderFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let reader: QaReader | null = null;

  return async function handle(msg: QaRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = reader;
        reader = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        reader = await factory(msg.model, {
          device: opts.device,
          dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await reader.ask(WARMUP.question, WARMUP.context);
          } catch {
            /* the first real question pays the compile cost instead */
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
      if (!reader) throw new Error("No model loaded");
      if (msg.context.trim().length === 0) {
        throw new Error("There is no passage to answer from.");
      }
      const encoding = await reader.ask(msg.question, msg.context);
      post({
        type: "result",
        id: msg.id,
        result: answerFrom(encoding, msg.context, reader),
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/** Free a reader, tolerating a backend that fails or has no `dispose`. */
async function disposeQuietly(reader: QaReader | null): Promise<void> {
  try {
    await reader?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
