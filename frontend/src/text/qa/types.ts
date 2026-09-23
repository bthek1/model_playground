// Worker protocol and result shape for extractive question answering.
//
// The one NLP task so far that does **not** ride `text/pipeline.worker.ts`, for
// the repo's standing reason: it is not a plain `pipeline()` call. The same
// criterion put `vision/zeroshot/`, `vision/sam/`, `vision/pose/`,
// `audio/enhance/` and `audio/vad/` in their own modules.
//
// What the pipeline cannot do here is give back **where** the answer is.
// `QuestionAnsweringPipeline` returns `{ answer, score }` — `start` and `end`
// are declared optional in its types and never populated (4.2.0, measured) —
// and this page's entire design is the answer marked *in the passage*. So the
// engine drives `AutoTokenizer` + `AutoModelForQuestionAnswering`, keeps the
// token indices the pipeline discards, and maps them back to characters through
// `text/offsets.ts`.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";

/**
 * One answer, as the page renders it.
 *
 * `text` and `decoded` are separate fields on purpose, and the difference is
 * the whole argument for `offsets.ts`: `text` is sliced out of the passage the
 * user supplied, `decoded` is what the tokenizer makes of the same token ids.
 * They routinely differ — WordPiece decoding re-spaces punctuation, so an
 * answer the page shows as "general-purpose compute shaders" decodes as
 * `general - purpose compute shaders`. Render `text`; `decoded` is kept so the
 * difference can be asserted rather than assumed.
 */
export interface QaAnswer {
  /** The answer **sliced from the passage**. Empty when alignment failed. */
  text: string;
  /** The tokenizer's own decode of the same span. Always present. */
  decoded: string;
  /** `p(start) * p(end)`, in [0, 1] — the only confidence signal there is. */
  score: number;
  /**
   * Character range into the passage, or `null` when the token→character
   * alignment could not be made exactly (see `wordPieceOffsets`). Null means
   * "no highlight", never an approximate one.
   */
  start: number | null;
  end: number | null;
  /**
   * The passage ran past the model's context window and the tail was never
   * read. Silent otherwise: the model answers confidently from the half it was
   * given.
   */
  truncated: boolean;
}

/** Load/warm-up progress. Alias of the shared `ModelProgress`. */
export type QaProgress = ModelProgress;

// --- Worker message protocol -------------------------------------------------

/** Main thread → worker. */
export type QaRequest = ModelRequest<
  {
    model: string;
    opts?: LoadOpts;
    dtypes?: Partial<Record<Backend, DtypeSpec>>;
  },
  { question: string; context: string }
>;

/** Worker → main thread. */
export type QaResponse = ModelResponse<QaAnswer>;
