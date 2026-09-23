// Picking the answer span out of a question-answering model's two logit
// vectors — the arithmetic `QuestionAnsweringPipeline` does, reimplemented so
// the **token indices** survive.
//
// That is the only reason this file exists. The pipeline computes exactly this,
// then throws the indices away and returns the decoded string
// (`{ answer, score }`, past a literal `// TODO add start and end?` in its own
// source). A page that marks the answer in the passage needs the indices, so
// the selection comes back here and the pipeline is not used.
//
// **It is deliberately a transcription, not an improvement.** Same masking, same
// two softmaxes, same `p(start=i) * p(end=j)` over every `i ≤ j`, in the same
// order — so the answer and the score this returns are the ones the pipeline
// would have returned, and `qa/engine.test.ts` can pin that as a property rather
// than as a guess. Three specific temptations resisted:
//
//   a maximum answer length   HF's Python pipeline caps spans at 15 tokens;
//                             Transformers.js does not cap at all. Adding a cap
//                             would silently change which answer comes back on
//                             exactly the questions where the model is unsure —
//                             which is the half of this page's subject.
//   skipping the CLS position `start[0]` is left unmasked so it contributes to
//                             the softmax denominator, and only then is its
//                             *score* zeroed. Masking it instead would rescale
//                             every probability on the page.
//   an early exit             the full O(n²) sweep is ~260k multiplies at the
//                             512-token limit, which is nothing beside the
//                             forward pass that produced the logits.
//
// Pure, so it is unit-testable without a model.

/** The chosen span, in **token** indices into the full input sequence. */
export interface SpanChoice {
  /** Inclusive token index. */
  startToken: number;
  /** Inclusive token index. */
  endToken: number;
  /** `p(start) * p(end)` — the pipeline's `score`, in [0, 1]. */
  score: number;
}

export interface SelectInput {
  /** Raw start logits, one per input token. */
  startLogits: readonly number[];
  /** Raw end logits, one per input token. */
  endLogits: readonly number[];
  /** The tokenized `[CLS] question [SEP] context [SEP]` sequence. */
  ids: readonly number[];
  /** 0 for padding. */
  mask: readonly number[];
  /** Every id the tokenizer considers special — `[CLS]`, `[SEP]`, `[PAD]`, … */
  specialIds: readonly number[];
  /** The separator's id, used to find where the context begins. */
  sepTokenId: number;
}

/**
 * The half-open token range of the context inside the full sequence.
 *
 * `[CLS] question [SEP] context [SEP]` — so the context starts one past the
 * *first* separator. Exported because the engine needs the same boundary to
 * line the context's tokens up with its characters, and deriving it twice is
 * how the two drift apart.
 */
export function contextRange(
  ids: readonly number[],
  sepTokenId: number,
  specialIds: readonly number[],
): { start: number; end: number } {
  const first = ids.findIndex((id) => id === sepTokenId);
  if (first === -1) return { start: 0, end: 0 };
  let end = first + 1;
  while (end < ids.length && !specialIds.includes(ids[end])) end += 1;
  return { start: first + 1, end };
}

/**
 * Choose the highest-scoring `(start, end)` pair that lies inside the context.
 *
 * Returns `null` only when there is no context to answer from at all — an empty
 * passage, or a sequence with no separator.
 */
export function selectAnswerSpan({
  startLogits,
  endLogits,
  ids,
  mask,
  specialIds,
  sepTokenId,
}: SelectInput): SpanChoice | null {
  const sepIndex = ids.findIndex((id) => id === sepTokenId);
  if (sepIndex === -1) return null;

  // Mask everything that cannot be part of an answer: the question, the
  // separators, any other special token, and the padding. Index 0 is left alone
  // on purpose — see the header.
  const starts = [...startLogits];
  const ends = [...endLogits];
  for (let i = 1; i < starts.length; i += 1) {
    if (mask[i] === 0 || i <= sepIndex || specialIds.includes(ids[i])) {
      starts[i] = -Infinity;
      ends[i] = -Infinity;
    }
  }

  const startProbs = softmax(starts);
  const endProbs = softmax(ends);
  // The CLS position is scored zero rather than masked: some checkpoints use it
  // to signal "no answer", and this family has no such head, so it can never be
  // the answer — but it has already done its work in the denominator above.
  startProbs[0] = 0;
  endProbs[0] = 0;

  let best: SpanChoice | null = null;
  for (let i = sepIndex + 1; i < ids.length; i += 1) {
    if (startProbs[i] === 0) continue;
    for (let j = i; j < ids.length; j += 1) {
      if (endProbs[j] === 0) continue;
      const score = startProbs[i] * endProbs[j];
      if (!best || score > best.score) {
        best = { startToken: i, endToken: j, score };
      }
    }
  }
  return best;
}

/** Standard max-subtracted softmax. `-Infinity` entries come out as exactly 0. */
function softmax(values: readonly number[]): number[] {
  let max = -Infinity;
  for (const v of values) if (v > max) max = v;
  if (!Number.isFinite(max)) return values.map(() => 0);

  const exps = values.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  return sum === 0 ? exps : exps.map((e) => e / sum);
}
