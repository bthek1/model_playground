// Prompt shaping for `/visual-question-answering`.
//
// **VQA is prompting a general VLM, and this file is the whole difference.**
// Neither classical VQA model has a usable export — `dandelin/vilt-b32-
// finetuned-vqa` and `Salesforce/blip-vqa-base` publish no ONNX — so the
// 3129-answer classification approach is simply unavailable in a tab. What is
// left is §3.1's engine with the question shaped differently, which is also what
// VQA has become everywhere else.
//
// Pure, for two reasons that are really the same one. It is unit-testable
// without a worker; and because it returns the exact string that will be sent,
// **the page can show the user what it is about to ask**. A prompt the page
// rewrites behind the user's back is the same class of problem as the
// `hypothesis_template` the zero-shot pipeline applies silently, which this repo
// was caught by once already: the user compares two prompts on screen while the
// model is being shown two different ones.

/** Generation cap for a normal answer: a sentence or two, not an essay. */
export const VQA_MAX_NEW_TOKENS = 128;

/**
 * Generation cap in terse mode. Sixteen rather than one or two: the cap is a
 * backstop, not the mechanism. A model told to answer in one word and then cut
 * off mid-word has been truncated, not instructed, and the demonstration this
 * page exists for — *the instruction is what shortens the answer* — would be
 * indistinguishable from the scissors.
 */
export const TERSE_MAX_NEW_TOKENS = 16;

/** Appended, with a space, to a question that is not already asking for brevity. */
export const TERSE_INSTRUCTION = "Answer in one word.";

/**
 * Questions that already ask for a short answer.
 *
 * Doubly instructing one ("Answer in one word. Answer in one word.") is not
 * merely untidy — a small instruction-tuned decoder given the same order twice
 * has a real habit of answering the *instruction* rather than the question, and
 * the user who typed the instruction themselves would have no way to see why.
 */
const ALREADY_TERSE =
  /\b(?:one|single|1)[-\s]word\b|\bin a word\b|\bone sentence\b|\bbriefly\b|\bshort answer\b|\byes or no\b|\bjust the (?:word|number|name)\b/i;

export interface ComposedPrompt {
  /** Exactly what is sent to the model — and exactly what the page displays. */
  prompt: string;
  /** The generation cap that travels with it. */
  maxNewTokens: number;
  /** True when this file added the terse instruction, rather than the user. */
  instructed: boolean;
}

/**
 * The question as it will actually be asked.
 *
 * Off: the question as typed, at the normal cap. On: the question plus a terse
 * instruction, at a short cap — the same weights, the same picture, a visibly
 * different answer. That comparison is the most legible lesson about prompting a
 * VLM that fits on one screen, and it costs nothing to download.
 */
export function composePrompt(
  question: string,
  { terse }: { terse: boolean },
): ComposedPrompt {
  const asked = question.trim().replace(/\s+/g, " ");
  if (!terse) {
    return { prompt: asked, maxNewTokens: VQA_MAX_NEW_TOKENS, instructed: false };
  }
  // An empty box composes to nothing rather than to a bare instruction: the
  // page disables GENERATE on it, and a prompt that is only an order would be
  // answered as one.
  if (!asked) {
    return { prompt: "", maxNewTokens: TERSE_MAX_NEW_TOKENS, instructed: false };
  }
  if (ALREADY_TERSE.test(asked)) {
    // The user got there first. Shorten the cap, add nothing.
    return { prompt: asked, maxNewTokens: TERSE_MAX_NEW_TOKENS, instructed: false };
  }
  return {
    prompt: `${asked} ${TERSE_INSTRUCTION}`,
    maxNewTokens: TERSE_MAX_NEW_TOKENS,
    instructed: true,
  };
}
