import { describe, expect, it } from "vitest";

import {
  composePrompt,
  TERSE_INSTRUCTION,
  TERSE_MAX_NEW_TOKENS,
  VQA_MAX_NEW_TOKENS,
} from "./prompt";

describe("composePrompt", () => {
  it("sends the question as typed when the toggle is off", () => {
    const composed = composePrompt("What colour is the car?", { terse: false });
    expect(composed.prompt).toBe("What colour is the car?");
    expect(composed.maxNewTokens).toBe(VQA_MAX_NEW_TOKENS);
    expect(composed.instructed).toBe(false);
  });

  it("appends the terse instruction and shortens the cap when it is on", () => {
    const composed = composePrompt("What colour is the car?", { terse: true });
    expect(composed.prompt).toBe(`What colour is the car? ${TERSE_INSTRUCTION}`);
    expect(composed.maxNewTokens).toBe(TERSE_MAX_NEW_TOKENS);
    expect(composed.instructed).toBe(true);
  });

  it("trims and collapses whitespace so the displayed prompt is the sent one", () => {
    // The page renders `prompt` verbatim as "what will be sent". If this
    // normalised anything the run did not, the two would drift.
    expect(composePrompt("  How   many  people? ", { terse: false }).prompt).toBe(
      "How many people?",
    );
  });

  describe("a question that already asks for brevity", () => {
    // Doubly instructing a small decoder makes it answer the *instruction*, and
    // the user who typed the instruction themselves could not see why.
    it.each([
      "Answer in one word: what animal is this?",
      "What animal is this? Answer in a single word.",
      "In a word, what is happening here?",
      "Briefly, what is in this picture?",
      "Short answer: how many people are there?",
      "Is there a cat here? Yes or no.",
      "How many people? Just the number.",
      "Describe this in one sentence.",
    ])("is not instructed again: %s", (question) => {
      const composed = composePrompt(question, { terse: true });
      expect(composed.prompt).toBe(question);
      expect(composed.instructed).toBe(false);
      // The cap still shortens — the user asked for a short answer, and the
      // cap is a backstop rather than the instruction.
      expect(composed.maxNewTokens).toBe(TERSE_MAX_NEW_TOKENS);
    });
  });

  it("does not mistake an unrelated 'one' or 'word' for an instruction", () => {
    const composed = composePrompt("Which one is the word on the sign?", {
      terse: true,
    });
    expect(composed.instructed).toBe(true);
    expect(composed.prompt).toContain(TERSE_INSTRUCTION);
  });

  it("composes an empty box to nothing rather than to a bare order", () => {
    // GENERATE is disabled here, but a prompt that is only an instruction would
    // be answered as one if it ever escaped.
    expect(composePrompt("   ", { terse: true }).prompt).toBe("");
    expect(composePrompt("   ", { terse: true }).instructed).toBe(false);
  });
});
