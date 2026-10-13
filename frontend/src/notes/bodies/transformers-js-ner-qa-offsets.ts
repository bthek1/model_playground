import type { NoteBlock } from "../types";

export const body: readonly NoteBlock[] = [
  {
    kind: "p",
    text: "In Transformers.js 4.2.0, the `token-classification` (NER) and `question-answering` pipelines do not return character offsets. The TypeScript types declare `start` and `end` as optional fields, so `result.start` compiles, but at runtime it is always `undefined`. The pipeline source has the work unwritten, with a `// TODO add start and end?` comment, and there is no `return_offsets_mapping` option anywhere in the library.",
  },
  { kind: "h2", text: "What it looks like" },
  {
    kind: "p",
    text: "NER returns `entity_group`, `score` and `word`. Question answering returns `answer` and `score`. Code that highlights spans by offset then treats every span as invalid and draws nothing, so the page shows your text with nothing marked. That looks exactly like a model that found no entities, and nothing throws.",
  },
  {
    kind: "p",
    text: "A unit test with a mocked pipeline will not catch it if the mock returns offsets, because the real pipeline never does. On this site only the end-to-end test that loads the real model did.",
  },
  { kind: "h2", text: "Recovering offsets for NER" },
  {
    kind: "p",
    text: "The `word` field is the tokenizer's decoding of the span, so you can find it in the source text yourself. Three details decide whether that is correct:",
  },
  {
    kind: "ul",
    items: [
      "Search forward from the end of the previous match, not from the start of the text. With `indexOf` from zero, a name that appears twice marks its first occurrence twice.",
      "Allow flexible whitespace on a second attempt. WordPiece decodes `Jones-Smith` as `Jones - Smith`, which does not occur in the source.",
      "Merge adjacent spans with the same label when nothing separates them. Even with `aggregation_strategy: \"simple\"`, \"Priya Raman\" can come back as `P` and `##riya Raman`. Do not merge spans separated by a space, or \"Berlin Munich\" becomes one location.",
    ],
  },
  {
    kind: "code",
    text: `function locate(text: string, words: string[]) {
  let from = 0;
  return words.map((word) => {
    const at = text.indexOf(word, from);
    if (at < 0) return null; // retry with flexible whitespace, or report it
    from = at + word.length;
    return { start: at, end: from };
  });
}`,
  },
  {
    kind: "p",
    text: "When an entity still cannot be placed, say so on screen rather than dropping it.",
  },
  { kind: "h2", text: "Recovering offsets for question answering" },
  {
    kind: "p",
    text: "Searching the context for the answer string is not good enough. Asked what WebGPU supports that WebGL does not, the model answers `general-purpose compute shaders`, but the tokenizer decodes those same tokens as `general - purpose compute shaders`, which is not in the passage, so `indexOf` returns -1. A search also finds the first occurrence, which is the wrong one when a passage names something twice.",
  },
  {
    kind: "p",
    text: "The reliable route is to run the model directly, choose the answer span the same way the pipeline does (the start and end logits, two softmaxes, the best `p(start) × p(end)` with start ≤ end), then map those tokens back to characters by walking the tokenizer's pieces along the passage. If any piece fails to line up, which happens with `[UNK]`, a lowercasing tokenizer or an accent-stripping normaliser, return no offsets and quote the answer instead of guessing. A highlight one word off reads as a styling bug. Check the checkpoint's `tokenizer_config.json` first: a cased model with `do_lower_case: false` lines up cleanly.",
  },
  {
    kind: "p",
    text: "On this site, [named entity recognition](/token-classification) recovers NER offsets by walking forward, and [question answering](/question-answering) aligns its spans by walking the tokenizer's pieces. Both have end-to-end tests that assert a character range, not just the answer text.",
  },
];
