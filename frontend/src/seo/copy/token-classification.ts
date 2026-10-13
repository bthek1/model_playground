import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Find the people, places and organisations in a piece of text and redact them, on your own GPU or CPU.",
  about: [
    "Named entity recognition, or NER, finds the names in a text and says what each one is: a person, an organisation, a location, or something else. It is a form of token classification — the model labels every word piece, and neighbouring pieces are joined back into whole names. Paste a paragraph or start from a sample (an internal memo, a news lede, a sentence in French), and each entity is highlighted in your own text with its type written beside it, so the result does not depend on telling colours apart.",
    "Two models are offered. BERT base NER was trained on English newswire (CoNLL-2003) and tags people, organisations, locations and a catch-all miscellaneous type; it is a 216 MB download on WebGPU or 109 MB quantized for the CPU. mBERT NER covers ten languages and tags dates in place of the catch-all, at 355 MB or 178 MB. Either one runs in a Web Worker, on your GPU through WebGPU or on the CPU through WebAssembly.",
    "Entity extraction is also the first step of redaction. The redact button replaces the chosen entity types with their label in brackets — people and locations by default — and the result can be copied. Changing which types are removed reuses the entities already found, so it runs nothing. Being able to redact names from a document you are not allowed to upload is the case this page is built for: the text never leaves the tab. An entity the page cannot place back in your text is reported on screen rather than dropped.",
  ],
  aliases: [
    "named entity recognition",
    "NER",
    "entity extraction",
    "redact names",
    "token classification",
  ],
};
