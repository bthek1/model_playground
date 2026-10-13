/**
 * Every note's metadata (#72), newest first. Small on purpose: `seo/pages.ts`
 * reads it to put the notes in `SITE_PAGES`, and that module is in the entry
 * chunk. The articles themselves live in `bodies/`, one chunk each.
 *
 * Relative imports only: `vite.config.ts` runs this in Node.
 */
import type { NoteMeta } from "./types";

export const NOTES: readonly NoteMeta[] = [
  {
    slug: "ort-web-missing-required-scale",
    title: "Fix “Missing required scale” in ONNX Runtime Web",
    description:
      "Why quantized decoders fail on ONNX Runtime Web's WASM backend with qdq_actions.cc “Missing required scale”, and the per-graph dtype fix.",
    published: "2026-10-13",
    modified: "2026-10-13",
    related: ["/translation", "/asr", "/summarization", "/text-generation"],
  },
  {
    slug: "transformers-js-ner-qa-offsets",
    title: "Transformers.js NER and QA return no offsets",
    description:
      "Transformers.js token-classification and question-answering leave start and end undefined. How to recover character offsets safely.",
    published: "2026-10-13",
    modified: "2026-10-13",
    related: ["/token-classification", "/question-answering"],
  },
  {
    slug: "webgpu-shader-f16",
    title: "WebGPU shader-f16: check it before loading q4f16",
    description:
      "“Program Gather requires f16 but the device does not support it”: why having a GPU is the wrong check for fp16 and q4f16 models.",
    published: "2026-10-13",
    modified: "2026-10-13",
    related: ["/image-text-to-text", "/video-text-to-text", "/text-generation"],
  },
  {
    slug: "transformers-js-postmessage",
    title: "Send Transformers.js tensors through postMessage",
    description:
      "Why a Transformers.js Tensor “could not be cloned” and a RawImage loses its methods in a Web Worker, and how to serialise both.",
    published: "2026-10-13",
    modified: "2026-10-13",
    related: ["/depth", "/image-classification", "/image-features"],
  },
  {
    slug: "int8-quantization-wrong",
    title: "When int8 quantization is wrong, not just worse",
    description:
      "Measured cases where a q8 ONNX model gives a confident wrong answer in the browser, and how a known-answer test catches it.",
    published: "2026-10-13",
    modified: "2026-10-13",
    related: ["/image-classification", "/super-resolution", "/zero-shot-image-classification", "/text-ranking"],
  },
];

export const NOTES_PATH = "/notes";

export function notePath(slug: string): string {
  return `${NOTES_PATH}/${slug}`;
}

/** The notes that name `path` as a page demonstrating them. */
export function notesFor(path: string): NoteMeta[] {
  return NOTES.filter((n) => n.related.includes(path));
}
