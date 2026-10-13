/**
 * Every note's body, keyed by slug — for the **build** (`scripts/seoPages.ts`)
 * and tests only. The app loads one body per article through
 * `import.meta.glob` in `routes/notes_.$slug.tsx`, so each is its own chunk;
 * importing this map from app code would put all of them in one, and turn the
 * route's dynamic imports into no-ops (`INEFFECTIVE_DYNAMIC_IMPORT`).
 *
 * Relative imports only: `vite.config.ts` runs this in Node.
 */
import type { NoteBlock } from "../types";

import { body as int8QuantizationWrong } from "./int8-quantization-wrong";
import { body as ortWebMissingRequiredScale } from "./ort-web-missing-required-scale";
import { body as transformersJsNerQaOffsets } from "./transformers-js-ner-qa-offsets";
import { body as transformersJsPostmessage } from "./transformers-js-postmessage";
import { body as webgpuShaderF16 } from "./webgpu-shader-f16";

export const NOTE_BODIES: Readonly<Record<string, readonly NoteBlock[]>> = {
  "ort-web-missing-required-scale": ortWebMissingRequiredScale,
  "transformers-js-ner-qa-offsets": transformersJsNerQaOffsets,
  "webgpu-shader-f16": webgpuShaderF16,
  "transformers-js-postmessage": transformersJsPostmessage,
  "int8-quantization-wrong": int8QuantizationWrong,
};
