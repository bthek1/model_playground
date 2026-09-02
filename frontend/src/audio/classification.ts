// Audio-classification model catalogue. Two flavours, both ONNX-exported and run
// through the generic pipeline worker: fixed-label tagging (AST, wav2vec2-KS) via
// `audio-classification`, and open-set scoring against free-text prompts (CLAP)
// via `zero-shot-audio-classification`. See docs/plans/…/audio-models-in-browser.md.
//
// NOTE: these are the `Xenova/*` repos, not `onnx-community/*`. The latter don't
// exist for AST or wav2vec2-KS — the Hub returns 401 and the load fails with
// "Unauthorized access to file". Verified against the Hub API before changing.

import type { PipelineTask } from "./pipelineTypes";

export interface ClassifierModel {
  id: string;
  label: string;
  hint: string;
  task: PipelineTask;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
}

export const CLASSIFIER_MODELS: ClassifierModel[] = [
  {
    id: "Xenova/ast-finetuned-audioset-10-10-0.4593",
    label: "AST (AudioSet)",
    hint: "527 general sound-event tags — music, speech, animals, machines…",
    params: 87,
    task: "audio-classification",
  },
  {
    id: "Xenova/wav2vec2-base-superb-ks",
    label: "wav2vec2 keyword spotting",
    hint: "Speech-command keyword spotting (yes/no/up/down/…).",
    params: 95,
    task: "audio-classification",
  },
  {
    id: "Xenova/clap-htsat-unfused",
    label: "CLAP (zero-shot)",
    hint: "Score the clip against your own text prompts — no fixed label set.",
    params: 153,
    task: "zero-shot-audio-classification",
  },
];

export const DEFAULT_CLASSIFIER_MODEL = CLASSIFIER_MODELS[0].id;

/** Default free-text prompts for the CLAP zero-shot path. */
export const DEFAULT_ZERO_SHOT_LABELS = [
  "a dog barking",
  "rain falling",
  "a car engine",
  "people speaking",
  "music playing",
];
