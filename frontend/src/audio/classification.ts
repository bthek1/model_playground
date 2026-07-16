// Audio-classification model catalogue. Two flavours, both ONNX-exported and run
// through the generic pipeline worker: fixed-label tagging (AST, wav2vec2-KS) via
// `audio-classification`, and open-set scoring against free-text prompts (CLAP)
// via `zero-shot-audio-classification`. See docs/plans/…/audio-models-in-browser.md.

import type { PipelineTask } from "./pipelineTypes";

export interface ClassifierModel {
  id: string;
  label: string;
  hint: string;
  task: PipelineTask;
}

export const CLASSIFIER_MODELS: ClassifierModel[] = [
  {
    id: "onnx-community/ast-finetuned-audioset-10-10-0.4593",
    label: "AST (AudioSet)",
    hint: "527 general sound-event tags — music, speech, animals, machines…",
    task: "audio-classification",
  },
  {
    id: "onnx-community/wav2vec2-base-superb-ks",
    label: "wav2vec2 keyword spotting",
    hint: "Speech-command keyword spotting (yes/no/up/down/…).",
    task: "audio-classification",
  },
  {
    id: "Xenova/clap-htsat-unfused",
    label: "CLAP (zero-shot)",
    hint: "Score the clip against your own text prompts — no fixed label set.",
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
