// Every model catalogue in the app, with the pages that offer it (#62).
//
// `/licences` renders this, and `licences.test.ts` walks it to prove every
// repo any page can download has a row in `MODEL_LICENCES`. A new catalogue
// goes here, or its models are missing from the attribution page and the test
// cannot see them — `conventions.test.ts` fails, naming the file, when an
// exported `*_MODELS` array is not on this list.
//
// Only the `/licences` route imports this module, so the catalogues it pulls
// together stay in that route's lazy chunk.

import { ASR_MODELS } from "@/audio/types";
import { CLASSIFIER_MODELS as AUDIO_CLASSIFIER_MODELS } from "@/audio/classification";
import { ENHANCE_MODELS } from "@/audio/enhance/types";
import { TTS_MODELS } from "@/audio/tts";
import { VAD_MODELS } from "@/audio/vad/types";
import type { CatalogueEntry } from "@/model/catalogue";
import { VIDEO_VLM_MODELS, VLM_MODELS } from "@/multimodal/types";
import {
  EMBED_MODELS,
  FILL_MASK_MODELS,
  NER_MODELS,
  QA_MODELS,
  RANKING_PAIRS,
  SUMMARIZER_MODELS,
  TEXTGEN_MODELS,
  TEXT_CLASSIFIER_MODELS,
  TRANSLATION_MODELS,
  ZERO_SHOT_TEXT_MODELS,
} from "@/text/catalogue";
import { MATTE_MODELS } from "@/vision/backgroundRemoval";
import { IMAGE_CLASSIFIER_MODELS } from "@/vision/classification";
import { DEPTH_MODELS } from "@/vision/depth";
import { DETECTOR_MODELS } from "@/vision/detection";
import { FEATURE_MODELS } from "@/vision/features";
import { ROBOTICS_ENTRIES } from "@/vision/grounding";
import { POSE_MODELS } from "@/vision/pose/types";
import { SAM_MODELS } from "@/vision/sam/types";
import { SEGMENTER_MODELS } from "@/vision/segmentation";
import { SUPER_RES_MODELS } from "@/vision/superRes";
import { ZERO_SHOT_MODELS } from "@/vision/zeroShot";
import { ZERO_SHOT_DETECTOR_MODELS } from "@/vision/zeroShotDetection";

export interface Catalogue {
  /** The exported constant's name — what the conventions test matches on. */
  name: string;
  /** The routes that offer these models. */
  pages: readonly string[];
  entries: readonly Pick<CatalogueEntry, "id" | "label">[];
}

export const MODEL_CATALOGUES: readonly Catalogue[] = [
  { name: "ASR_MODELS", pages: ["/asr"], entries: ASR_MODELS },
  { name: "CLASSIFIER_MODELS", pages: ["/audio-classification"], entries: AUDIO_CLASSIFIER_MODELS },
  { name: "TTS_MODELS", pages: ["/text-to-speech"], entries: TTS_MODELS },
  { name: "ENHANCE_MODELS", pages: ["/audio-to-audio"], entries: ENHANCE_MODELS },
  { name: "VAD_MODELS", pages: ["/vad"], entries: VAD_MODELS },
  { name: "IMAGE_CLASSIFIER_MODELS", pages: ["/image-classification"], entries: IMAGE_CLASSIFIER_MODELS },
  { name: "DEPTH_MODELS", pages: ["/depth", "/image-to-3d"], entries: DEPTH_MODELS },
  { name: "DETECTOR_MODELS", pages: ["/object-detection"], entries: DETECTOR_MODELS },
  { name: "SEGMENTER_MODELS", pages: ["/segmentation"], entries: SEGMENTER_MODELS },
  {
    name: "ZERO_SHOT_MODELS",
    pages: ["/zero-shot-image-classification", "/video-classification"],
    entries: ZERO_SHOT_MODELS,
  },
  {
    name: "ZERO_SHOT_DETECTOR_MODELS",
    pages: ["/zero-shot-object-detection"],
    entries: ZERO_SHOT_DETECTOR_MODELS,
  },
  { name: "FEATURE_MODELS", pages: ["/image-features"], entries: FEATURE_MODELS },
  { name: "SAM_MODELS", pages: ["/mask-generation"], entries: SAM_MODELS },
  { name: "POSE_MODELS", pages: ["/pose"], entries: POSE_MODELS },
  { name: "MATTE_MODELS", pages: ["/background-removal"], entries: MATTE_MODELS },
  { name: "SUPER_RES_MODELS", pages: ["/super-resolution"], entries: SUPER_RES_MODELS },
  { name: "ROBOTICS_ENTRIES", pages: ["/robotics"], entries: ROBOTICS_ENTRIES },
  {
    name: "VLM_MODELS",
    pages: ["/image-text-to-text", "/visual-question-answering"],
    entries: VLM_MODELS,
  },
  { name: "VIDEO_VLM_MODELS", pages: ["/video-text-to-text"], entries: VIDEO_VLM_MODELS },
  { name: "TEXT_CLASSIFIER_MODELS", pages: ["/text-classification"], entries: TEXT_CLASSIFIER_MODELS },
  { name: "NER_MODELS", pages: ["/token-classification"], entries: NER_MODELS },
  {
    name: "ZERO_SHOT_TEXT_MODELS",
    pages: ["/zero-shot-classification", "/summarization"],
    entries: ZERO_SHOT_TEXT_MODELS,
  },
  { name: "QA_MODELS", pages: ["/question-answering"], entries: QA_MODELS },
  { name: "FILL_MASK_MODELS", pages: ["/fill-mask"], entries: FILL_MASK_MODELS },
  { name: "EMBED_MODELS", pages: ["/text-features", "/sentence-similarity"], entries: EMBED_MODELS },
  { name: "TRANSLATION_MODELS", pages: ["/translation"], entries: TRANSLATION_MODELS },
  { name: "SUMMARIZER_MODELS", pages: ["/summarization"], entries: SUMMARIZER_MODELS },
  { name: "TEXTGEN_MODELS", pages: ["/text-generation"], entries: TEXTGEN_MODELS },
  { name: "RANKING_PAIRS", pages: ["/text-ranking"], entries: RANKING_PAIRS },
];
