/**
 * Every task page's copy, keyed by route (#71) — for the **build** only
 * (`scripts/seoPages.ts`) and tests. A route imports its own file directly, so
 * its chunk carries one page's words; importing this map from app code would
 * put all of them in whichever chunk did it, and `conventions.test.ts` fails
 * by name if anything outside `scripts/` and tests does.
 *
 * Relative imports only: `vite.config.ts` runs this in Node.
 */
import type { PageCopy } from "./types";

import { copy as playground } from "./playground";
import { copy as training } from "./training";
import { copy as tensor } from "./tensor";
import { copy as discreteMaths } from "./discrete-maths";
import { copy as asr } from "./asr";
import { copy as audioClassification } from "./audio-classification";
import { copy as textToSpeech } from "./text-to-speech";
import { copy as audioToAudio } from "./audio-to-audio";
import { copy as vad } from "./vad";
import { copy as imageClassification } from "./image-classification";
import { copy as depth } from "./depth";
import { copy as objectDetection } from "./object-detection";
import { copy as segmentation } from "./segmentation";
import { copy as zeroShotImageClassification } from "./zero-shot-image-classification";
import { copy as zeroShotObjectDetection } from "./zero-shot-object-detection";
import { copy as imageFeatures } from "./image-features";
import { copy as maskGeneration } from "./mask-generation";
import { copy as pose } from "./pose";
import { copy as videoClassification } from "./video-classification";
import { copy as backgroundRemoval } from "./background-removal";
import { copy as superResolution } from "./super-resolution";
import { copy as imageTo3d } from "./image-to-3d";
import { copy as tabularClassification } from "./tabular-classification";
import { copy as tabularRegression } from "./tabular-regression";
import { copy as timeSeriesForecasting } from "./time-series-forecasting";
import { copy as graph } from "./graph";
import { copy as linkPrediction } from "./link-prediction";
import { copy as graphClassification } from "./graph-classification";
import { copy as imageTextToText } from "./image-text-to-text";
import { copy as visualQuestionAnswering } from "./visual-question-answering";
import { copy as videoTextToText } from "./video-text-to-text";
import { copy as textClassification } from "./text-classification";
import { copy as tokenClassification } from "./token-classification";
import { copy as questionAnswering } from "./question-answering";
import { copy as zeroShotClassification } from "./zero-shot-classification";
import { copy as fillMask } from "./fill-mask";
import { copy as translation } from "./translation";
import { copy as summarization } from "./summarization";
import { copy as textGeneration } from "./text-generation";
import { copy as textFeatures } from "./text-features";
import { copy as sentenceSimilarity } from "./sentence-similarity";
import { copy as textRanking } from "./text-ranking";
import { copy as rl } from "./rl";
import { copy as robotics } from "./robotics";

export const PAGE_COPY: Readonly<Record<string, PageCopy>> = {
  "/playground": playground,
  "/training": training,
  "/tensor": tensor,
  "/discrete-maths": discreteMaths,
  "/asr": asr,
  "/audio-classification": audioClassification,
  "/text-to-speech": textToSpeech,
  "/audio-to-audio": audioToAudio,
  "/vad": vad,
  "/image-classification": imageClassification,
  "/depth": depth,
  "/object-detection": objectDetection,
  "/segmentation": segmentation,
  "/zero-shot-image-classification": zeroShotImageClassification,
  "/zero-shot-object-detection": zeroShotObjectDetection,
  "/image-features": imageFeatures,
  "/mask-generation": maskGeneration,
  "/pose": pose,
  "/video-classification": videoClassification,
  "/background-removal": backgroundRemoval,
  "/super-resolution": superResolution,
  "/image-to-3d": imageTo3d,
  "/tabular-classification": tabularClassification,
  "/tabular-regression": tabularRegression,
  "/time-series-forecasting": timeSeriesForecasting,
  "/graph": graph,
  "/link-prediction": linkPrediction,
  "/graph-classification": graphClassification,
  "/image-text-to-text": imageTextToText,
  "/visual-question-answering": visualQuestionAnswering,
  "/video-text-to-text": videoTextToText,
  "/text-classification": textClassification,
  "/token-classification": tokenClassification,
  "/question-answering": questionAnswering,
  "/zero-shot-classification": zeroShotClassification,
  "/fill-mask": fillMask,
  "/translation": translation,
  "/summarization": summarization,
  "/text-generation": textGeneration,
  "/text-features": textFeatures,
  "/sentence-similarity": sentenceSimilarity,
  "/text-ranking": textRanking,
  "/rl": rl,
  "/robotics": robotics,
};
