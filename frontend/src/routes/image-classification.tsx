// Image Classification — the first Computer Vision route, and the simplest
// instance of the four-slot pattern in the app: an image in, a ranked label list
// out, no decode step at all. It runs client-side in the generic vision Web
// Worker (WebGPU, WASM fallback) — the picture never leaves the machine.
//
// One deliberate decision in the OUTPUT slot: **the top five, never the argmax.**
// A 0.31 / 0.29 near-tie is the case worth seeing, and a single confident-looking
// label is exactly what hides it. The margin between first and second is printed
// for the same reason.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { ImageIcon } from "lucide-react";

import { RunButton } from "@/components/model/RunButton";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ScoreList } from "@/components/model/ScoreList";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { useImageClassifier } from "@/hooks/useImageClassifier";
import { useImagePick } from "@/hooks/useImagePick";
import { useModelSelection } from "@/model/useModelSelection";
import { useTaskSlots } from "@/model/useTaskSlots";
import {
  DEFAULT_IMAGE_CLASSIFIER,
  IMAGE_CLASSIFIER_MODELS,
} from "@/vision/classification";
import { IMAGE_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/image-classification")({
  component: ImageClassificationPage,
});

function ImageClassificationPage() {
  const session = useModelSelection({
    routeKey: "image-classification",
    models: IMAGE_CLASSIFIER_MODELS,
    fallback:
      IMAGE_CLASSIFIER_MODELS.find((m) => m.id === DEFAULT_IMAGE_CLASSIFIER) ??
      IMAGE_CLASSIFIER_MODELS[0],
  });
  const model = session.model.id;
  const task = useImageClassifier(model);
  const {
    ready,
    running,
    result,
    run,
  } = task;

  // Picking, the object-URL lifecycle and decode errors are the same on every
  // vision route, so they live in `useImagePick` rather than here.
  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const busy = running || preparing !== null;
  // Each error in the slot that produced it (§4): a failed decode belongs to
  // RUN, a failed download to LOAD, a failed inference to OUTPUT.
  const slots = useTaskSlots(session, task, {
    models: IMAGE_CLASSIFIER_MODELS,
    busy,
  });
  const { runError } = slots;

  const classifyCurrent = () => {
    if (!picked) return;
    clearError();
    void run(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  const margin =
    result && result.length > 1 ? result[0].score - result[1].score : null;

  return (
    <ModelPage
      icon={ImageIcon}
      title="Image Classification"
      description={
        <>
          Label a picture entirely in your browser. An ImageNet-1k classifier
          runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — the image is
          never uploaded.
        </>
      }
      labels={{ output: "Predictions" }}
      select={
        <ModelPicker {...slots.picker} />
      }
      load={
        <ModelStatus {...slots.status} />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to classify an image. You can pick a picture first."
          controls={
            <RunButton
              disabled={!ready || busy || !picked}
              onRun={classifyCurrent}
              icon={ImageIcon}
              running={running}
              runningLabel="Classifying…"
            >
              Classify
            </RunButton>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the first two are easy, the rest are the cases where a top-1 label starts to mislead."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          />
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Predictions"
          description="Top five, highest score first — a near-tie at the top is the thing worth seeing."
          meta={
            margin != null ? (
              <span className="tabular-nums">
                top-2 margin {margin.toFixed(2)}
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Classifying…"
          error={runError}
          empty="Pick an image and its five most likely labels appear here, with their scores."
        >
          {result && result.length > 0 && (
            <ScoreList scores={result} format="percent" />
          )}
        </OutputPanel>
      }
    />
  );
}
