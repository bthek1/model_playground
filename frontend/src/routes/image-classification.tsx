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
import { ImageIcon, Loader2 } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { Button } from "@/components/ui/button";
import { useImageClassifier } from "@/hooks/useImageClassifier";
import { useImagePick } from "@/hooks/useImagePick";
import type { ClassLabel } from "@/model/types";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
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
  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    result,
    error,
    load,
    retry,
    cancel,
    run,
  } = useImageClassifier(model);
  useCacheRefresh(session, ready);

  // Picking, the object-URL lifecycle and decode errors are the same on every
  // vision route, so they live in `useImagePick` rather than here.
  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const busy = running || preparing !== null;
  // Each error in the slot that produced it (§4): a failed decode belongs to
  // RUN, a failed download to LOAD, a failed inference to OUTPUT.
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

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
        <ModelPicker
          models={IMAGE_CLASSIFIER_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
        />
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          loadProgress={loadProgress}
          loadedInMs={loadedInMs}
          cached={session.isCached}
          error={loadError}
          onLoad={session.onLoad(load)}
          onCancel={session.onCancel(cancel)}
          onRetry={retry}
          disabled={busy}
        />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to classify an image. You can pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked}
              onClick={classifyCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Classifying…
                </>
              ) : (
                <>
                  <ImageIcon className="size-4" /> Classify
                </>
              )}
            </Button>
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
            <div className="space-y-4">
              {margin != null && margin < 0.1 && (
                <p className="text-xs text-amber-600 dark:text-amber-500">
                  The top two are within {margin.toFixed(2)} of each other — this
                  model is not confident, whatever the first row says.
                </p>
              )}
              <ul className="space-y-2">
                {result.map((p) => (
                  <ScoreRow key={p.label} label={p.label} score={p.score} />
                ))}
              </ul>
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

function ScoreRow({ label, score }: ClassLabel) {
  const pct = Math.round(score * 100);
  return (
    <li className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate">{label}</span>
        <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
          {pct}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary"
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
    </li>
  );
}
