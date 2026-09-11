// Zero-Shot Object Detection — type a phrase, get a box. /object-detection
// without the fixed 80-class vocabulary, /zero-shot-image-classification with
// localisation, and the page where the two halves of the category meet.
//
// Three decisions, each guarding a specific failure:
//
//  1. **The queries are sent verbatim — no template.** The sibling zero-shot
//     page wraps each bare noun in `"a photo of a {}"` because that is how CLIP
//     was trained. This pipeline tokenizes `candidate_labels` as given, so the
//     text on screen *is* the text the model sees, and the defaults carry their
//     articles. Quietly templating here would mean the page shows one prompt and
//     scores another.
//  2. **The threshold starts at 0.1, not 0.4.** An open-vocabulary detector
//     spreads its probability over an unbounded label space, so OWLv2's
//     confident hits land where D-FINE's uncertain ones do. A closed-detector
//     default shows an empty canvas on a picture full of correctly-found
//     objects — which reads as a broken page, not a bad default.
//  3. **The slider re-filters; only editing a query re-runs.** The model is
//     asked once at `MODEL_THRESHOLD` and the visible set is derived from what
//     came back — the same pure derivation `/object-detection` and `/vad` use.
//     Here it matters more: a re-run costs a 300 MB model a second of work.
//
// The boxes are also scaled back to the source (`scaleDetections`), for the same
// reason and with the same failure mode as `/object-detection`: without it every
// box is a constant fraction too small and too far top-left, which looks like a
// mediocre model rather than a bug in our arithmetic.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, ScanText } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { PhraseList } from "@/components/vision/PhraseList";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useImagePick } from "@/hooks/useImagePick";
import { aboveThreshold } from "@/hooks/useObjectDetector";
import {
  groupByQuery,
  useZeroShotDetector,
} from "@/hooks/useZeroShotDetector";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  colorForLabel,
  drawBoxes,
  drawPixels,
  scaleDetections,
  type Detection,
} from "@/vision/draw";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";
import {
  DEFAULT_QUERIES,
  DEFAULT_THRESHOLD,
  DEFAULT_ZERO_SHOT_DETECTOR,
  MAX_INFERENCE_SIDE,
  THRESHOLD_RANGE,
  ZERO_SHOT_DETECTOR_MODELS,
} from "@/vision/zeroShotDetection";

export const Route = createFileRoute("/zero-shot-object-detection")({
  component: ZeroShotObjectDetectionPage,
});

function ZeroShotObjectDetectionPage() {
  const session = useModelSelection({
    routeKey: "zero-shot-object-detection",
    models: ZERO_SHOT_DETECTOR_MODELS,
    fallback:
      ZERO_SHOT_DETECTOR_MODELS.find(
        (m) => m.id === DEFAULT_ZERO_SHOT_DETECTOR,
      ) ?? ZERO_SHOT_DETECTOR_MODELS[0],
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
  } = useZeroShotDetector(model);
  useCacheRefresh(session, ready);

  const [queries, setQueries] = useState<string[]>(DEFAULT_QUERIES);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [live, setLive] = useState(false);
  // The frame the boxes belong to, and the factor that maps their coordinates
  // back onto it. Captured together, because they are only meaningful as a pair.
  const [frame, setFrame] = useState<{ image: RawImage; scale: number } | null>(
    null,
  );
  // The queries the boxes on screen were actually produced by. Editing the list
  // must not silently relabel a result that predates the edit.
  const [asked, setAsked] = useState<string[]>(DEFAULT_QUERIES);

  const detect = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setFrame({ image, scale: image.width / (small.width || image.width) });
      setAsked(queries);
      await run(small, queries, { consume: small !== image });
    },
    [run, queries],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (f: RawImage) => {
      if (!ready || queries.length === 0) return;
      await detect(f);
    },
    [ready, queries, detect],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // Pure, and on the main thread on purpose: dragging the slider re-derives the
  // visible boxes from detections already in hand. Nothing is re-run.
  const visible = useMemo(
    () => scaleDetections(aboveThreshold(result, threshold), frame?.scale ?? 1),
    [result, threshold, frame],
  );
  const grouped = useMemo(
    () => groupByQuery(visible, asked),
    [visible, asked],
  );

  const detectCurrent = () => {
    if (!picked) return;
    clearError();
    void detect(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={ScanText}
      title="Zero-Shot Object Detection"
      description={
        <>
          Describe what you are looking for and the model boxes it — no class
          list, no fine-tuning. Runs on your own GPU in a Web Worker.
        </>
      }
      labels={{ output: "Detections" }}
      aside={
        live && camera.fps != null ? (
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {camera.fps} fps on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <ModelPicker
          models={ZERO_SHOT_DETECTOR_MODELS}
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
          disabledHint="Load a model to detect your queries. You can write them and pick a picture first."
          controls={
            <Button
              disabled={
                !ready || busy || !picked || live || queries.length === 0
              }
              onClick={detectCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Detecting…
                </>
              ) : (
                <>
                  <ScanText className="size-4" /> Detect
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the street and football shots have things a COCO detector has no class for."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
            camera={{
              live,
              onToggle: setLive,
              videoRef: camera.videoRef,
              error: camera.error,
              fps: camera.fps,
            }}
          >
            <div className="space-y-3">
              <PhraseList
                id="new-query"
                title="Queries"
                items={queries}
                onChange={setQueries}
                placeholder="a red umbrella"
                disabled={live}
                emptyHint="Add at least one phrase to look for."
                hint={
                  session.model.queries === "phrase" ? (
                    <>
                      Grounding DINO grounds free text, so a phrase works better
                      than a class name — and it may answer with a fragment of
                      what you typed.
                    </>
                  ) : (
                    <>
                      Sent to the model exactly as written — this route applies
                      no prompt template, so the article is yours to choose.
                    </>
                  )
                }
              />

              <div className="space-y-1.5">
                <Label htmlFor="threshold">
                  Confidence threshold:{" "}
                  <span className="tabular-nums">{threshold.toFixed(2)}</span>
                </Label>
                <input
                  id="threshold"
                  type="range"
                  min={THRESHOLD_RANGE.min}
                  max={THRESHOLD_RANGE.max}
                  step={THRESHOLD_RANGE.step}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="block w-full max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Open-vocabulary scores run far lower than a COCO detector's —
                  0.1 is a confident hit here. Nothing re-runs; editing a query
                  is what asks the model again.
                </p>
              </div>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Detections"
          description="Boxes over the frame the model saw, grouped by the phrase that found them — including the phrases that found nothing."
          meta={
            result ? (
              <span className="tabular-nums">
                {visible.length} of {result.length} above {threshold.toFixed(2)}
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Detecting…"
          error={runError}
          empty="Write a phrase, pick an image, and everything the model matches to it is boxed here."
        >
          {result && frame && (
            <QueryDetectionView
              detections={visible}
              groups={grouped}
              source={frame.image}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function QueryDetectionView({
  detections,
  groups,
  source,
}: {
  detections: readonly Detection[];
  groups: readonly { query: string; detections: Detection[] }[];
  source: RawImage;
}) {
  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, source);
      drawBoxes(ctx, detections);
    },
    [detections, source],
  );

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={source.width}
        height={source.height}
        draw={paint}
        label={`Detections: ${detections.length} match${detections.length === 1 ? "" : "es"}`}
        testId="detection-canvas"
      />

      <ul className="space-y-2 text-sm" data-testid="query-groups">
        {groups.map((group) => (
          <li key={group.query} className="space-y-1">
            <div className="flex items-center gap-2">
              <span
                aria-hidden
                className="size-3 shrink-0 rounded-sm"
                style={{ background: colorForLabel(group.query) }}
              />
              <span className="truncate font-medium">{group.query}</span>
              <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                {group.detections.length}
              </span>
            </div>
            {group.detections.length > 0 ? (
              <ul className="pl-5 font-mono text-xs text-muted-foreground tabular-nums">
                {group.detections.map((d, i) => (
                  <li key={`${group.query}-${i}`}>{d.score.toFixed(3)}</li>
                ))}
              </ul>
            ) : (
              <p className="pl-5 text-xs text-muted-foreground">
                Nothing above the threshold.
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
