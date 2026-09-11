// Object Detection — boxes over a photo or a live webcam feed, drawn on the
// user's own GPU. The flagship live demo of the category: this is the page where
// the app is visibly doing real-time inference locally.
//
// Three decisions, each guarding a specific failure:
//
//  1. **Absolute pixels, not fractions.** `useObjectDetector` pins
//     `percentage: false`. The default returns 0–1 fractions, `drawBoxes` wants
//     pixels, and the wrong choice piles every box into the top-left corner. A
//     unit test asserts the flag rather than trusting this paragraph.
//  2. **The threshold slider re-filters; it does not re-run.** The model is asked
//     once for everything above a deliberately low floor and the slider derives
//     the visible set from that list — the same pure-derivation trick `/vad` uses.
//     A slider that re-runs the model is a slider nobody drags.
//  3. **Boxes are scaled back to the source.** The frame is downscaled before
//     inference (resolution is the throttle), so the coordinates come back in the
//     smaller frame's pixels while the canvas shows the original. Without
//     `scaleDetections` every box is drawn a constant fraction too small and too
//     far top-left — which reads as a mediocre detector, not as a bug.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, ScanSearch } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useImagePick } from "@/hooks/useImagePick";
import {
  aboveThreshold,
  useObjectDetector,
} from "@/hooks/useObjectDetector";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_DETECTOR,
  DEFAULT_THRESHOLD,
  DETECTOR_MODELS,
  MAX_INFERENCE_SIDE,
} from "@/vision/detection";
import {
  colorForLabel,
  drawBoxes,
  drawPixels,
  scaleDetections,
  type Detection,
} from "@/vision/draw";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/object-detection")({
  component: ObjectDetectionPage,
});

function ObjectDetectionPage() {
  const session = useModelSelection({
    routeKey: "object-detection",
    models: DETECTOR_MODELS,
    fallback:
      DETECTOR_MODELS.find((m) => m.id === DEFAULT_DETECTOR) ??
      DETECTOR_MODELS[0],
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
  } = useObjectDetector(model);
  useCacheRefresh(session, ready);

  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [live, setLive] = useState(false);
  // The frame the boxes belong to, and the factor that maps their coordinates
  // back onto it. Both are captured at run time, together, because they are only
  // meaningful as a pair.
  const [frame, setFrame] = useState<{
    image: RawImage;
    scale: number;
  } | null>(null);

  // One path for both sources. A camera frame arrives already downscaled, so
  // `downscale` is a no-op there and the scale is 1; a 12-megapixel photo the
  // user dropped is capped here instead of being pushed through the model at
  // full size, and the boxes are mapped back onto the original for display.
  const detect = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setFrame({ image, scale: image.width / (small.width || image.width) });
      await run(small, { consume: small !== image });
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (f: RawImage) => {
      if (!ready) return;
      await detect(f);
    },
    [ready, detect],
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

  const detectCurrent = () => {
    if (!picked) return;
    clearError();
    void detect(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={ScanSearch}
      title="Object Detection"
      description={
        <>
          Find and box every object in a picture — or in your webcam, frame by
          frame — on your own GPU. Nothing leaves the machine.
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
          models={DETECTOR_MODELS}
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
          disabledHint="Load a model to detect objects. You can pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked || live}
              onClick={detectCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Detecting…
                </>
              ) : (
                <>
                  <ScanSearch className="size-4" /> Detect
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the football and street shots are the ones with enough in them to argue about."
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
            <div className="space-y-1.5">
              <Label htmlFor="threshold">
                Confidence threshold:{" "}
                <span className="tabular-nums">{threshold.toFixed(2)}</span>
              </Label>
              <input
                id="threshold"
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="block w-full max-w-xs"
              />
              <p className="text-xs text-muted-foreground">
                Nothing re-runs — the boxes are re-filtered from the detections
                the model already returned.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Detections"
          description="Boxes over the frame the model saw, plus the list — a wrong box is only legible next to its label."
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
          empty="Pick an image and every object the model finds is boxed here, with its score."
        >
          {result && frame && (
            <DetectionView detections={visible} source={frame.image} />
          )}
        </OutputPanel>
      }
    />
  );
}

function DetectionView({
  detections,
  source,
}: {
  detections: readonly Detection[];
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
        label={`Detections: ${detections.length} object${detections.length === 1 ? "" : "s"}`}
        testId="detection-canvas"
      />

      {detections.length > 0 ? (
        <ul className="space-y-1 text-sm">
          {detections.map((d, i) => (
            <li
              key={`${d.label}-${i}`}
              className="flex items-center gap-2 tabular-nums"
            >
              <span
                aria-hidden
                className="size-3 shrink-0 rounded-sm"
                style={{ background: colorForLabel(d.label) }}
              />
              <span className="truncate">{d.label}</span>
              <span className="ml-auto font-mono text-xs text-muted-foreground">
                {d.score.toFixed(2)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          Nothing above the threshold. Drag it down — the model returned
          lower-scoring guesses, and they are often the interesting ones.
        </p>
      )}
    </div>
  );
}
