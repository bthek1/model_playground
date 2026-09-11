// Image Segmentation — per-pixel class masks composited over the picture, with a
// legend and per-class toggles, computed in the browser.
//
// **Semantic segmentation**: one label per pixel, no instances. The page says so
// rather than leaving it implied, because "segmentation" covers three different
// tasks and only this one has a real browser path — Mask2Former, OneFormer and
// EoMT publish no ONNX export at all. DETR panoptic is offered as the one
// exception and labelled as such.
//
// The output shape drives everything here: the pipeline returns **one
// single-channel mask per class present**, not an indexed label map. They are
// composited into a single canvas by `drawMasks`; rendering 150 separate images
// is the failure mode that helper exists to prevent.
//
// Opacity and the class toggles are **pure derivations** over masks already in
// hand — neither re-runs the model, which is the whole reason the masks are kept
// rather than a finished canvas. Colours come from `colorForLabel`, which hashes
// the label, so a class keeps its colour between runs and between frames; a
// class that changes colour every frame is worse than no colour at all.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, Shapes } from "lucide-react";
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
  coverageOf,
  useSegmenter,
  visibleMasks,
  type SegmentMask,
} from "@/hooks/useSegmenter";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { colorForLabel, drawMasks, drawPixels } from "@/vision/draw";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";
import {
  DEFAULT_MASK_ALPHA,
  DEFAULT_SEGMENTER,
  MAX_INFERENCE_SIDE,
  SEGMENTER_MODELS,
} from "@/vision/segmentation";

export const Route = createFileRoute("/segmentation")({
  component: SegmentationPage,
});

function SegmentationPage() {
  const session = useModelSelection({
    routeKey: "segmentation",
    models: SEGMENTER_MODELS,
    fallback:
      SEGMENTER_MODELS.find((m) => m.id === DEFAULT_SEGMENTER) ??
      SEGMENTER_MODELS[0],
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
  } = useSegmenter(model);
  useCacheRefresh(session, ready);

  const [source, setSource] = useState<RawImage | null>(null);
  const [live, setLive] = useState(false);
  const [alpha, setAlpha] = useState(DEFAULT_MASK_ALPHA);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set());

  const segment = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setSource(small);
      // A new picture has a new label set, so a class hidden on the last one
      // must not silently hide a same-named class here.
      setHidden(new Set());
      await run(small);
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (!ready) return;
      await segment(frame);
    },
    [ready, segment],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const toggle = (label: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });

  const segmentCurrent = () => {
    if (!picked) return;
    clearError();
    void segment(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Shapes}
      title="Image Segmentation"
      description={
        <>
          Label every pixel, not just the picture. Semantic segmentation in a Web
          Worker on your own GPU — the image never leaves the machine.
        </>
      }
      labels={{ output: "Masks" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={SEGMENTER_MODELS}
            value={model}
            onChange={session.setModel}
            disabled={loading || busy}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          {/* A segmenter can only ever say what its training set contained, so
              the label space is stated before the run rather than discovered
              from a confusing result. */}
          <p
            data-testid="class-space"
            className="text-xs text-muted-foreground"
          >
            <span className="font-medium capitalize">{session.model.kind}</span>{" "}
            · {session.model.classes}
          </p>
        </div>
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
          disabledHint="Load a model to segment an image. You can pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked || live}
              onClick={segmentCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Segmenting…
                </>
              ) : (
                <>
                  <Shapes className="size-4" /> Segment
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the street scene has the most classes; the face models want a portrait instead."
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
              <Label htmlFor="alpha">
                Overlay opacity:{" "}
                <span className="tabular-nums">{alpha.toFixed(2)}</span>
              </Label>
              <input
                id="alpha"
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="block w-full max-w-xs"
              />
              <p className="text-xs text-muted-foreground">
                Nothing re-runs — the overlay is re-composited from the masks
                already returned.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Masks"
          description="One mask per class present, composited into a single overlay. Click a class to hide it."
          meta={
            result ? (
              <span className="tabular-nums">
                {result.length - hidden.size} of {result.length} classes
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Segmenting…"
          error={runError}
          empty="Pick an image and its classes appear here, painted over the picture with a legend."
        >
          {result && source && (
            <SegmentationView
              masks={result}
              source={source}
              alpha={alpha}
              hidden={hidden}
              onToggle={toggle}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function SegmentationView({
  masks,
  source,
  alpha,
  hidden,
  onToggle,
}: {
  masks: readonly SegmentMask[];
  source: RawImage;
  alpha: number;
  hidden: ReadonlySet<string>;
  onToggle: (label: string) => void;
}) {
  const shown = useMemo(() => visibleMasks(masks, hidden), [masks, hidden]);
  const coverage = useMemo(() => coverageOf(masks), [masks]);

  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, source);
      drawMasks(ctx, shown, {
        width: source.width,
        height: source.height,
        alpha,
      });
    },
    [shown, source, alpha],
  );

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={source.width}
        height={source.height}
        draw={paint}
        label={`Segmentation overlay: ${shown.length} classes shown`}
        testId="segmentation-canvas"
      />

      <ul className="flex flex-wrap gap-1.5">
        {coverage.map(({ label, coverage: pct }) => {
          const off = hidden.has(label);
          return (
            <li key={label}>
              <button
                type="button"
                aria-pressed={!off}
                onClick={() => onToggle(label)}
                className="flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs data-[off=true]:opacity-40"
                data-off={off}
              >
                <span
                  aria-hidden
                  className="size-3 shrink-0 rounded-sm"
                  style={{ background: colorForLabel(label) }}
                />
                <span className="truncate">{label}</span>
                <span className="font-mono text-muted-foreground tabular-nums">
                  {(pct * 100).toFixed(0)}%
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
