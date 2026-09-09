// Depth Estimation — a per-pixel depth map from a single photo or a live camera
// frame, computed entirely in the browser. The best-looking output in the
// Computer Vision category, and one forward pass.
//
// The page's one substantive obligation, and it is a claim about the model
// rather than about the code: **this is relative depth, not metres.** The values
// are an inverse-depth map on an arbitrary per-image scale, so two frames cannot
// be compared without aligning them first. That sits next to the colour bar in
// words, because a page that shows a depth map without saying so teaches
// something false. Depth Pro is the exception — it is metric — and the legend
// flips to match, which is exactly why the direction is read off the catalogue
// entry rather than hard-coded.
//
// The second thing that matters is that `drawHeatmap` normalises per frame. An
// inverse-depth map on an arbitrary scale painted without rescaling comes out
// uniformly black or uniformly white, and the page then looks broken rather than
// wrong — a much harder bug to notice.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { FlaskConical, Layers3, Loader2, Mountain } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useDepth, depthDims, type DepthResult } from "@/hooks/useDepth";
import { useImagePick } from "@/hooks/useImagePick";
import { sizeEstimate } from "@/model/size";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_DEPTH_MODEL,
  DEPTH_MODELS,
  isHeavy,
  MAX_INFERENCE_SIDE,
} from "@/vision/depth";
import { drawHeatmap, drawPixels, RAMP_CSS, rangeOf } from "@/vision/draw";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/depth")({ component: DepthPage });

/**
 * The opt-in for Depth Pro, the one entry heavy enough that stating the size in
 * the picker is not enough. Same gate as `/text-to-audio` puts in front of
 * MusicGen, and for the same reason: a gigabyte is a decision, not a detail.
 */
function HeavyModelNotice({ sizeLabel }: { sizeLabel: string }) {
  return (
    <Card data-testid="heavy-model-notice">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="size-4" /> Metric depth — and about a gigabyte
        </CardTitle>
        <CardDescription>Read this before starting the download.</CardDescription>
      </CardHeader>
      <CardContent className="text-sm">
        <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
          <li>
            <span className="font-medium text-foreground">{sizeLabel}</span> of
            weights — twenty times the default model, and cached only after it
            has finished. Nothing downloads until you press Load.
          </li>
          <li>
            It needs <span className="font-medium text-foreground">WebGPU</span>.
            On CPU it is not slow, it is unusable.
          </li>
          <li>
            What you get for it: depth in{" "}
            <span className="font-medium text-foreground">real metres</span>,
            plus a focal-length estimate — the only model here that is not
            relative.
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}

function DepthPage() {
  const session = useModelSelection({
    routeKey: "depth",
    models: DEPTH_MODELS,
    fallback:
      DEPTH_MODELS.find((m) => m.id === DEFAULT_DEPTH_MODEL) ?? DEPTH_MODELS[0],
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
  } = useDepth(model, session.autoLoad);
  useCacheRefresh(session, ready);

  // The frame the current result belongs to, kept so OUTPUT can show the source
  // beside its depth map. On a live feed that is the last frame the model saw,
  // not the one the camera is showing now — pairing a map with a *later* frame
  // would misrepresent both.
  const [source, setSource] = useState<RawImage | null>(null);
  const [live, setLive] = useState(false);

  // Cap the source before inference and show *that* frame as the source: the
  // depth map is an answer about the pixels the model actually saw, so pairing
  // it with a higher-resolution original would misdescribe what produced it. A
  // camera frame is already capped, so this is a no-op on the live path.
  const estimate = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setSource(small);
      await run(small);
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        if (ready) await estimate(next.image);
        else setSource(await downscale(next.image, MAX_INFERENCE_SIDE));
      },
    });

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (!ready) return;
      await estimate(frame);
    },
    [ready, estimate],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const size = useMemo(
    () => sizeEstimate(session.model.params, session.model.bytes),
    [session.model],
  );

  const estimateCurrent = () => {
    if (!picked) return;
    clearError();
    void estimate(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Mountain}
      title="Depth Estimation"
      description={
        <>
          Turn one photo into a depth map, on your own GPU. A single forward pass
          in a Web Worker — the picture is never uploaded.
        </>
      }
      labels={{ output: "Depth map" }}
      select={
        <ModelPicker
          models={DEPTH_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
        />
      }
      load={
        <div className="space-y-3">
          {isHeavy(session.model) && !session.isCached && (
            <HeavyModelNotice sizeLabel={size.label} />
          )}
          <ModelStatus
            status={status}
            backend={backend}
            loadProgress={loadProgress}
            loadedInMs={loadedInMs}
            cached={session.isCached}
            restoring={session.restoring}
            error={loadError}
            onLoad={session.onLoad(load)}
            onCancel={session.onCancel(cancel)}
            onRetry={retry}
            disabled={busy}
          />
        </div>
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to estimate depth. You can pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked || live}
              onClick={estimateCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Estimating…
                </>
              ) : (
                <>
                  <Layers3 className="size-4" /> Estimate depth
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — a scene with a clear foreground and background shows the most."
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
          />
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Depth map"
          description={
            session.model.metric
              ? "Depth in metres, near to far."
              : "Relative depth — brighter is nearer. Not metres."
          }
          running={running}
          runningLabel="Estimating depth…"
          error={runError}
          empty="Pick an image and its depth map appears here, beside the picture it came from."
        >
          {result && source && (
            <DepthView
              result={result}
              source={source}
              metric={session.model.metric ?? false}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function DepthView({
  result,
  source,
  metric,
}: {
  result: DepthResult;
  source: RawImage;
  metric: boolean;
}) {
  const tensor = result.predicted_depth;
  const { width, height } = depthDims(tensor);
  const range = useMemo(() => rangeOf(tensor.data ?? []), [tensor]);

  const paintDepth = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawHeatmap(ctx, tensor.data ?? [], width, height);
    },
    [tensor, width, height],
  );
  const paintSource = useCallback(
    (ctx: CanvasRenderingContext2D) => drawPixels(ctx, source),
    [source],
  );

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <figure className="space-y-1">
          <OverlayCanvas
            width={source.width}
            height={source.height}
            draw={paintSource}
            label="Source frame"
            testId="depth-source"
          />
          <figcaption className="text-xs text-muted-foreground">
            Source
          </figcaption>
        </figure>
        <figure className="space-y-1">
          <OverlayCanvas
            width={width}
            height={height}
            draw={paintDepth}
            label="Estimated depth map"
            testId="depth-map"
          />
          <figcaption className="text-xs text-muted-foreground">
            Depth · {width}x{height}
          </figcaption>
        </figure>
      </div>

      <div className="space-y-1">
        <div
          data-testid="depth-legend"
          className="h-3 w-full rounded-sm"
          style={{ background: RAMP_CSS }}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          {/* Depth Anything emits *inverse* depth — a big number is close. Depth
              Pro emits metres, where a big number is far away. Same ramp, so the
              ends have to be labelled from the model, not from the picture. */}
          <span>{metric ? "near" : "far"}</span>
          <span className="font-mono tabular-nums">
            {range.lo.toFixed(2)} … {range.hi.toFixed(2)}
            {metric ? " m" : ""}
          </span>
          <span>{metric ? "far" : "near"}</span>
        </div>
        {!metric && (
          <p className="text-xs text-amber-600 dark:text-amber-500">
            These are <span className="font-medium">relative</span> depths on an
            arbitrary scale, not metres — and the scale is re-fitted to each
            image, so two frames cannot be compared without aligning them first.
          </p>
        )}
      </div>
    </div>
  );
}
