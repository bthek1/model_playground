// Image to 3D — one photo, one depth model, an interactive point cloud.
//
// The browser-runnable half of the **Image to 3D** slug. Full reconstruction
// (Zero123++ and friends) is Stable-Diffusion-derived and stays on a server;
// what runs in a tab is depth plus arithmetic, which per vision.md §3.12 is the
// cheapest impressive 3-D demo available — it adds **no new model at all**, only
// `/depth`'s checkpoint and an unprojection.
//
// This is also the one route where both runtimes appear on the same page, and
// they stay apart: inference is Transformers.js in `src/vision/`, rendering is
// hand-written WGSL in `src/webgpu/`, and neither imports the other. They meet
// here, in a route, as a `Float32Array`.
//
// The page's standing obligation is a claim about the *geometry*, not the code:
// **the focal length is an assumption.** Depth Anything V2 predicts relative
// depth with no camera intrinsics, so `f` is a slider with a plausible default
// and the cloud is plausible rather than metric. Two clouds from two photos
// cannot be compared. That sits next to the control, in words.
//
// Both sliders re-derive from the cached depth map — neither re-runs the model.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Boxes, Loader2, RotateCcw } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
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
import { useDepth, depthDims, type DepthResult } from "@/hooks/useDepth";
import { useImagePick } from "@/hooks/useImagePick";
import { usePointCloudView } from "@/hooks/usePointCloudView";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { useWebGPU } from "@/hooks/useWebGPU";
import {
  DEFAULT_DEPTH_MODEL,
  DEPTH_MODELS,
  MAX_INFERENCE_SIDE,
} from "@/vision/depth";
import { drawHeatmap } from "@/vision/draw";
import { downscale } from "@/vision/image";
import {
  DEFAULT_FOCAL_RATIO,
  unproject,
  type PointCloud,
} from "@/vision/pointCloud";
import { IMAGE_SAMPLES } from "@/vision/samples";
import { toPayload } from "@/vision/image";

export const Route = createFileRoute("/image-to-3d")({ component: ImageTo3DPage });

/** Stride choices, as "one point per N pixels on each axis". */
const STRIDES = [1, 2, 4, 8];

function ImageTo3DPage() {
  const session = useModelSelection({
    routeKey: "image-to-3d",
    models: DEPTH_MODELS,
    fallback:
      DEPTH_MODELS.find((m) => m.id === DEFAULT_DEPTH_MODEL) ?? DEPTH_MODELS[0],
  });
  const model = session.model.id;
  const probe = useBackendProbe();
  // LOAD is two things on this page: the weight download *and* the GPU probe,
  // because the 3-D half needs a device that the inference half does not.
  const gpu = useWebGPU();

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
  } = useDepth(model);
  useCacheRefresh(session, ready);

  const [source, setSource] = useState<RawImage | null>(null);
  const [live, setLive] = useState(false);
  const [focalRatio, setFocalRatio] = useState(DEFAULT_FOCAL_RATIO);
  const [stride, setStride] = useState(2);

  const estimate = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setSource(small);
      await run(small);
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (ready) await estimate(frame);
    },
    [ready, estimate],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  // **The whole reason the sliders are cheap.** The cloud is derived from the
  // depth map already in hand, so moving the focal length or the stride is
  // arithmetic over a few hundred thousand points — never another inference.
  // Same pure-derivation rule /vad uses for its threshold.
  const cloud: PointCloud | null = useMemo(() => {
    if (!result || !source) return null;
    const tensor = result.predicted_depth;
    const { width, height } = depthDims(tensor);
    if (width === 0 || height === 0) return null;
    return unproject(
      { data: tensor.data ?? [], width, height },
      {
        focal: focalRatio * Math.max(width, height),
        stride,
        // Depth Anything emits inverse depth (big = near); Depth Pro emits
        // metres (big = far). Reading this off the catalogue entry rather than
        // hard-coding it is what keeps the scene from turning inside out on the
        // metric model.
        inverse: !session.model.metric,
        colour: toPayload(source),
      },
    );
  }, [result, source, focalRatio, stride, session.model.metric]);

  // `result` as the fit key: the camera re-frames when a new depth map arrives
  // and stays put while the sliders re-derive from the one in hand.
  const view = usePointCloudView(cloud, result);

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const estimateCurrent = () => {
    if (!picked) return;
    clearError();
    void estimate(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Boxes}
      title="Image to 3D"
      description={
        <>
          Turn one photo into a point cloud you can orbit. Depth runs in a Web
          Worker, the cloud renders through hand-written WGSL — this is the
          depth-to-cloud half of image-to-3D; full reconstruction is diffusion
          and stays on a server.
        </>
      }
      labels={{ output: "Point cloud" }}
      select={
        <ModelPicker
          models={DEPTH_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          backend={probe}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
        />
      }
      load={
        <div className="space-y-3">
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
          {/* The GPU half of LOAD. Stated here even when it is fine, so the
              answer to "will the 3-D view work on this machine" is visible
              before a 50 MB download rather than after it. */}
          {!gpu.loading && gpu.capabilities?.status !== "ready" && (
            <p
              data-testid="webgpu-unavailable"
              className="text-xs text-amber-600 dark:text-amber-500"
            >
              No WebGPU device here, so the interactive 3-D view is unavailable.
              Depth estimation still runs, and the depth map is shown instead.
            </p>
          )}
        </div>
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a depth model to build a point cloud. You can pick a picture first."
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
                  <Boxes className="size-4" /> Build point cloud
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — a scene with a clear foreground and background makes the most convincing cloud."
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
            <CloudControls
              focalRatio={focalRatio}
              onFocalRatio={setFocalRatio}
              stride={stride}
              onStride={setStride}
              points={cloud?.count ?? null}
              metric={session.model.metric ?? false}
            />
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Point cloud"
          description="Drag to orbit, scroll to zoom. Geometry is plausible, not measured."
          meta={cloud ? `${cloud.count.toLocaleString()} points` : undefined}
          running={running}
          runningLabel="Estimating depth…"
          error={runError}
          empty="Pick an image and its depth map becomes a point cloud here — one you can turn around."
          actions={
            cloud &&
            view.supported && (
              <Button size="sm" variant="outline" onClick={view.reset}>
                <RotateCcw className="size-4" /> Reset view
              </Button>
            )
          }
        >
          {cloud && result && (
            <CloudView
              cloud={cloud}
              result={result}
              view={view}
              gpuReady={gpu.capabilities?.status === "ready"}
              gpuProbing={gpu.loading}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * The two sliders. Both re-derive the cloud; neither re-runs the model, which
 * is what makes them draggable rather than a form you submit.
 */
function CloudControls({
  focalRatio,
  onFocalRatio,
  stride,
  onStride,
  points,
  metric,
}: {
  focalRatio: number;
  onFocalRatio: (v: number) => void;
  stride: number;
  onStride: (v: number) => void;
  points: number | null;
  metric: boolean;
}) {
  return (
    <div className="space-y-3 rounded-lg border p-3">
      <div className="space-y-1.5">
        <Label htmlFor="focal" className="text-xs">
          Focal length
          <span className="ml-1 font-mono text-muted-foreground tabular-nums">
            {focalRatio.toFixed(2)}x long side
          </span>
        </Label>
        <input
          id="focal"
          type="range"
          min={0.3}
          max={2}
          step={0.05}
          value={focalRatio}
          data-testid="focal-slider"
          onChange={(e) => onFocalRatio(Number(e.target.value))}
          className="w-full"
        />
        <p className="text-xs text-amber-600 dark:text-amber-500">
          {metric ? (
            <>
              This model predicts metres and its own focal length. The slider
              overrides it — leave it at the default for the model's own geometry.
            </>
          ) : (
            <>
              This is an <span className="font-medium">assumption</span>, not a
              measurement: the model predicts relative depth with no camera
              intrinsics, so the shape is plausible rather than metric, and two
              photos' clouds cannot be compared.
            </>
          )}
        </p>
      </div>

      <div className="space-y-1.5">
        <Label className="text-xs">
          Point density
          {points != null && (
            <span className="ml-1 font-mono text-muted-foreground tabular-nums">
              {points.toLocaleString()} points
            </span>
          )}
        </Label>
        <div className="flex flex-wrap gap-1.5">
          {STRIDES.map((s) => (
            <Button
              key={s}
              size="sm"
              variant={stride === s ? "default" : "outline"}
              aria-pressed={stride === s}
              data-testid={`stride-${s}`}
              onClick={() => onStride(s)}
            >
              1 / {s * s}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}

function CloudView({
  cloud,
  result,
  view,
  gpuReady,
  gpuProbing,
}: {
  cloud: PointCloud;
  result: DepthResult;
  view: ReturnType<typeof usePointCloudView>;
  gpuReady: boolean;
  gpuProbing: boolean;
}) {
  const tensor = result.predicted_depth;
  const { width, height } = depthDims(tensor);

  const paintDepth = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawHeatmap(ctx, tensor.data ?? [], width, height);
    },
    [tensor, width, height],
  );

  // Degrade honestly. `detectWebGPU()` never throws, so the failure here is a
  // `false` — and a page that answered it with an empty canvas would look like
  // the model had failed rather than like the machine lacking a GPU.
  const canRender = gpuReady && view.supported !== false;

  return (
    <div className="space-y-3">
      {canRender ? (
        <canvas
          ref={view.canvasRef}
          data-testid="cloud-canvas"
          role="img"
          aria-label="Interactive point cloud — drag to orbit, scroll to zoom"
          onPointerDown={view.onPointerDown}
          onWheel={(e) => view.onWheel(e.deltaY)}
          className="h-72 w-full cursor-grab touch-none rounded active:cursor-grabbing"
        />
      ) : (
        <div className="space-y-2" data-testid="cloud-fallback">
          <ErrorNote
            message={
              gpuProbing
                ? null
                : "The 3-D view needs a WebGPU device, and this browser did not provide one. Here is the depth map the cloud would have been built from."
            }
          />
          <OverlayCanvas
            width={width}
            height={height}
            draw={paintDepth}
            label="Estimated depth map"
            testId="depth-map"
          />
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        <span className="tabular-nums">{cloud.count.toLocaleString()}</span>{" "}
        points from a {width}x{height} depth map
        {canRender && <> · drag to orbit, scroll to zoom</>}
      </p>
      <p className="text-xs text-muted-foreground">
        Depth is <span className="font-medium">relative</span>, and the focal
        length is assumed — so this is a plausible shape rather than a
        measurement of the scene.
      </p>
    </div>
  );
}
