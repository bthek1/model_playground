// Super Resolution — 2x upscaling with Swin2SR, tile by tile, in the browser.
//
// This route sits on the taxonomy's **Image to Image** slug, and the slug
// promises more than the model delivers. Image-to-image *editing* is diffusion:
// dozens of denoising passes over a latent, which is not a thing a tab does. So
// the page says, in the header and not in a tooltip, that it covers
// super-resolution specifically and that the editing half stays on a server
// (docs/roadmaps/vision.md §3.12).
//
// Two things shape everything below:
//
//  1. **A run is many inferences, not one.** A photo is cut into overlapping
//     tiles and each is its own forward pass, because a transformer on a
//     full-resolution image exhausts memory. That makes the run *long* — thirty
//     tiles on CPU is minutes — so the cost is quoted in tiles and seconds
//     **before** the button is pressed, and Stop is a first-class control rather
//     than a refresh.
//  2. **The comparison is the output.** A 2x image on its own proves nothing.
//     The result is shown under a draggable split against a bicubic upscale of
//     the same input, which is the only presentation that shows the model doing
//     work rather than merely producing pixels.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, Maximize2, Square, Wand2 } from "lucide-react";
import { useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { CompareSlider } from "@/components/vision/CompareSlider";
import { DownloadImageButton } from "@/components/vision/DownloadImageButton";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { Button } from "@/components/ui/button";
import { useImagePick } from "@/hooks/useImagePick";
import { useSuperRes, RunCancelled } from "@/hooks/useSuperRes";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import type { PixelBuffer, Pixels } from "@/vision/draw";
import { downscale } from "@/vision/image";
import { upscale } from "@/vision/resample";
import { IMAGE_SAMPLES } from "@/vision/samples";
import {
  DEFAULT_SUPER_RES_MODEL,
  formatDuration,
  MAX_SOURCE_SIDE,
  MS_PER_TILE,
  SUPER_RES_MODELS,
} from "@/vision/superRes";
import { planTiles } from "@/vision/tile";

export const Route = createFileRoute("/super-resolution")({
  component: SuperResolutionPage,
});

function SuperResolutionPage() {
  const session = useModelSelection({
    routeKey: "super-resolution",
    models: SUPER_RES_MODELS,
    fallback:
      SUPER_RES_MODELS.find((m) => m.id === DEFAULT_SUPER_RES_MODEL) ??
      SUPER_RES_MODELS[0],
  });
  const model = session.model.id;
  const probe = useBackendProbe();
  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    result,
    source,
    tiles,
    error,
    load,
    retry,
    cancel,
    run,
    stop,
    meta,
  } = useSuperRes(model, session.autoLoad);
  useCacheRefresh(session, ready);

  // The capped source, kept so the guard can quote a tile count before the run
  // and so the comparison has the exact pixels the model was given.
  const [input, setInput] = useState<RawImage | null>(null);

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        setInput(await downscale(next.image, MAX_SOURCE_SIDE));
      },
    });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // Quoted **before** the run starts. Discovering that an upscale is four
  // minutes long halfway through it is the difference between a slow page and a
  // page that appears to have hung.
  const guard = useMemo(() => {
    if (!input) return null;
    const count = planTiles(input.width, input.height).tiles.length;
    const perTile = MS_PER_TILE[backend ?? probe ?? "wasm"] ?? MS_PER_TILE.wasm;
    return {
      tiles: count,
      out: { width: input.width * meta.scale, height: input.height * meta.scale },
      duration: formatDuration(count * perTile),
    };
  }, [input, backend, probe, meta.scale]);

  const upscaleCurrent = () => {
    if (!input) return;
    clearError();
    void run(input).catch((e) => {
      // Stopping is a decision, not a failure — the hook surfaces real errors
      // in OUTPUT, and this one has nothing to report.
      if (!(e instanceof RunCancelled)) return;
    });
  };

  return (
    <ModelPage
      icon={Maximize2}
      title="Super Resolution"
      description={
        <>
          Double a picture's resolution on your own machine, one overlapping tile
          at a time. This is the <span className="font-medium">image-to-image</span>{" "}
          task's super-resolution half — editing and img2img are diffusion, and
          stay on a server.
        </>
      }
      labels={{ output: "Comparison" }}
      select={
        <ModelPicker
          models={SUPER_RES_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          backend={probe}
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
          restoring={session.restoring}
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
          disabledHint="Load a model to upscale. You can pick a picture first."
          controls={
            <>
              <Button
                disabled={!ready || busy || !input}
                onClick={upscaleCurrent}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Upscaling…
                  </>
                ) : (
                  <>
                    <Wand2 className="size-4" /> Upscale 2x
                  </>
                )}
              </Button>
              {running && (
                <Button variant="outline" onClick={stop} data-testid="stop-run">
                  <Square className="size-4" /> Stop
                </Button>
              )}
              {tiles && (
                <span
                  data-testid="tile-progress"
                  className="self-center font-mono text-xs text-muted-foreground tabular-nums"
                >
                  tile {tiles.done} / {tiles.total}
                </span>
              )}
            </>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — fine texture (fur, brickwork, lettering) is where a super-resolution model earns its keep."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          />

          {guard && (
            <p
              data-testid="size-guard"
              className="text-xs text-muted-foreground"
            >
              {input?.width}x{input?.height} →{" "}
              <span className="tabular-nums">
                {guard.out.width}x{guard.out.height}
              </span>
              <span className="mx-1 opacity-50">·</span>
              <span className="tabular-nums">{guard.tiles}</span>{" "}
              {guard.tiles === 1 ? "tile" : "tiles"}
              <span className="mx-1 opacity-50">·</span>
              {guard.duration} on {backend ?? probe ?? "this machine"}. You can
              stop part-way.
            </p>
          )}
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Model vs bicubic"
          description="Drag the handle. Left is the model, right is a plain bicubic upscale of the same input."
          meta={result ? `${result.width}x${result.height}` : undefined}
          running={running}
          runningLabel={
            tiles ? `Upscaling tile ${tiles.done} of ${tiles.total}…` : "Upscaling…"
          }
          error={runError}
          empty="Pick an image and upscale it. The result appears here beside a bicubic upscale of the same picture, so the difference is visible rather than asserted."
          actions={
            result && (
              <DownloadImageButton image={result} filename="upscaled.png" />
            )
          }
        >
          {result && source && <Comparison result={result} source={source} />}
        </OutputPanel>
      }
    />
  );
}

function Comparison({
  result,
  source,
}: {
  result: PixelBuffer;
  source: Pixels;
}) {
  // The baseline. Derived from the same source the model saw, at the same
  // output size — anything else compares two different questions.
  const baseline = useMemo(
    () => upscale(source, Math.round(result.width / source.width)),
    [source, result.width],
  );

  return (
    <div className="space-y-3">
      <CompareSlider
        before={baseline}
        after={result}
        beforeLabel="Bicubic"
        afterLabel="Swin2SR"
        testId="sr-compare"
      />
      <p className="text-xs text-muted-foreground">
        Both halves are the same {result.width}x{result.height} pixels the
        download contains. Bicubic is what the browser itself would do; the model
        is one forward pass per overlapping tile, feathered at the seams.
      </p>
    </div>
  );
}
