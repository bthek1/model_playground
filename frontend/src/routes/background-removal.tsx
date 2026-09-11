// Background Removal — an alpha matte from a single photo, composited in the
// browser. The preprocessing step everyone skips before image-to-3D, and a page
// that stands on its own.
//
// Two obligations, and both are claims the page has to make out loud:
//
//  1. **The soft edge is the output.** The models here are *matting* models, not
//     segmenters: their answer at a strand of hair is 0.4, not 0 or 1. So the
//     page never thresholds — the cut-out is blended, and the raw matte is
//     offered as its own view so the edge can be judged rather than assumed.
//     `vision/matte.ts` holds the arithmetic and the test that pins it.
//  2. **RMBG-1.4 is not free for commercial use.** It is the better matte and it
//     is offered, but MODNet (Apache-2.0) is the default, and picking RMBG puts
//     the restriction on screen next to the choice rather than in a footnote.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, Scissors, Scissors as Cut } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { DownloadImageButton } from "@/components/vision/DownloadImageButton";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { LicenceNote } from "@/components/vision/LicenceNote";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { Button } from "@/components/ui/button";
import { useBackgroundRemoval } from "@/hooks/useBackgroundRemoval";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useImagePick } from "@/hooks/useImagePick";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_MATTE_MODEL,
  MATTE_MODELS,
  MAX_INFERENCE_SIDE,
} from "@/vision/backgroundRemoval";
import { downscale } from "@/vision/image";
import {
  compositeOver,
  coveredFraction,
  drawCutout,
  drawRgba,
  matteToGrey,
  type Rgb,
  type RgbaImage,
} from "@/vision/matte";
import { IMAGE_SAMPLES, PORTRAIT_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/background-removal")({
  component: BackgroundRemovalPage,
});

/** What to put behind the subject. "None" is the cut-out itself. */
const BACKGROUNDS: { id: string; label: string; colour: Rgb | null }[] = [
  { id: "none", label: "Transparent", colour: null },
  { id: "white", label: "White", colour: [255, 255, 255] },
  { id: "black", label: "Black", colour: [17, 17, 17] },
  { id: "green", label: "Studio green", colour: [22, 163, 74] },
  { id: "blue", label: "Studio blue", colour: [37, 99, 235] },
];

type View = "cutout" | "matte";

function BackgroundRemovalPage() {
  const session = useModelSelection({
    routeKey: "background-removal",
    models: MATTE_MODELS,
    fallback:
      MATTE_MODELS.find((m) => m.id === DEFAULT_MATTE_MODEL) ?? MATTE_MODELS[0],
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
    error,
    load,
    retry,
    cancel,
    run,
    meta,
  } = useBackgroundRemoval(model);
  useCacheRefresh(session, ready);

  const [live, setLive] = useState(false);
  const [background, setBackground] = useState(BACKGROUNDS[0]);
  const [view, setView] = useState<View>("cutout");

  const cut = useCallback(
    async (image: RawImage) => {
      await run(await downscale(image, MAX_INFERENCE_SIDE));
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (ready) await run(frame, { consume: true });
    },
    [ready, run],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const cutCurrent = () => {
    if (!picked) return;
    clearError();
    void cut(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Scissors}
      title="Background Removal"
      description={
        <>
          Cut the subject out of a photo on your own machine. One forward pass
          produces a soft alpha matte — the picture is never uploaded.
        </>
      }
      labels={{ output: "Cut-out" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={MATTE_MODELS}
            value={model}
            onChange={session.setModel}
            disabled={loading || busy}
            backend={probe}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          {/* Beside the choice, not in a footnote: the licence is a property of
              the model, and this is the moment the user is picking one. */}
          <LicenceNote licence={meta.licence} />
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
          disabledHint="Load a model to remove a background. You can pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked || live}
              onClick={cutCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Cutting out…
                </>
              ) : (
                <>
                  <Cut className="size-4" /> Remove background
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            // Portraits first, and that ordering is load-bearing: MODNet is a
            // *portrait* matting model, and on a photo with no person in it it
            // returns an almost empty matte rather than failing — which reads
            // as a broken page. RMBG-1.4 is the general-purpose one.
            samples={[...PORTRAIT_SAMPLES, ...IMAGE_SAMPLES]}
            sampleHint="Samples — the portraits suit MODNet, which is a portrait matting model. RMBG-1.4 handles the general scenes below it."
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
          title="Cut-out"
          description="The matte is soft — partly-covered pixels stay partly covered."
          running={running}
          runningLabel="Removing the background…"
          error={runError}
          empty="Pick an image and the subject appears here on a transparency checkerboard, with the raw matte a click away."
          actions={
            result && (
              <DownloadImageButton image={result} filename="cutout.png" />
            )
          }
        >
          {result && (
            <CutoutView
              result={result}
              background={background}
              onBackground={setBackground}
              view={view}
              onView={setView}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function CutoutView({
  result,
  background,
  onBackground,
  view,
  onView,
}: {
  result: RgbaImage;
  background: (typeof BACKGROUNDS)[number];
  onBackground: (b: (typeof BACKGROUNDS)[number]) => void;
  view: View;
  onView: (v: View) => void;
}) {
  // A background swap and a view flip are **derivations**, not runs: the matte
  // is already in `result`'s alpha channel, so changing what sits behind the
  // subject is arithmetic over pixels we already have. Same rule /vad applies to
  // its threshold and /object-detection to its score floor.
  const painted = useMemo(() => {
    if (view === "matte") return matteToGrey(result);
    return background.colour ? compositeOver(result, background.colour) : result;
  }, [result, background, view]);

  const transparent = view === "cutout" && background.colour === null;

  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      if (transparent) drawCutout(ctx, painted);
      else drawRgba(ctx, painted);
    },
    [painted, transparent],
  );

  const covered = useMemo(() => coveredFraction(result), [result]);

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={result.width}
        height={result.height}
        draw={paint}
        label={view === "matte" ? "Alpha matte" : "Subject with the background removed"}
        testId={view === "matte" ? "matte-view" : "cutout-view"}
      />

      <div className="flex flex-wrap gap-1.5">
        {(["cutout", "matte"] as const).map((v) => (
          <Button
            key={v}
            size="sm"
            variant={view === v ? "default" : "outline"}
            aria-pressed={view === v}
            onClick={() => onView(v)}
          >
            {v === "cutout" ? "Cut-out" : "Matte"}
          </Button>
        ))}
      </div>

      {view === "cutout" && (
        <div className="space-y-1.5">
          <p className="text-xs text-muted-foreground">Background</p>
          <div className="flex flex-wrap gap-1.5">
            {BACKGROUNDS.map((b) => (
              <Button
                key={b.id}
                size="sm"
                variant={background.id === b.id ? "default" : "outline"}
                aria-pressed={background.id === b.id}
                onClick={() => onBackground(b)}
              >
                {b.colour && (
                  <span
                    aria-hidden
                    className="size-3 rounded-full border border-current/30"
                    style={{ background: `rgb(${b.colour.join(" ")})` }}
                  />
                )}
                {b.label}
              </Button>
            ))}
          </div>
        </div>
      )}

      <p data-testid="matte-coverage" className="text-xs text-muted-foreground">
        <span className="tabular-nums">{(covered * 100).toFixed(1)}%</span> of
        the frame is subject
        <span className="mx-1 opacity-50">·</span>
        {result.width}x{result.height}
      </p>

      <p className="text-xs text-muted-foreground">
        Partly-covered pixels — a strand of hair, a blurred edge — keep their
        in-between value rather than being rounded to fully in or fully out.
        Switch to <span className="font-medium">Matte</span> to see them.
      </p>
    </div>
  );
}
