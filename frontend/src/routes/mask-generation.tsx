// Mask Generation — click a point, get a mask in milliseconds. SAM's
// architecture (encode once, decode many) was designed for exactly the
// interaction a web page provides, which is why the browser version feels as
// good as the desktop one.
//
// Four decisions, each guarding a specific failure:
//
//  1. **Two states, not one.** LOAD covers the download; a separate *encoding*
//     state covers the per-image vision-encoder pass. Collapsing them means the
//     user clicks, waits a second, and is told nothing — the exact experience
//     the split architecture exists to avoid. The page shows both, and shows
//     the decode time next to each mask, because sub-100 ms is a claim and this
//     page can simply prove it.
//  2. **Clicks land on the picture, in the RUN slot.** The point markers are
//     *input*, so they are drawn over the input surface, not over the result
//     (model-page-pattern.md §4) — `ImageSourcePanel`'s `preview` override
//     exists for this one case. The mask is the result and lives in OUTPUT.
//  3. **The click is converted to source pixels by the canvas, not here.** The
//     canvas is sized to the source and scaled down by CSS, so a raw offset is
//     wrong by the scale factor — and a mis-mapped point produces a perfectly
//     plausible mask of whatever happens to be there. `OverlayCanvas`'s
//     `onPick` owns that arithmetic.
//  4. **All three candidates are offered.** SAM returns three masks per click
//     precisely because a point is ambiguous — the wheel, the door, or the
//     whole car — and showing only the top-scoring one hides the most
//     interesting thing about the model.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, MousePointerClick, Scissors, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { Button } from "@/components/ui/button";
import { useImagePick } from "@/hooks/useImagePick";
import { useSam } from "@/hooks/useSam";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { drawMasks, drawPixels, OVERLAY_COLORS } from "@/vision/draw";
import { downscale } from "@/vision/image";
import { coverage } from "@/vision/sam/sam";
import {
  DEFAULT_SAM_MODEL,
  MAX_INFERENCE_SIDE,
  SAM_MODELS,
  type SamMask,
  type SamPoint,
} from "@/vision/sam/types";
import { IMAGE_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/mask-generation")({
  component: MaskGenerationPage,
});

function MaskGenerationPage() {
  const session = useModelSelection({
    routeKey: "mask-generation",
    models: SAM_MODELS,
    fallback:
      SAM_MODELS.find((m) => m.id === DEFAULT_SAM_MODEL) ?? SAM_MODELS[0],
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
    encoding,
    encoded,
    encodedInMs,
    encode,
    run,
    reset,
    load,
    retry,
    cancel,
  } = useSam(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [points, setPoints] = useState<SamPoint[]>([]);
  const [candidate, setCandidate] = useState(0);
  // The frame that was encoded, at the resolution the masks come back in.
  const [frame, setFrame] = useState<RawImage | null>(null);

  const prepare = useCallback(
    async (token: string, image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setFrame(small);
      setPoints([]);
      setCandidate(0);
      await encode(`${token}:${small.width}x${small.height}`, small);
    },
    [encode],
  );

  const { picked, preparing, error: ioError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        reset();
        setPoints([]);
        setCandidate(0);
        // A picture the user has picked is shown whether or not a model exists,
        // so the page is legible before the download. Only the encode is gated.
        if (ready) await prepare(next.name, next.image);
        else setFrame(await downscale(next.image, MAX_INFERENCE_SIDE));
      },
    });

  // Loading a model with a picture already on screen encodes it, rather than
  // making the user re-pick to get to a usable state.
  useEffect(() => {
    if (!ready || !picked || encoded || encoding) return;
    void prepare(picked.name, picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  }, [ready, picked, encoded, encoding, prepare]);

  const busy = running || encoding || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const decode = useCallback(
    (next: SamPoint[]) => {
      if (next.length === 0) return;
      void run(next).catch(() => {
        /* the hook surfaces it in OUTPUT */
      });
    },
    [run],
  );

  const addPoint = useCallback(
    (point: SamPoint) => {
      setPoints((prev) => {
        const next = [...prev, point];
        decode(next);
        return next;
      });
      setCandidate(0);
    },
    [decode],
  );

  const clearPoints = () => {
    setPoints([]);
    setCandidate(0);
    reset();
  };

  const masks = result?.masks ?? [];
  const shown = masks[candidate] ?? null;

  return (
    <ModelPage
      icon={Scissors}
      title="Mask Generation"
      description={
        <>
          Click anything in the picture and SAM cuts it out. The image is encoded
          once; every click after that is a few milliseconds on your own GPU.
        </>
      }
      labels={{ output: "Mask" }}
      aside={
        encoded && result ? (
          <span
            data-testid="decode-ms"
            className="font-mono text-xs text-muted-foreground tabular-nums"
          >
            decode {Math.round(result.ms)} ms on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <ModelPicker
          models={SAM_MODELS}
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
          disabledHint="Load a model to cut things out. You can pick a picture first."
          controls={
            <>
              <Button
                variant="outline"
                disabled={!ready || !encoded || points.length === 0}
                onClick={clearPoints}
              >
                <Trash2 className="size-4" /> Clear points
              </Button>
              <Button
                variant="outline"
                disabled={!ready || !encoded || !frame || busy}
                onClick={() =>
                  frame &&
                  addPoint({
                    x: frame.width / 2,
                    y: frame.height / 2,
                    positive: true,
                  })
                }
              >
                {/* The keyboard route to a mask. Clicking a canvas is
                    inherently a pointer gesture, and a page whose only input is
                    a click is a page some people cannot use at all. */}
                <MousePointerClick className="size-4" /> Point at the centre
              </Button>
            </>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the street and football shots have the most things worth cutting out of them."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
            preview={
              frame ? (
                <OverlayCanvas
                  width={frame.width}
                  height={frame.height}
                  draw={makePointPainter(frame, points)}
                  onPick={
                    encoded && !busy
                      ? ({ x, y, alt }) =>
                          addPoint({ x, y, positive: !alt })
                      : undefined
                  }
                  label={
                    encoded
                      ? `Click to place a point. ${points.length} placed.`
                      : "The picture, not yet encoded."
                  }
                  testId="point-canvas"
                  className="max-h-72"
                />
              ) : undefined
            }
          >
            <div className="space-y-1.5">
              <EncodeState
                ready={ready}
                encoding={encoding}
                encoded={encoded}
                encodedInMs={encodedInMs}
                hasImage={frame !== null}
              />
              <p className="text-xs text-muted-foreground">
                Click the picture to say “this is the object”.{" "}
                <kbd className="rounded border px-1">Alt</kbd>-click to say “this
                is not” — a second point is how you tell SAM you meant the wheel
                and not the car.
              </p>
              {points.length > 0 && (
                <ul
                  data-testid="points"
                  className="flex flex-wrap gap-1.5 font-mono text-xs"
                >
                  {points.map((p, i) => (
                    <li
                      key={`${p.x}-${p.y}-${i}`}
                      className="rounded-md border px-2 py-1 tabular-nums"
                    >
                      <span
                        aria-hidden
                        className="mr-1 inline-block size-2 rounded-full align-middle"
                        style={{ background: pointColor(p) }}
                      />
                      {p.positive ? "+" : "−"} {Math.round(p.x)},{" "}
                      {Math.round(p.y)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Mask"
          description="The mask composited over the frame SAM saw, with its own IoU estimate."
          meta={
            result ? (
              <span className="tabular-nums">
                {masks.length} candidate{masks.length === 1 ? "" : "s"}
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Decoding…"
          error={runError}
          empty="Pick a picture, then click something in it. The mask appears here in a few milliseconds."
        >
          {shown && frame && (
            <MaskView
              source={frame}
              masks={masks}
              selected={candidate}
              onSelect={setCandidate}
              ms={result?.ms ?? 0}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * The encode state, spelled out. This is the half of the wait that is not the
 * download, and the user has no way to guess it exists.
 */
function EncodeState({
  ready,
  encoding,
  encoded,
  encodedInMs,
  hasImage,
}: {
  ready: boolean;
  encoding: boolean;
  encoded: boolean;
  encodedInMs: number | null;
  hasImage: boolean;
}) {
  if (!ready || !hasImage) return null;
  if (encoding) {
    return (
      <p
        data-testid="encoding"
        className="flex items-center gap-2 text-sm text-muted-foreground"
      >
        <Loader2 className="size-4 animate-spin" />
        Encoding the image — once, for every click that follows.
      </p>
    );
  }
  if (!encoded) return null;
  return (
    <p data-testid="encoded" className="text-sm text-muted-foreground">
      Encoded
      {encodedInMs != null && (
        <>
          {" "}
          in <span className="tabular-nums">{encodedInMs} ms</span>
        </>
      )}
      . Clicks from here are decode-only.
    </p>
  );
}

const POSITIVE = OVERLAY_COLORS[3]; // green
const NEGATIVE = OVERLAY_COLORS[4]; // red

function pointColor(point: SamPoint): string {
  return point.positive ? POSITIVE : NEGATIVE;
}

/** Paint the frame plus its point markers. Stable per (frame, points). */
function makePointPainter(source: RawImage, points: readonly SamPoint[]) {
  return (ctx: CanvasRenderingContext2D) => {
    drawPixels(ctx, source);
    for (const point of points) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = pointColor(point);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      ctx.restore();
    }
  };
}

function MaskView({
  source,
  masks,
  selected,
  onSelect,
  ms,
}: {
  source: RawImage;
  masks: readonly SamMask[];
  selected: number;
  onSelect: (i: number) => void;
  ms: number;
}) {
  const mask = masks[selected];
  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, source);
      if (!mask) return;
      drawMasks(
        ctx,
        [
          {
            label: "mask",
            data: mask.data,
            width: mask.width,
            height: mask.height,
          },
        ],
        { width: source.width, height: source.height },
      );
    },
    [source, mask],
  );

  const covered = useMemo(() => (mask ? coverage(mask) : 0), [mask]);

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={source.width}
        height={source.height}
        draw={paint}
        label={`Mask candidate ${selected + 1} of ${masks.length}`}
        testId="mask-canvas"
      />

      <div className="space-y-1.5">
        <span className="text-sm font-medium">Candidates</span>
        <div className="flex flex-wrap gap-2">
          {masks.map((m, i) => (
            <Button
              key={i}
              variant="outline"
              size="sm"
              aria-pressed={i === selected}
              onClick={() => onSelect(i)}
            >
              {i + 1} · <span className="tabular-nums">{m.score.toFixed(3)}</span>
            </Button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          {/* The candidates are the interesting part: a single point is
              ambiguous by construction — the wheel, the door, or the whole car
              — and SAM returns one mask for each reading rather than guessing. */}
          A point is ambiguous on purpose. SAM returns one mask per reading of
          it, with its own IoU estimate; the highest score is not always the one
          you meant.
        </p>
      </div>

      <p data-testid="mask-facts" className="text-xs text-muted-foreground">
        Decoded in <span className="tabular-nums">{Math.round(ms)} ms</span> ·
        covers <span className="tabular-nums">{(covered * 100).toFixed(1)}%</span>{" "}
        of the frame · IoU estimate{" "}
        <span className="tabular-nums">{mask?.score.toFixed(3) ?? "—"}</span>
      </p>
    </div>
  );
}
