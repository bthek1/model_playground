// Robotics — grounding an instruction in what the camera sees.
//
// Type "a red block", point the camera or pick a picture, and the page answers
// with a box from an open-vocabulary detector (OWLv2) and a *relative* distance
// for it from a depth model (Depth Anything V2 Small). The honest browser half
// of robot learning: perception, not control. The control half — a policy that
// turns this into motion — needs demonstrations, a simulator and real hardware
// latency, none of which a tab has, and OUTPUT says so before any result exists.
//
// Nothing here is a new model. Both checkpoints ship on their own pages; this
// route composes them (`vision/grounding.ts`), loads them as one pair in two
// workers (`hooks/useGrounding.ts`) and reads a distance per box off the depth
// map (`vision/groundingDepth.ts`). Four decisions, each guarding a failure:
//
//  1. **The ordering, not a number in metres.** Depth Anything emits inverse
//     depth on a per-image scale, so the page ranks the boxes ("nearest", "2nd
//     nearest") and quotes the raw value only as what it is. The direction comes
//     from the catalogue (`DepthModel.metric`), so an inverted reading cannot be
//     introduced here.
//  2. **The threshold re-derives.** The detector is asked once at a low floor;
//     dragging the slider re-filters and re-ranks on the main thread. Only the
//     GENERATE button ("Locate") spends.
//  3. **The result is the frame captured inside the run.** A new pick, or a new
//     phrase, must not restyle the previous answer — the boxes, the depth map
//     and the phrases they answer are captured together.
//  4. **OWLv2 wants labels, not sentences.** A phrase that reads like an
//     instruction is flagged beside the list rather than left to underperform.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Bot, Crosshair, Loader2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { CloningPage } from "@/components/robotics/CloningPage";
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
import { depthDims, type DepthResult } from "@/hooks/useDepth";
import { useGrounding, type GroundingResult } from "@/hooks/useGrounding";
import { useImagePick } from "@/hooks/useImagePick";
import { aboveThreshold } from "@/hooks/useObjectDetector";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  colorForLabel,
  drawBoxes,
  drawHeatmap,
  drawPixels,
  RAMP_CSS,
  rangeOf,
  scaleDetections,
} from "@/vision/draw";
import {
  DEFAULT_GROUNDING_QUERIES,
  DEFAULT_ROBOTICS_ENTRY,
  type GroundingEntry,
  ROBOTICS_ENTRIES,
  type RoboticsEntry,
  sentenceLike,
} from "@/vision/grounding";
import {
  groundDepths,
  nearestOf,
  rankLabel,
  type GroundedDetection,
} from "@/vision/groundingDepth";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";
import {
  DEFAULT_THRESHOLD,
  MAX_INFERENCE_SIDE,
  THRESHOLD_RANGE,
} from "@/vision/zeroShotDetection";

export const Route = createFileRoute("/robotics")({ component: RoboticsPage });

/**
 * The frame a result belongs to, captured inside the run: the original for
 * drawing, the size both models actually saw (the boxes' pixel space), the
 * factor between the two, and the phrases that were asked.
 */
interface Captured {
  image: RawImage;
  seen: { width: number; height: number };
  scale: number;
  asked: string[];
}

function RoboticsPage() {
  const session = useModelSelection({
    routeKey: "robotics",
    models: ROBOTICS_ENTRIES,
    fallback:
      ROBOTICS_ENTRIES.find((e) => e.id === DEFAULT_ROBOTICS_ENTRY) ??
      ROBOTICS_ENTRIES[0],
  });
  // Two kinds of entry, one picker (#54). Each view owns its own hooks, so the
  // cloning entry never mounts the grounding pair's two workers — and never
  // downloads a byte of them.
  if (session.model.kind === "cloning") {
    return (
      <CloningPage
        select={
          <ModelPicker
            models={ROBOTICS_ENTRIES}
            value={session.model.id}
            onChange={session.setModel}
          />
        }
      />
    );
  }
  return <GroundingPage session={session} entry={session.model} />;
}

function GroundingPage({
  session,
  entry,
}: {
  session: ReturnType<typeof useModelSelection<RoboticsEntry>>;
  entry: GroundingEntry;
}) {

  const {
    pair,
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    error,
    load,
    retry,
    cancel,
    run,
  } = useGrounding(entry.id);
  useCacheRefresh(session, ready);
  // Gates the pair in SELECT, before anything downloads, when either half
  // declares a backend this machine cannot give it (`GroundingEntry.backends`).
  const probe = useBackendProbe();

  // The pair has a composite id, which is never itself in the browser cache.
  // "Cached" means both halves are — one cached half still downloads the other.
  const pairCached =
    session.cached.has(pair.detector.id) && session.cached.has(pair.depth.id);
  const cachedIds = useMemo(
    () => new Set(pairCached ? [entry.id] : []),
    [pairCached, entry.id],
  );

  const [queries, setQueries] = useState<string[]>(DEFAULT_GROUNDING_QUERIES);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [live, setLive] = useState(false);
  // The answer and the frame it answers, set together from inside the run.
  // Reading the hook's `result` beside a separately-set frame would pair a new
  // result with the previous frame for one render on every press.
  const [shot, setShot] = useState<{
    frame: Captured;
    result: GroundingResult;
  } | null>(null);

  const ground = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      const seen = {
        width: small.width || image.width,
        height: small.height || image.height,
      };
      const asked = [...queries];
      const out = await run(small, asked, { consume: small !== image });
      setShot({
        frame: { image, seen, scale: image.width / seen.width, asked },
        result: out,
      });
    },
    [run, queries],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick();

  const onFrame = useCallback(
    async (f: RawImage) => {
      if (!ready || queries.length === 0) return;
      await ground(f);
    },
    [ready, queries, ground],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;
  const metric = pair.depth.metric ?? false;
  const wordy = sentenceLike(queries);

  // Pure, on the main thread: the threshold re-filters and re-ranks what the
  // detector already returned. Nothing is re-run.
  const grounded = useMemo(() => {
    if (!shot) return [];
    const { result, frame } = shot;
    const kept = aboveThreshold(result.detections, threshold).sort(
      (a, b) => b.score - a.score,
    );
    const tensor = result.depth.predicted_depth;
    const { width, height } = depthDims(tensor);
    return groundDepths(
      kept,
      { data: tensor.data ?? [], width, height },
      frame.seen,
      { metric },
    );
  }, [shot, threshold, metric]);

  const locateCurrent = () => {
    if (!picked) return;
    clearError();
    void ground(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Bot}
      title="Robotics"
      description={
        <>
          Ground an instruction in what the camera sees: name a thing, and an
          open-vocabulary detector boxes it while a depth model says which match
          is nearest. Two models, both on your own GPU, in two Web Workers.
        </>
      }
      labels={{ select: "Pair", run: "Phrases & frame", output: "Grounding" }}
      aside={
        live && camera.fps != null ? (
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {camera.fps} fps on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <div className="space-y-2">
          <ModelPicker
            models={ROBOTICS_ENTRIES}
            value={entry.id}
            onChange={session.setModel}
            disabled={loading || busy}
            cached={cachedIds}
            backend={probe}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Two models: <strong>{pair.detector.label}</strong> finds what you
            name, and <strong>{pair.depth.label}</strong> estimates depth for the
            same frame. The size above is the <strong>combined</strong> download
            — both are live at once — and both are the checkpoints{" "}
            <code>/zero-shot-object-detection</code> and <code>/depth</code>{" "}
            already use, so either page's cache covers its half.
          </p>
        </div>
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          loadProgress={loadProgress}
          loadedInMs={loadedInMs}
          cached={pairCached}
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
          disabledHint="Load the pair to locate your phrases. You can write them and pick a picture first."
          controls={
            <Button
              disabled={
                !ready || busy || !picked || live || queries.length === 0
              }
              onClick={locateCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Locating…
                </>
              ) : (
                <>
                  <Crosshair className="size-4" /> Locate
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the city street has cars and bicycles at clearly different distances."
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
                id="new-phrase"
                title="Phrases"
                items={queries}
                onChange={setQueries}
                placeholder="a red block"
                disabled={live}
                emptyHint="Add at least one thing to look for."
                hint={
                  pair.queries === "label" ? (
                    <>
                      Short noun phrases — <em>a red block</em>, not{" "}
                      <em>pick up the red block nearest the camera</em>.{" "}
                      {pair.detector.label} scores each phrase as a class name,
                      and sends it exactly as written.
                    </>
                  ) : (
                    <>This detector grounds free text, so a phrase works.</>
                  )
                }
              />
              {pair.queries === "label" && wordy.length > 0 && (
                <p
                  className="text-xs text-amber-600 dark:text-amber-500"
                  data-testid="phrase-warning"
                >
                  {wordy.map((q) => `“${q}”`).join(", ")}{" "}
                  {wordy.length === 1 ? "reads" : "read"} like a sentence. The
                  detector will score it as one long class name — the object
                  alone usually does better.
                </p>
              )}

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
                  Re-filters and re-ranks the boxes already found. Nothing
                  re-runs; pressing Locate is what asks the models again.
                </p>
              </div>

              <p className="text-xs text-muted-foreground">
                The camera runs <strong>both</strong> models on every frame, one
                frame at a time — the per-frame cost is the two times OUTPUT
                reports, added together.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Grounding"
          description={
            <span data-testid="grounding-limitation">
              This is <strong>grounding, not control</strong>: the page finds
              what you named and says which match is nearest, and issues no
              action. The control half does not port to a tab — behaviour
              cloning needs recorded demonstrations, a policy like PushT's needs
              a simulator, and action chunking is about real hardware latency.
              Distances are{" "}
              {metric ? "in metres" : "relative, not metres"}.
            </span>
          }
          meta={
            shot ? (
              <span className="tabular-nums">
                detect {shot.result.detectMs} ms · depth{" "}
                {shot.result.depthMs} ms
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Detecting, then estimating depth…"
          error={runError}
          empty="Name a thing, pick an image, and press Locate: every match is boxed, the depth map appears beside it, and the nearest match is called out."
        >
          {shot && (
            <GroundingView
              grounded={grounded}
              depth={shot.result.depth}
              frame={shot.frame}
              metric={metric}
              threshold={threshold}
              total={shot.result.detections.length}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function GroundingView({
  grounded,
  depth,
  frame,
  metric,
  threshold,
  total,
}: {
  grounded: readonly GroundedDetection[];
  depth: DepthResult;
  frame: Captured;
  metric: boolean;
  threshold: number;
  total: number;
}) {
  const tensor = depth.predicted_depth;
  const { width, height } = depthDims(tensor);
  const range = useMemo(() => rangeOf(tensor.data ?? []), [tensor]);
  const nearest = nearestOf(grounded);
  // Phrases asked *in this run* that found nothing — the list the result was
  // produced by, not the one in the box now.
  const missing = frame.asked.filter(
    (q) => !grounded.some((g) => g.label === q),
  );

  // Boxes are in the pixels the models saw; the canvas shows the original.
  const onSource = useMemo(
    () => scaleDetections(grounded, frame.scale),
    [grounded, frame.scale],
  );
  const onMap = useMemo(
    () => scaleDetections(grounded, width / (frame.seen.width || 1)),
    [grounded, width, frame.seen.width],
  );

  const paintSource = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, frame.image);
      drawBoxes(ctx, onSource);
      markNearest(ctx, onSource, grounded, 4);
    },
    [frame.image, onSource, grounded],
  );
  const paintDepth = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawHeatmap(ctx, tensor.data ?? [], width, height);
      drawBoxes(ctx, onMap, { lineWidth: 1, font: "11px system-ui, sans-serif" });
    },
    [tensor, width, height, onMap],
  );

  return (
    <div className="space-y-3">
      <div
        data-testid="nearest-callout"
        className="rounded-md border px-3 py-2 text-sm"
      >
        {nearest ? (
          <>
            Nearest match:{" "}
            <span className="font-medium">{nearest.label}</span>{" "}
            <span className="font-mono text-xs text-muted-foreground tabular-nums">
              score {nearest.score.toFixed(2)}
            </span>
            {grounded.length > 1 && (
              <span className="text-muted-foreground">
                {" "}
                — nearer than the other {grounded.length - 1}
              </span>
            )}
          </>
        ) : (
          <>
            Nothing above {threshold.toFixed(2)} matched any phrase
            {total > 0 ? ` (${total} below it)` : ""}. The page will not box
            something it was not asked for.
          </>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <figure className="space-y-1">
          <OverlayCanvas
            width={frame.image.width}
            height={frame.image.height}
            draw={paintSource}
            label={`Matches: ${grounded.length}`}
            testId="grounding-canvas"
          />
          <figcaption className="text-xs text-muted-foreground">
            The frame the models saw
          </figcaption>
        </figure>
        <figure className="space-y-1">
          <OverlayCanvas
            width={width}
            height={height}
            draw={paintDepth}
            label="Estimated depth map"
            testId="grounding-depth"
          />
          <figcaption className="text-xs text-muted-foreground">
            Depth · {width}x{height}
          </figcaption>
        </figure>
      </div>

      <div className="space-y-1">
        <div
          aria-hidden
          className="h-3 w-full rounded-sm"
          style={{ background: RAMP_CSS }}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{metric ? "near" : "far"}</span>
          <span className="font-mono tabular-nums">
            {range.lo.toFixed(2)} … {range.hi.toFixed(2)}
            {metric ? " m" : ""}
          </span>
          <span>{metric ? "far" : "near"}</span>
        </div>
        {!metric && (
          <p className="text-xs text-amber-600 dark:text-amber-500">
            Relative inverse depth on this frame's own scale —{" "}
            <span className="font-medium">not metres</span>. The page ranks the
            matches rather than quoting a distance it does not have.
          </p>
        )}
      </div>

      <ul className="space-y-1.5 text-sm" data-testid="grounding-rows">
        {grounded.map((g, i) => {
          const cx = (g.box.xmin + g.box.xmax) / 2 / frame.seen.width;
          const cy = (g.box.ymin + g.box.ymax) / 2 / frame.seen.height;
          return (
            <li
              key={`${g.label}-${i}`}
              data-testid="grounding-row"
              data-label={g.label}
              data-rank={g.rank ?? ""}
              data-cx={cx.toFixed(3)}
              data-cy={cy.toFixed(3)}
              className="flex flex-wrap items-center gap-x-2"
            >
              <span
                aria-hidden
                className="size-3 shrink-0 rounded-sm"
                style={{ background: colorForLabel(g.label) }}
              />
              <span className="font-medium">{g.label}</span>
              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                {g.score.toFixed(2)}
              </span>
              <span className={g.nearest ? "font-medium" : "text-muted-foreground"}>
                {rankLabel(g.rank)}
              </span>
              {g.nearness != null && g.depth != null && (
                <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                  nearness {g.nearness.toFixed(2)} · raw {g.depth.toFixed(2)}
                </span>
              )}
            </li>
          );
        })}
      </ul>
      {missing.length > 0 && (
        <p className="text-xs text-muted-foreground" data-testid="grounding-missing">
          Nothing above the threshold for:{" "}
          {missing.map((q) => `“${q}”`).join(", ")}
        </p>
      )}
      {grounded.length > 0 && (
        <p className="text-xs text-muted-foreground">
          <em>Nearness</em> is where the box's median depth sits between the
          farthest (0) and nearest (1) point in this frame. <em>Raw</em> is the
          depth model's own output, {metric ? "in metres" : "bigger = nearer"}.
        </p>
      )}
    </div>
  );
}

/** A heavier outline on the nearest match, so the answer reads at a glance. */
function markNearest(
  ctx: CanvasRenderingContext2D,
  boxes: readonly { box: { xmin: number; ymin: number; xmax: number; ymax: number } }[],
  grounded: readonly GroundedDetection[],
  lineWidth: number,
) {
  const i = grounded.findIndex((g) => g.nearest);
  if (i < 0) return;
  const { box } = boxes[i];
  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = colorForLabel(grounded[i].label);
  ctx.setLineDash([8, 4]);
  ctx.strokeRect(
    box.xmin - lineWidth,
    box.ymin - lineWidth,
    box.xmax - box.xmin + lineWidth * 2,
    box.ymax - box.ymin + lineWidth * 2,
  );
  ctx.restore();
}
