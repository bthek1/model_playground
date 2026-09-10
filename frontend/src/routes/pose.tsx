// Keypoint Detection — 17 COCO joints per person, drawn as a skeleton over a
// photo or a live camera feed.
//
// **The one page that deliberately holds two models live** (roadmap §5). Top-down
// pose is a detector followed by a pose model on each person's crop, and neither
// half is useful alone. Four consequences, each visible on the page:
//
//  1. **The size guardrail fires on the sum.** A catalogue entry names both
//     checkpoints and quotes their combined download — 180 MB for the nano pair.
//     Quoting half the bytes would be worse than quoting none, and the pair
//     crosses `LARGE_MODEL_BYTES` less obviously than a single big model does.
//  2. **One aggregate progress bar, not two.** Both models download together, so
//     their progress events interleave and `model/progress.ts` has both
//     denominators before either file completes.
//  3. **The threshold changes what is computed, not what is shown.** Unlike
//     `/object-detection`'s slider, this one travels in the run payload:
//     filtering afterwards would mean running the pose model on people the user
//     has already excluded, and that pass is the expensive half.
//  4. **A low-confidence joint is dimmed, not hidden and not drawn boldly.** A
//     heatmap always has a maximum somewhere, so an occluded ankle comes back as
//     a confident-looking guess. Opacity is how the page tells the truth about
//     it.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, PersonStanding } from "lucide-react";
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
import { usePose } from "@/hooks/usePose";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { drawPixels, OVERLAY_COLORS } from "@/vision/draw";
import { downscale } from "@/vision/image";
import { scalePeople } from "@/vision/pose/pose";
import {
  COCO_KEYPOINTS,
  drawSkeleton,
  JOINT_CONFIDENCE,
  type Person,
} from "@/vision/pose/skeleton";
import {
  DEFAULT_MAX_PEOPLE,
  DEFAULT_PERSON_THRESHOLD,
  DEFAULT_POSE_MODEL,
  MAX_INFERENCE_SIDE,
  MAX_PEOPLE_LIMIT,
  POSE_MODELS,
} from "@/vision/pose/types";
import { IMAGE_SAMPLES } from "@/vision/samples";

export const Route = createFileRoute("/pose")({
  component: PosePage,
});

function PosePage() {
  const backendProbe = useBackendProbe();
  const session = useModelSelection({
    routeKey: "pose",
    models: POSE_MODELS,
    fallback:
      POSE_MODELS.find((m) => m.id === DEFAULT_POSE_MODEL) ?? POSE_MODELS[0],
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
  } = usePose(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [threshold, setThreshold] = useState(DEFAULT_PERSON_THRESHOLD);
  const [maxPeople, setMaxPeople] = useState(DEFAULT_MAX_PEOPLE);
  const [live, setLive] = useState(false);
  const [frame, setFrame] = useState<{ image: RawImage; scale: number } | null>(
    null,
  );

  const estimate = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setFrame({ image, scale: image.width / (small.width || image.width) });
      await run(small, {
        threshold,
        maxPeople,
        consume: small !== image,
      });
    },
    [run, threshold, maxPeople],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        if (ready) await estimate(next.image);
        else setFrame({ image: next.image, scale: 1 });
      },
    });

  const onFrame = useCallback(
    async (f: RawImage) => {
      if (!ready) return;
      await estimate(f);
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

  // The skeletons come back in the inference frame's pixels while the canvas
  // shows the original. Scaling the boxes but not the joints would shrink each
  // skeleton away from its own outline.
  const people = useMemo(
    () => scalePeople(result?.people ?? [], frame?.scale ?? 1),
    [result, frame],
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
      icon={PersonStanding}
      title="Keypoint Detection"
      description={
        <>
          Find every person and draw their skeleton — two models running back to
          back on your own GPU, one to find people and one to pose them.
        </>
      }
      labels={{ output: "Skeletons" }}
      aside={
        live && camera.fps != null ? (
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {camera.fps} fps on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <ModelPicker
          models={POSE_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
          backend={backendProbe}
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
          disabledHint="Load both models to find poses. You can pick a picture first."
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
                  <PersonStanding className="size-4" /> Find poses
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the football match is the one with several people to pose at once."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
            camera={
              session.model.live
                ? {
                    live,
                    onToggle: setLive,
                    videoRef: camera.videoRef,
                    error: camera.error,
                    fps: camera.fps,
                  }
                : undefined
            }
          >
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="person-threshold">
                  Person confidence:{" "}
                  <span className="tabular-nums">{threshold.toFixed(2)}</span>
                </Label>
                <input
                  id="person-threshold"
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="block w-full max-w-xs"
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="max-people">
                  Most people to pose:{" "}
                  <span className="tabular-nums">{maxPeople}</span>
                </Label>
                <input
                  id="max-people"
                  type="range"
                  min={1}
                  max={MAX_PEOPLE_LIMIT}
                  step={1}
                  value={maxPeople}
                  onChange={(e) => setMaxPeople(Number(e.target.value))}
                  className="block w-full max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  {/* Unlike the detection page's slider this one is not a pure
                      re-filter, and saying so is the honest thing: each extra
                      person is another pose forward pass, so the cap changes
                      what gets computed rather than what gets shown. */}
                  Each person is another forward pass through the pose model, so
                  both of these change the work rather than filtering the result
                  — moving them re-runs.
                </p>
              </div>

              {!session.model.live && (
                <p className="text-xs text-muted-foreground">
                  This pair is still-image only — a 260 MB detector is not a
                  webcam demo. Switch to the nano pair for live mode.
                </p>
              )}
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Skeletons"
          description="One skeleton per person, with each joint's opacity showing how sure the model is of it."
          meta={
            result ? (
              <span className="tabular-nums">
                {people.length} of {result.detected} · detect{" "}
                {Math.round(result.detectMs)} ms · pose{" "}
                {Math.round(result.poseMs)} ms
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Estimating…"
          error={runError}
          empty="Pick a picture with people in it and each one's skeleton is drawn here, joint by joint."
        >
          {result && frame && (
            <PoseView people={people} source={frame.image} />
          )}
        </OutputPanel>
      }
    />
  );
}

/** A stable colour per person, so a skeleton keeps its colour between frames. */
function personColor(i: number): string {
  return OVERLAY_COLORS[i % OVERLAY_COLORS.length];
}

function PoseView({
  people,
  source,
}: {
  people: readonly Person[];
  source: RawImage;
}) {
  const [selected, setSelected] = useState(0);

  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, source);
      people.forEach((person, i) => drawSkeleton(ctx, person, personColor(i)));
    },
    [people, source],
  );

  const person = people[selected] ?? people[0];

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={source.width}
        height={source.height}
        draw={paint}
        label={`Skeletons: ${people.length} ${people.length === 1 ? "person" : "people"}`}
        testId="pose-canvas"
      />

      {people.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No people above the confidence threshold. Drag it down, or try the
          football sample.
        </p>
      ) : (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2" data-testid="people">
            {people.map((p, i) => (
              <Button
                key={i}
                variant="outline"
                size="sm"
                aria-pressed={i === selected}
                onClick={() => setSelected(i)}
              >
                <span
                  aria-hidden
                  className="size-2.5 rounded-full"
                  style={{ background: personColor(i) }}
                />
                Person {i + 1}{" "}
                <span className="tabular-nums opacity-70">
                  {p.score.toFixed(2)}
                </span>
              </Button>
            ))}
          </div>

          {person && (
            <details className="text-xs">
              {/* Available, not dominant: the picture is the answer, and a
                  17-row table above it would bury the thing the page is for. */}
              <summary className="cursor-pointer text-muted-foreground">
                Joint confidences for person {selected + 1}
              </summary>
              <ul
                data-testid="joints"
                className="mt-2 grid grid-cols-2 gap-x-4 font-mono tabular-nums sm:grid-cols-3"
              >
                {person.keypoints.map((k) => (
                  <li
                    key={k.index}
                    data-joint={COCO_KEYPOINTS[k.index] ?? k.index}
                    className={
                      k.score < JOINT_CONFIDENCE
                        ? "text-muted-foreground/60"
                        : "text-muted-foreground"
                    }
                  >
                    {/* The position, not only the confidence. A skeleton offset
                        by the crop's origin looks like a mediocre model rather
                        than a bug in our arithmetic, and these are the numbers
                        that make that difference checkable — by a reader and by
                        the `@slow` spec's anatomical assertion. */}
                    {COCO_KEYPOINTS[k.index] ?? `joint ${k.index}`}{" "}
                    {k.score.toFixed(2)}{" "}
                    <span className="opacity-60">
                      {Math.round(k.x)},{Math.round(k.y)}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          <p className="text-xs text-muted-foreground">
            A heatmap always has a maximum somewhere, so every joint comes back
            with a position whether or not it is in the picture. The faint ones
            are guesses.
          </p>
        </div>
      )}
    </div>
  );
}
