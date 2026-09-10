// Video Classification — **a frame-level baseline, and labelled as one.**
//
// None of the four real video transformers (VideoMAE, TimeSformer, X-CLIP,
// V-JEPA 2) has an ONNX export: they attend across time as well as space, and
// nobody has shipped a browser-runnable one. What *can* be built honestly is the
// control experiment — sample frames, score each with CLIP zero-shot, pool over
// a sliding window — and the page's thesis is the question *does motion actually
// matter*, answered from the other direction: this is what you get when the
// model can only see single frames.
//
// **The labelling is not decoration, it is the correctness requirement.**
// Shipping this as "Video Classification" without the framing teaches something
// false, so the limitation is stated next to the result rather than in a
// footnote, and an E2E spec asserts the page copy. That is unusual and
// deliberate.
//
// Two efficiencies make it usable at all, and both are "encode once, decode
// many" — the same idea, applied twice on one page:
//
//  1. The **label embeddings are constant across every frame**, so the
//     zero-shot worker's text cache computes them once per label edit rather
//     than once per frame. On a 60-frame clip that is the difference between
//     seconds and a minute.
//  2. The **window slider re-derives**. Pooling is pure (`pool.ts`) over scores
//     already in hand, so changing it re-draws the chart without re-scoring the
//     clip — which here would be N CLIP passes.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { Clapperboard, Loader2, Upload, X } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { PhraseList } from "@/components/vision/PhraseList";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  useVideoClassifier,
  type ClipScores,
} from "@/hooks/useVideoClassifier";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { pooledVerdict, slidingMean } from "@/vision/pool";
import {
  applyTemplate,
  DEFAULT_ZERO_SHOT_MODEL,
  TEMPLATES,
  ZERO_SHOT_MODELS,
} from "@/vision/zeroShot";
import {
  DEFAULT_SAMPLE_FPS,
  MAX_FRAMES,
  VIDEO_SAMPLES,
  type VideoSample,
} from "@/vision/video";

const EChart = lazy(() => import("@/components/charts/EChart"));

export const Route = createFileRoute("/video-classification")({
  component: VideoClassificationPage,
});

/** The default window, in frames. Wide enough to damp a single-frame spike. */
const DEFAULT_WINDOW = 5;
const MAX_WINDOW = 21;

function VideoClassificationPage() {
  const session = useModelSelection({
    routeKey: "video-classification",
    models: ZERO_SHOT_MODELS,
    fallback:
      ZERO_SHOT_MODELS.find((m) => m.id === DEFAULT_ZERO_SHOT_MODEL) ??
      ZERO_SHOT_MODELS[0],
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
    clipProgress,
    load,
    retry,
    cancel,
    run,
    stop,
  } = useVideoClassifier(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [sample, setSample] = useState<VideoSample>(VIDEO_SAMPLES[0]);
  const [labels, setLabels] = useState<string[]>([...VIDEO_SAMPLES[0].labels]);
  const [fps, setFps] = useState(DEFAULT_SAMPLE_FPS);
  const [window, setWindow] = useState(DEFAULT_WINDOW);
  const [upload, setUpload] = useState<{ url: string; name: string } | null>(
    null,
  );
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Exactly one object URL alive at a time, and none after unmount — the same
  // rule `useImagePick` enforces for stills, and a video is a much bigger
  // buffer to leave pinned.
  const ownedUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    },
    [],
  );

  const pickSample = (next: VideoSample) => {
    if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    ownedUrl.current = null;
    setUpload(null);
    setSample(next);
    // The sample's own label set: a zero-shot score is relative to the labels
    // given, so a clip with no plausible distractor demonstrates nothing.
    setLabels([...next.labels]);
  };

  const pickFile = (file: File | undefined) => {
    if (!file) return;
    if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    const url = URL.createObjectURL(file);
    ownedUrl.current = url;
    setUpload({ url, name: file.name });
    setIoError(null);
  };

  const source = upload ?? { url: sample.url, name: sample.label };

  const classify = () => {
    setIoError(null);
    void run(source.url, labels, TEMPLATES.photo, { fps, maxFrames: MAX_FRAMES })
      .catch((e: unknown) => {
        // A clip that will not decode is an *input* failure and belongs in RUN,
        // beside the file that caused it — the model is fine.
        setIoError(e instanceof Error ? e.message : String(e));
      });
  };

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // Pure, on the main thread, and the whole point: moving the window re-derives
  // the chart from scores already in hand. Re-running would be N CLIP passes.
  const series = useMemo(
    () => (result?.frames ?? []).map((f) => f.scores),
    [result],
  );
  const pooled = useMemo(() => slidingMean(series, window), [series, window]);
  const verdict = useMemo(() => pooledVerdict(series), [series]);

  return (
    <ModelPage
      icon={Clapperboard}
      title="Video Classification"
      description={
        <>
          A <strong>frame-level baseline</strong>: frames are sampled from the
          clip and scored one at a time by CLIP, then pooled over time. The model
          never sees motion — that is the experiment.
        </>
      }
      labels={{ output: "Scores over time" }}
      select={
        <ModelPicker
          models={ZERO_SHOT_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || running}
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
          disabled={running}
        />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to score a clip. You can pick one and write the labels first."
          controls={
            running ? (
              <Button variant="outline" onClick={stop}>
                <X className="size-4" /> Stop
              </Button>
            ) : (
              <Button
                disabled={!ready || labels.length === 0}
                onClick={classify}
              >
                <Clapperboard className="size-4" /> Score the clip
              </Button>
            )
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3 md:overflow-y-auto">
            <div className="flex min-h-40 flex-1 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-dashed bg-muted/20 p-2">
              {/* `preload="none"` is the §1.2 rule applied to the *input*, not
                  just the weights: a bundled clip is 2 MB from the same origin
                  the models come from, and mounting it with a `src` fetches it
                  on arrival for a user who may never press anything. Nothing
                  moves until they hit play or score the clip — and the sampler
                  builds its own element anyway, so this one is preview only. */}
              <video
                key={source.url}
                src={source.url}
                crossOrigin="anonymous"
                preload="none"
                controls
                muted
                playsInline
                aria-label={`Clip: ${source.name}`}
                className="max-h-64 max-w-full rounded"
              />
            </div>

            {clipProgress && (
              <p
                data-testid="clip-progress"
                className="flex items-center gap-2 text-sm text-muted-foreground"
              >
                <Loader2 className="size-4 animate-spin" />
                {clipProgress.phase === "sampling"
                  ? "Sampling frames"
                  : "Scoring frames"}{" "}
                <span className="tabular-nums">
                  {clipProgress.done}
                  {clipProgress.total > 0 && ` / ${clipProgress.total}`}
                </span>
              </p>
            )}

            <PhraseList
              id="new-video-label"
              title="Labels"
              items={labels}
              onChange={setLabels}
              placeholder="a car chase"
              disabled={running}
              emptyHint="Add at least one label to score against."
              hint={
                labels[0]
                  ? `Sent to the model as “${applyTemplate(TEMPLATES.photo, labels[0])}”, once per frame.`
                  : "Each label is templated before it reaches the text tower."
              }
            />

            <div className="space-y-1.5">
              <Label htmlFor="sample-fps">
                Frames per second: <span className="tabular-nums">{fps}</span>
              </Label>
              <input
                id="sample-fps"
                type="range"
                min={1}
                max={4}
                step={1}
                value={fps}
                disabled={running}
                onChange={(e) => setFps(Number(e.target.value))}
                className="block w-full max-w-xs"
              />
              <p className="text-xs text-muted-foreground">
                Every frame is a CLIP vision pass, so this is the cost dial.
                Capped at {MAX_FRAMES} frames however long the clip is —
                changing it re-samples and re-scores.
              </p>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="pool-window">
                Pooling window:{" "}
                <span className="tabular-nums">{window} frames</span>
              </Label>
              <input
                id="pool-window"
                type="range"
                min={1}
                max={MAX_WINDOW}
                step={2}
                value={window}
                onChange={(e) => setWindow(Number(e.target.value))}
                className="block w-full max-w-xs"
              />
              <p className="text-xs text-muted-foreground">
                Nothing re-runs — the chart is re-derived from the scores already
                in hand.
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">
                Sample clips — each comes with labels worth arguing about.
              </p>
              <div className="flex flex-wrap gap-2">
                {VIDEO_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    variant="outline"
                    size="sm"
                    disabled={running}
                    title={s.hint}
                    onClick={() => pickSample(s)}
                  >
                    {s.label}
                  </Button>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={running}
                  onClick={() => fileRef.current?.click()}
                >
                  <Upload className="size-4" /> Upload a clip
                </Button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  aria-label="Upload a video"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    pickFile(file);
                  }}
                />
              </div>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Scores over time"
          description="One line per label, per sampled frame, smoothed by the pooling window."
          meta={
            result ? (
              <span className="tabular-nums">
                {result.frames.length} frames · {result.duration.toFixed(1)}s
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Scoring the clip…"
          error={runError}
          empty="Pick a clip and write some labels. Each sampled frame is scored, and the scores are charted over time."
        >
          {result && result.frames.length > 0 && (
            <ClipView
              result={result}
              pooled={pooled}
              verdict={verdict}
              window={window}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function ClipView({
  result,
  pooled,
  verdict,
  window,
}: {
  result: ClipScores;
  pooled: number[][];
  verdict: ReturnType<typeof pooledVerdict>;
  window: number;
}) {
  const times = result.frames.map((f) => f.time.toFixed(1));

  const option = useMemo(
    () => ({
      grid: { left: 44, right: 12, top: 28, bottom: 32 },
      tooltip: { trigger: "axis" as const },
      legend: { type: "scroll" as const, top: 0 },
      xAxis: {
        type: "category" as const,
        data: times,
        name: "s",
        nameLocation: "end" as const,
      },
      yAxis: { type: "value" as const, min: 0, max: 1 },
      series: result.labels.map((label, l) => ({
        name: label,
        type: "line" as const,
        smooth: true,
        showSymbol: false,
        data: pooled.map((row) => Number((row[l] ?? 0).toFixed(4))),
      })),
    }),
    [result.labels, pooled, times],
  );

  return (
    <div className="space-y-4">
      <p data-testid="pooled-winner" className="text-sm">
        Pooled over the whole clip:{" "}
        <span className="font-medium">
          {result.labels[verdict.index] ?? "—"}
        </span>{" "}
        <span className="font-mono tabular-nums">
          {verdict.score.toFixed(3)}
        </span>
        <span className="text-muted-foreground">
          {" "}
          · window {window} frame{window === 1 ? "" : "s"}
        </span>
      </p>

      <div className="h-56 w-full">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Loading chart…
            </div>
          }
        >
          <EChart option={option} />
        </Suspense>
      </div>

      {/* The filmstrip: a spike in the chart has to be traceable to a frame, or
          the chart is just a shape. */}
      <div className="space-y-1.5">
        <p className="text-xs text-muted-foreground">
          The frames the model actually saw
          {result.capped && " — capped, so this is not the whole clip"}.
        </p>
        <ul
          data-testid="filmstrip"
          className="flex gap-1 overflow-x-auto pb-1"
        >
          {result.frames.map((frame) => (
            <li key={frame.time} className="shrink-0">
              {frame.thumb ? (
                <img
                  src={frame.thumb}
                  alt={`Frame at ${frame.time.toFixed(1)}s`}
                  className="h-14 rounded"
                />
              ) : (
                <span className="block h-14 w-20 rounded bg-muted" />
              )}
              <span className="block text-center font-mono text-[0.6rem] text-muted-foreground tabular-nums">
                {frame.time.toFixed(1)}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* **Next to the result, not in a footnote.** Shipping this as "Video
          Classification" without the framing teaches something false, and an
          E2E spec asserts this copy for that reason. */}
      <p
        data-testid="baseline-note"
        className="text-xs text-amber-600 dark:text-amber-500"
      >
        This is a <span className="font-medium">frame-level baseline</span>, not
        video classification. CLIP scores each frame on its own and never sees
        motion, so it cannot tell opening a door from closing one, or a person
        standing up from sitting down. A real video transformer — VideoMAE,
        TimeSformer, X-CLIP, V-JEPA 2 — attends across time as well as space and
        would separate exactly those cases. None of them has an ONNX export, so
        none of them runs in a tab.
      </p>

      <p className="text-xs text-muted-foreground">
        The scores are also relative to the labels you gave: the model picks the
        best match from your list and has no way to say “none of these”.
      </p>
    </div>
  );
}
