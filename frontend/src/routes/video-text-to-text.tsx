// Video Text to Text — **a frame sampler plus an image model, and the page says
// so.**
//
// That is not a disclaimer, it is what a video-language model *is*, in a tab and
// in PyTorch alike: SmolVLM2 takes a handful of still frames, each encoded into
// its own image tokens, all in the decoder's context together. There is no
// temporal attention here to hide. `/video-classification` carries the same
// honesty requirement for a different reason — there the framing is what stops
// the page teaching something false — and the same rule applies: the sentence is
// next to the result and an E2E spec asserts it.
//
// Four things about this page are consequences of the frame count rather than
// decoration:
//
//  1. **The frame count is the cost dial, and the user can feel it.** Each frame
//     is 64 image tokens at 512px, attended over for every generated token, so
//     four frames against eight is a doubling of the slowest run in the app.
//     It is on a slider and capped at 8 — a 32-frame clip is not a slower page,
//     it is a broken one.
//  2. **Frames multiply the tile problem.** `/image-text-to-text` learned that
//     the processor splits anything over 512px into tiles, each costing its own
//     image tokens *plus* a global view. With N frames that multiplies by N, so
//     the 512 cap is not an optimisation here — it is the difference between a
//     page and a hang.
//  3. **Sampling is uniform, and shown.** The filmstrip in OUTPUT is the frames
//     the model was actually given. A page that samples invisibly makes every
//     wrong answer unattributable.
//  4. **The reverse toggle is the experiment.** Feed the frames backwards and
//     see whether the answer changes. Often it does not, and that is a real
//     property of small video VLMs rather than a gotcha — they are describing a
//     picture, not reading a sequence. Unlike a threshold it genuinely costs an
//     inference, because the model must actually see the other order, and the
//     page says so before the click.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { Film, Loader2, Sparkles, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useVideoPick } from "@/hooks/useVideoPick";
import { useVlm } from "@/hooks/useVlm";
import { supportsShaderF16 } from "@/model/backend";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_FRAMES,
  MAX_FRAMES,
  MIN_FRAMES,
  orderFrames,
} from "@/multimodal/frames";
import {
  DEFAULT_VIDEO_VLM_MODEL,
  VIDEO_VLM_MODELS,
} from "@/multimodal/types";
import { thumbnail, VIDEO_SAMPLES, type SampledFrame } from "@/vision/video";

/**
 * Preset questions. INPUT — tapping one fills the box and runs nothing.
 *
 * Chosen so the reverse toggle has something to bite on: a question about what
 * *happens* can only be answered from the order, while a question about what is
 * *in* the clip cannot tell the two orders apart. Having both on screen is what
 * makes the experiment legible.
 */
const PRESETS = [
  "What happens in this video?",
  "Describe the scene.",
  "What are the people doing?",
  "Does anything change between the first and last frame?",
] as const;

const DEFAULT_QUESTION = PRESETS[0];

export const Route = createFileRoute("/video-text-to-text")({
  component: VideoTextToTextPage,
});

/** What one answer was produced from — captured inside the run. */
interface AskedRun {
  question: string;
  frames: SampledFrame[];
  reversed: boolean;
}

function VideoTextToTextPage() {
  const backendProbe = useBackendProbe({ requireShaderF16: true });
  const session = useModelSelection({
    routeKey: "video-text-to-text",
    models: VIDEO_VLM_MODELS,
    fallback:
      VIDEO_VLM_MODELS.find((m) => m.id === DEFAULT_VIDEO_VLM_MODEL) ??
      VIDEO_VLM_MODELS[0],
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
    partial,
    error,
    load,
    retry,
    cancel,
    run,
  } = useVlm(model);
  useCacheRefresh(session, ready);

  // All three are held state. Typing, dragging the slider and flipping the
  // order change what the next GENERATE sends and cost nothing until then.
  const [question, setQuestion] = useState<string>(DEFAULT_QUESTION);
  const [frameCount, setFrameCount] = useState(DEFAULT_FRAMES);
  const [reversed, setReversed] = useState(false);
  // What the answer on screen was actually produced from.
  const [asked, setAsked] = useState<AskedRun | null>(null);

  const {
    picked,
    sampling,
    error: ioError,
    clearError,
    pickFile,
    pickSample,
    take,
  } = useVideoPick();

  const fileRef = useRef<HTMLInputElement>(null);

  const [hasF16, setHasF16] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    void supportsShaderF16().then((ok) => live && setHasF16(ok));
    return () => {
      live = false;
    };
  }, []);
  const f16Missing = backendProbe === "wasm" && hasF16 === false;

  const busy = running || sampling !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;
  const canRun = ready && !busy && picked != null && question.trim().length > 0;

  const ask = useCallback(async () => {
    const prompt = question.trim();
    if (!prompt) return;
    clearError();
    // Decoding is cached on (clip, frame count), so pressing GENERATE twice
    // costs one decode and two inferences. The empty list is a failed decode —
    // `useVideoPick` has already put the reason in the RUN slot.
    const frames = await take(frameCount);
    if (frames.length === 0) return;
    // The order the *model* sees. The filmstrip keeps the sampled order, so a
    // reversed run is visibly the same frames read the other way rather than a
    // different set of pictures.
    const sent = orderFrames(frames, reversed);
    setAsked({ question: prompt, frames, reversed });
    await run(
      sent.map((f) => f.image),
      prompt,
    );
  }, [clearError, frameCount, question, reversed, run, take]);

  const askCurrent = () => {
    void ask().catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Film}
      title="Video Text to Text"
      description={
        <>
          Ask a question about a clip. The model is a{" "}
          <strong>frame sampler plus an image model</strong>: a handful of stills
          are taken from the video and read together — there is no motion in what
          it sees.
        </>
      }
      labels={{ output: "Answer" }}
      aside={
        result ? (
          <span
            data-testid="generate-ms"
            className="font-mono text-xs text-muted-foreground tabular-nums"
          >
            {(result.ms / 1000).toFixed(1)}s on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <div className="space-y-3">
          <ModelPicker
            models={VIDEO_VLM_MODELS}
            value={model}
            onChange={session.setModel}
            disabled={loading || busy}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
            backend={backendProbe}
          />
          {f16Missing && (
            <p
              data-testid="f16-note"
              className="text-xs leading-snug text-amber-600 dark:text-amber-500"
            >
              This browser has a GPU, but its adapter does not support
              half-precision (<code>shader-f16</code>) in shaders. This model is
              4-bit with f16 activations, so it would download and then fail on
              the first operator — the page disables it rather than charging you
              for that.
            </p>
          )}
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
          disabledHint="Load the model to ask about a clip. You can pick one and write your question first."
          controls={
            <Button disabled={!canRun} onClick={askCurrent} title="Ask the model">
              {busy ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  {sampling ? "Sampling frames…" : "Generating…"}
                </>
              ) : (
                <>
                  <Sparkles className="size-4" /> Generate
                </>
              )}
            </Button>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3 md:overflow-y-auto">
            <div className="flex min-h-40 flex-1 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-dashed bg-muted/20 p-2">
              {picked ? (
                // `preload="none"`: a bundled clip is megabytes from the same
                // host the weights come from, and mounting it with a `src`
                // would fetch it for a user who may never press anything. The
                // sampler builds its own element, so this is preview only.
                <video
                  key={picked.url}
                  src={picked.url}
                  crossOrigin="anonymous"
                  preload="none"
                  controls
                  muted
                  playsInline
                  aria-label={`Clip: ${picked.name}`}
                  className="max-h-64 max-w-full rounded"
                />
              ) : (
                <p className="px-4 text-center text-sm text-balance text-muted-foreground">
                  Pick a sample clip below, or upload one.
                </p>
              )}
            </div>

            {sampling && (
              <p
                data-testid="sampling-progress"
                className="flex items-center gap-2 text-sm text-muted-foreground"
              >
                <Loader2 className="size-4 animate-spin" />
                Sampling frames{" "}
                <span className="tabular-nums">
                  {sampling.done} / {sampling.total}
                </span>
              </p>
            )}

            <div className="space-y-1.5">
              <label className="text-sm font-medium" htmlFor="video-question">
                Question
              </label>
              <textarea
                id="video-question"
                data-testid="question-input"
                rows={2}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask something about the clip…"
                className="w-full min-w-0 resize-y rounded-lg border border-input bg-transparent px-2.5 py-1.5 text-base transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 md:text-sm dark:bg-input/30"
              />
              <div
                className="flex flex-wrap gap-2"
                role="group"
                aria-label="Preset questions"
                data-testid="presets"
              >
                {PRESETS.map((preset) => (
                  <Button
                    key={preset}
                    variant="outline"
                    size="sm"
                    aria-pressed={preset === question}
                    onClick={() => setQuestion(preset)}
                    title="Fill the question box — this does not run the model"
                  >
                    {preset}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="frame-count">
                Frames sampled:{" "}
                <span className="tabular-nums">{frameCount}</span>
              </Label>
              <input
                id="frame-count"
                data-testid="frame-count"
                type="range"
                min={MIN_FRAMES}
                max={MAX_FRAMES}
                step={1}
                value={frameCount}
                disabled={busy}
                onChange={(e) => setFrameCount(Number(e.target.value))}
                className="block w-full max-w-xs"
              />
              <p className="text-xs text-muted-foreground">
                {/* The cost dial, stated as a cost. Each frame is ~64 image
                    tokens the decoder attends over for every word it writes. */}
                Every frame carries its own image tokens, so this is the cost
                dial: {MAX_FRAMES} frames is roughly {MAX_FRAMES / MIN_FRAMES}x
                the work of one. Nothing runs when you move it — it changes what
                the next Generate samples.
              </p>
            </div>

            {/* The experiment. Unlike a threshold this legitimately costs an
                inference, because the model has to actually see the other
                order — so the line says so rather than letting the user guess
                why this one is different. */}
            <div className="space-y-1.5 rounded-lg border bg-muted/20 p-3">
              <label className="flex items-center gap-2 text-sm font-medium">
                <input
                  type="checkbox"
                  data-testid="reverse-toggle"
                  checked={reversed}
                  disabled={busy}
                  onChange={(e) => setReversed(e.target.checked)}
                />
                Feed the frames in reverse
              </label>
              <p className="text-xs text-muted-foreground">
                Flipping this runs nothing, but the next Generate is a real
                second inference — the model has to see the other order. If the
                answer does not change, the model is describing a picture rather
                than reading a sequence. That is the usual result at this size,
                and it is the point.
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">
                Sample clips — short, and hosted alongside the weights.
              </p>
              <div className="flex flex-wrap gap-2">
                {VIDEO_SAMPLES.map((sample) => (
                  <Button
                    key={sample.id}
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    title={sample.hint}
                    onClick={() => pickSample(sample)}
                  >
                    {sample.label}
                  </Button>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={busy}
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
          title="Answer"
          description="What the model wrote, from the sampled frames and your question."
          meta={
            result ? (
              <span>
                {result.tokens} tokens · {(result.encodeMs / 1000).toFixed(1)}s
                encoding
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Generating…"
          error={runError}
          empty="Pick a clip, ask a question, and the frames the model saw appear here beside its answer."
        >
          {(partial != null || result != null) && (
            <AnswerView partial={partial} text={result?.text} asked={asked} />
          )}
        </OutputPanel>
      }
    />
  );
}

function AnswerView({
  partial,
  text,
  asked,
}: {
  partial: { stage: "encoding" } | { stage: "generating"; text: string } | null;
  text?: string;
  asked: AskedRun | null;
}) {
  if (partial?.stage === "encoding") {
    return (
      <p
        data-testid="answer-encoding"
        className="flex items-center gap-2 text-sm text-muted-foreground"
      >
        <Loader2 className="size-4 animate-spin" />
        Encoding {asked?.frames.length ?? 0} frames — every one of them is read
        before the first word appears.
      </p>
    );
  }

  const body = partial?.stage === "generating" ? partial.text : text;
  if (body == null) return null;

  return (
    <div className="space-y-3">
      {asked && (
        <p className="text-xs text-muted-foreground" data-testid="answer-asked">
          Asked: {asked.question} · {asked.frames.length} frame
          {asked.frames.length === 1 ? "" : "s"}
          {asked.reversed ? ", in reverse" : ", in order"}
        </p>
      )}
      {body ? (
        // The testid goes on a wrapper: `Markdown` takes only
        // `{children, className}` and silently drops everything else.
        <div data-testid="answer-text">
          <Markdown>{body}</Markdown>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          The model generated nothing. Try a more specific question, or more
          frames.
        </p>
      )}

      {asked && asked.frames.length > 0 && (
        // **The frames the model was actually given.** A page that samples
        // invisibly makes every wrong answer unattributable, so this is the
        // input as the model received it — not the clip currently picked.
        <div className="space-y-1.5">
          <p className="text-xs text-muted-foreground">
            The frames the model saw
            {asked.reversed && " — read right to left, as sent"}.
          </p>
          <ul data-testid="filmstrip" className="flex gap-1 overflow-x-auto pb-1">
            {asked.frames.map((frame) => (
              <li key={frame.time} className="shrink-0">
                <Thumb frame={frame} />
                <span className="block text-center font-mono text-[0.6rem] text-muted-foreground tabular-nums">
                  {frame.time.toFixed(1)}s
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {partial == null && (
        <p
          data-testid="sampler-note"
          className="text-xs text-amber-600 dark:text-amber-500"
        >
          {/* Next to the result, not in a footnote — the same requirement
              `/video-classification` carries, for the same reason. */}
          This is a <span className="font-medium">frame sampler</span> in front
          of an image model. The clip above was reduced to the stills shown here
          and nothing between them reached the model, so it cannot tell opening a
          door from closing one. That is what a video-language model of this size
          is, not a limitation of running it in a browser.
        </p>
      )}
    </div>
  );
}

/**
 * One filmstrip thumbnail.
 *
 * Memoised on the **frame**, not on the mount: `toCanvas().toDataURL()`
 * re-encodes a JPEG and eight of those per token of a streaming answer is real
 * jank, but the list is keyed by timestamp — and two clips sampled at the same
 * count produce the same timestamps, so a mount-once cache would show the
 * previous clip's pictures under the new clip's answer.
 */
function Thumb({ frame }: { frame: SampledFrame }) {
  const src = useMemo(() => thumbnail(frame.image), [frame.image]);
  return src ? (
    <img
      src={src}
      alt={`Frame at ${frame.time.toFixed(1)}s`}
      className="h-14 rounded"
    />
  ) : (
    <span className="block h-14 w-20 rounded bg-muted" />
  );
}
