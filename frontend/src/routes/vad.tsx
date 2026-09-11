// Voice Activity Detection — Silero VAD v5 in the browser, on the CPU.
//
// The output of this task is a *timeline*, not a label, which is what makes it
// worth a page: the model returns a speech probability every 32 ms, drawn under
// the waveform on the same time axis, and the threshold that turns those numbers
// into segments is the user's to drag. Dragging re-derives segments from the
// probabilities already in hand (`audio/vad/segments.ts`) — it never re-runs the
// model, so the boundaries move with the pointer.
//
// Two things set this route apart from the other audio pages:
//   * It runs on **WASM by design**, not as a GPU fallback. A 576-sample window
//     scored in 0.3 ms leaves nothing for a GPU to win, and Silero's LSTM/`If`
//     graph isn't covered by ORT's WebGPU provider anyway — see
//     `audio/vad/session.ts`.
//   * The catalogue contains a model with **no weights at all**. The energy
//     baseline is there to be beaten, and it makes the page useful before any
//     download.
//
// See docs/guides/adding-a-model.md §9 (a bare ONNX graph).

import { createFileRoute } from "@tanstack/react-router";
import { Activity, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";

import { AUDIO_SAMPLES } from "@/audio/samples";
import {
  DEFAULT_THRESHOLD,
  speechSeconds,
  toSegments,
} from "@/audio/vad/segments";
import { DEFAULT_VAD_MODEL, VAD_MODELS } from "@/audio/vad/types";
import { formatTimestamp } from "@/audio/waveform";
import { AudioSourcePanel } from "@/components/audio/AudioSourcePanel";
import { VadTimeline } from "@/components/audio/VadTimeline";
import { Waveform } from "@/components/audio/Waveform";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAudioPick } from "@/hooks/useAudioPick";
import { useVad } from "@/hooks/useVad";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";

export const Route = createFileRoute("/vad")({
  component: VadPage,
});

const RECORD_SECONDS = 6;

function VadPage() {
  // Like audio-to-audio, the weights come through ONNX Runtime's own fetch and
  // land in the HTTP cache rather than the Cache Storage bucket the probe reads,
  // so this route keeps the persisted *selection* but never reports a model as
  // cached and never auto-resumes. One extra click beats bandwidth spent unasked.
  const session = useModelSelection({
    routeKey: "vad",
    models: VAD_MODELS,
    fallback:
      VAD_MODELS.find((m) => m.id === DEFAULT_VAD_MODEL) ?? VAD_MODELS[0],
  });
  const model = session.model.id;
  const {
    status,
    loading,
    ready,
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
  } = useVad(model);
  useCacheRefresh(session, ready);

  const input = useAudioPick();
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  // The clip the *current* probabilities belong to, captured at run time. Not
  // the same thing as the clip currently held in INPUT: picking a new one must
  // not redraw the waveform under an older result's timeline.
  const [scored, setScored] = useState<Float32Array | null>(null);

  const busy = running || input.preparing !== null;
  // Each error in the slot that produced it (§4): capture failures in RUN, a
  // load failure in LOAD, a detection failure in OUTPUT.
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // The threshold drag lands here, not in the worker. `toSegments` over a few
  // hundred floats is microseconds; re-running the model would be ~100 ms and
  // would make the boundaries lag the pointer.
  const segments = useMemo(
    () =>
      result
        ? toSegments(result.probabilities, {
            threshold,
            frameSamples: result.frameSamples,
            sampleRate: result.sampleRate,
          })
        : [],
    [result, threshold],
  );

  // The only trigger. `take()` hands over a copy because the worker detaches
  // the buffer it is given; the clip itself stays in INPUT, so the same audio
  // can be re-scored against a different checkpoint without re-uploading it.
  const detectCurrent = () => {
    const audio = input.take();
    if (!audio) return;
    input.clearError();
    setScored(audio.slice());
    void run(audio).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Activity}
      title="Voice Activity Detection"
      description={
        <>
          Find the speech in a recording. Silero VAD scores every 32&nbsp;ms of
          audio in a Web Worker on your CPU — nothing is uploaded — and you set
          the threshold that turns those scores into segments.
        </>
      }
      labels={{ output: "Speech timeline" }}
      select={
        <ModelPicker
          models={VAD_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={busy || loading}
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
          error={input.error}
          disabledHint="Load a detector to analyse a clip. You can pick one first."
          controls={
            <Button
              disabled={!ready || busy || !input.clip}
              onClick={detectCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Detecting…
                </>
              ) : (
                <>
                  <Activity className="size-4" /> Detect speech
                </>
              )}
            </Button>
          }
        >
          <AudioSourcePanel
            clip={input.clip}
            preparing={input.preparing}
            samples={AUDIO_SAMPLES}
            sampleHint="Samples — speech mixed with silence shows the detector off best."
            onFile={input.pickFile}
            onSample={input.pickSample}
            onRecord={input.record}
            recordSeconds={RECORD_SECONDS}
            busy={busy}
          />
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Speech timeline"
          description={
            result
              ? `${result.probabilities.length} frames of ${Math.round((result.frameSamples / result.sampleRate) * 1000)} ms.`
              : undefined
          }
          meta={
            result ? (
              <span className="tabular-nums">
                {segments.length} segment{segments.length === 1 ? "" : "s"} ·{" "}
                {speechSeconds(segments).toFixed(1)}s speech of{" "}
                {(result.samples / result.sampleRate).toFixed(1)}s
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Detecting…"
          error={runError}
          empty="Pick a clip, then press Detect speech — its probability appears here frame by frame, with the segments it implies."
        >
          {result && scored && (
            <div className="space-y-5">
              <div className="space-y-1">
                <Waveform samples={scored} className="text-muted-foreground" />
                <VadTimeline
                  probabilities={result.probabilities}
                  threshold={threshold}
                  className="block text-primary"
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="threshold">
                  Threshold:{" "}
                  <span className="tabular-nums">{threshold.toFixed(2)}</span>
                </Label>
                <input
                  id="threshold"
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="block w-full max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Frames above the line count as speech. Nothing re-runs — the
                  segments below are re-derived from the same scores.
                </p>
              </div>

              {segments.length > 0 ? (
                <ul className="space-y-1 text-sm">
                  {segments.map((s) => (
                    <li
                      key={`${s.start}-${s.end}`}
                      className="flex items-center gap-3 tabular-nums"
                    >
                      <span className="text-muted-foreground">
                        {formatTimestamp(s.start)} → {formatTimestamp(s.end)}
                      </span>
                      <span className="text-xs text-muted-foreground/80">
                        {(s.end - s.start).toFixed(1)}s
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No speech above this threshold — drag it down.
                </p>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

