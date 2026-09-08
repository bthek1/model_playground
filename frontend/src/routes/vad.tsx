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
import { Activity, Loader2, Mic, Upload } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { decodeToMono, recordMic } from "@/audio/io";
import { AUDIO_SAMPLES, type AudioSample } from "@/audio/samples";
import {
  DEFAULT_THRESHOLD,
  speechSeconds,
  toSegments,
} from "@/audio/vad/segments";
import { DEFAULT_VAD_MODEL, VAD_MODELS } from "@/audio/vad/types";
import { formatTimestamp } from "@/audio/waveform";
import { VadTimeline } from "@/components/audio/VadTimeline";
import { Waveform } from "@/components/audio/Waveform";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
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
  } = useVad(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [input, setInput] = useState<Float32Array | null>(null);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [preparing, setPreparing] = useState<null | "file" | "mic" | "sample">(
    null,
  );
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = running || preparing !== null;
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

  async function detectFrom(
    source: () => Promise<Float32Array>,
    kind: "file" | "mic" | "sample",
  ) {
    setIoError(null);
    setPreparing(kind);
    try {
      const audio = await source();
      // `run` transfers the buffer to the worker, which detaches it — keep our
      // own copy so the waveform still has samples to draw.
      setInput(audio.slice());
      await run(audio);
    } catch (e) {
      setIoError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreparing(null);
    }
  }

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    void detectFrom(async () => decodeToMono(await file.arrayBuffer()), "file");
  };

  const onSample = (sample: AudioSample) =>
    void detectFrom(async () => {
      const response = await fetch(sample.url);
      if (!response.ok) throw new Error(`${sample.label}: ${response.status}`);
      return decodeToMono(await response.arrayBuffer());
    }, "sample");

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
          disabledHint="Load a detector to analyse a clip."
          controls={
            <>
              <Button
                disabled={!ready || busy}
                onClick={() =>
                  void detectFrom(() => recordMic(RECORD_SECONDS), "mic")
                }
              >
                {preparing === "mic" ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Recording…
                  </>
                ) : (
                  <>
                    <Mic className="size-4" /> Record {RECORD_SECONDS}s
                  </>
                )}
              </Button>

              <Button
                variant="outline"
                disabled={!ready || busy}
                onClick={() => fileRef.current?.click()}
              >
                {preparing === "file" ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Decoding…
                  </>
                ) : (
                  <>
                    <Upload className="size-4" /> Upload audio
                  </>
                )}
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={onFile}
              />
            </>
          }
        >
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Record, upload, or start from a known clip — speech mixed with
              silence shows the detector off best.
            </p>
            <div className="flex flex-wrap gap-2">
              {AUDIO_SAMPLES.map((sample) => (
                <Button
                  key={sample.id}
                  variant="outline"
                  size="sm"
                  disabled={!ready || busy}
                  onClick={() => onSample(sample)}
                >
                  {sample.label}
                </Button>
              ))}
            </div>
          </div>
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
          empty="Run a clip and its speech probability appears here, frame by frame, with the segments it implies."
        >
          {result && input && (
            <div className="space-y-5">
              <div className="space-y-1">
                <Waveform samples={input} className="text-muted-foreground" />
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

