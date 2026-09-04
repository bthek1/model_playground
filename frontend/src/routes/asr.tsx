// Automatic Speech Recognition — real-time, in-browser voice-to-text. A
// Whisper/Moonshine ONNX model runs client-side in a Web Worker (WebGPU, with a
// WASM fallback); the mic is captured continuously and the transcript updates
// live as you speak. Audio is decoded to 16 kHz mono via the Web Audio API.
//
// Four-slot page pattern (docs/standards/model-page-pattern.md). The sample
// clips and the retained take live in RUN, not OUTPUT — both are *inputs* you
// can re-run a model over; only the transcript is output. Their markup is in
// `components/audio/AsrTransport.tsx`.

import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Mic, Square, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { decodeToMono } from "@/audio/io";
import { type AudioSample } from "@/audio/samples";
import { ASR_MODELS, DEFAULT_ASR_MODEL } from "@/audio/types";
import { formatTimestamp } from "@/audio/waveform";
import { AudioTake, SampleClips } from "@/components/audio/AsrTransport";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useLiveAsr } from "@/hooks/useLiveAsr";
import {
  useCacheRefresh,
  useModelSelection,
} from "@/model/useModelSelection";

export const Route = createFileRoute("/asr")({
  component: AsrPage,
});

function AsrPage() {
  // Selection and the resume decision survive a refresh; the weights themselves
  // are re-loaded from the browser cache (model/useModelSelection.ts).
  const session = useModelSelection({
    routeKey: "asr",
    models: ASR_MODELS,
    fallback: ASR_MODELS.find((m) => m.id === DEFAULT_ASR_MODEL) ?? ASR_MODELS[0],
  });
  const model = session.model.id;
  const {
    status,
    ready,
    loading,
    backend,
    recording,
    running,
    stream,
    clip,
    sampleRate,
    text,
    chunks,
    error,
    start,
    stop,
    transcribeClip,
    loadProgress,
    loadedInMs,
    load,
    retry,
    cancel,
  } = useLiveAsr(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [decoding, setDecoding] = useState(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [sample, setSample] = useState<AudioSample | null>(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = recording || decoding || loadingSample != null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    void (async () => {
      setIoError(null);
      setSample(null);
      setDecoding(true);
      try {
        await transcribeClip(await decodeToMono(await file.arrayBuffer()));
      } catch (err) {
        setIoError(err instanceof Error ? err.message : String(err));
      } finally {
        setDecoding(false);
      }
    })();
  };

  // Fetch a known clip, decode it, and run the current model on it. The clip's
  // reference transcript stays on screen so the output can be eyeballed against
  // it — an end-to-end model health check that doesn't depend on the mic.
  const runSample = (s: AudioSample) => {
    void (async () => {
      setIoError(null);
      setSample(s);
      setLoadingSample(s.id);
      try {
        const res = await fetch(s.url);
        if (!res.ok) {
          throw new Error(`Couldn't fetch ${s.label} (HTTP ${res.status})`);
        }
        await transcribeClip(await decodeToMono(await res.arrayBuffer()));
      } catch (err) {
        setIoError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingSample(null);
      }
    })();
  };

  return (
    <ModelPage
      icon={Mic}
      title="Automatic Speech Recognition"
      description={
        <>
          Real-time voice-to-text, entirely in your browser — a Whisper/Moonshine
          ONNX model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker.
          Nothing is uploaded to a server. Load a model, press{" "}
          <span className="font-medium">Start listening</span>, and speak.
        </>
      }
      select={
        <ModelPicker
          models={ASR_MODELS}
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
          disabledHint="Load a model to start transcribing."
          controls={
            <>
              {recording ? (
                <Button variant="destructive" onClick={stop}>
                  <Square className="size-4" /> Stop
                </Button>
              ) : (
                <Button disabled={!ready || decoding} onClick={() => void start()}>
                  <Mic className="size-4" /> Start listening
                </Button>
              )}

              <Button
                variant="outline"
                disabled={!ready || busy}
                onClick={() => fileRef.current?.click()}
              >
                {decoding ? (
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
          <SampleClips
            selected={sample}
            loadingId={loadingSample}
            disabled={!ready || busy || running}
            onRun={runSample}
          />

          <AudioTake
            recording={recording}
            stream={stream}
            clip={clip}
            sampleRate={sampleRate}
            canTranscribe={ready && !busy && !running}
            transcribing={running}
            onTranscribe={(c) => void transcribeClip(c)}
          />
        </InputPanel>
      }
      output={
        <OutputPanel
          title={
            <>
              Transcript
              {recording && (
                <span className="flex items-center gap-1.5 text-xs font-normal text-destructive">
                  <span className="relative flex size-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-destructive/70" />
                    <span className="relative inline-flex size-2 rounded-full bg-destructive" />
                  </span>
                  Listening…
                </span>
              )}
            </>
          }
          description={
            recording
              ? "Updating live as you speak"
              : text
                ? "Final transcript"
                : "Press Start listening and speak, or upload a clip"
          }
          // While recording, the growing transcript *is* the feedback — a spinner
          // over it would hide the thing the user is watching.
          running={running && !recording && !text}
          runningLabel="Transcribing…"
          error={runError}
          empty="Your transcript will appear here."
        >
          {(text.trim() || chunks.length > 0 || recording) &&
            (chunks.length > 0 ? (
              // Timestamped segments when the model returns them (Whisper does;
              // `return_timestamps: true`). Times are relative to the whole take.
              <ol className="min-h-16 space-y-1.5">
                {chunks.map((c, i) => (
                  <li
                    key={`${c.timestamp[0]}-${i}`}
                    className="flex gap-3 text-sm leading-relaxed"
                  >
                    <span className="shrink-0 pt-0.5 font-mono text-xs text-muted-foreground tabular-nums">
                      {formatTimestamp(c.timestamp[0])}
                    </span>
                    <span>{c.text.trim()}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="min-h-16 text-sm leading-relaxed whitespace-pre-wrap">
                {text.trim() || (
                  <span className="text-muted-foreground">…</span>
                )}
              </p>
            ))}
        </OutputPanel>
      }
    />
  );
}
