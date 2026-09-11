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
import { AudioLines, Loader2, Mic, Square, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

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
  // The selected model survives a refresh; the weights do not, and are not
  // re-fetched until the LOAD button is pressed (model/useModelSelection.ts).
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
  } = useLiveAsr(model);
  useCacheRefresh(session, ready);

  const [decoding, setDecoding] = useState(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [sample, setSample] = useState<AudioSample | null>(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  // The clip Transcribe will run on: a sample, an upload, or the last take.
  // Held rather than consumed, so the same audio can be pushed through a second
  // model without re-fetching or re-recording it.
  const [pending, setPending] = useState<Float32Array | null>(null);

  // A finished take becomes the pending clip. Input-to-input, not a run: the
  // live loop that produced it has already stopped, and Transcribe is what
  // re-runs it.
  useEffect(() => {
    if (clip) setPending(clip);
  }, [clip]);

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
        setPending(await decodeToMono(await file.arrayBuffer()));
      } catch (err) {
        setIoError(err instanceof Error ? err.message : String(err));
      } finally {
        setDecoding(false);
      }
    })();
  };

  // Fetch a known clip and decode it into the input. It is **not** transcribed
  // here: the clip's reference transcript is on screen so the output can be
  // eyeballed against it, and the user decides when to spend a model on it.
  const selectSample = (s: AudioSample) => {
    void (async () => {
      setIoError(null);
      setSample(s);
      setLoadingSample(s.id);
      try {
        const res = await fetch(s.url);
        if (!res.ok) {
          throw new Error(`Couldn't fetch ${s.label} (HTTP ${res.status})`);
        }
        setPending(await decodeToMono(await res.arrayBuffer()));
      } catch (err) {
        setIoError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingSample(null);
      }
    })();
  };

  // The one-shot trigger, beside the live loop's Start/Stop. Both are explicit
  // presses; nothing here runs off a decode.
  const transcribeCurrent = () => {
    if (!pending) return;
    setIoError(null);
    void transcribeClip(pending).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
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
          disabledHint="Load a model to transcribe. You can pick a clip first."
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

              {/* Not gated on a model: decoding a file is input, and the
                  clip sits in INPUT until Transcribe is pressed. */}
              <Button
                variant="outline"
                disabled={busy}
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

              <Button
                disabled={!ready || busy || running || !pending}
                onClick={transcribeCurrent}
              >
                {running && !recording ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Transcribing…
                  </>
                ) : (
                  <>
                    <AudioLines className="size-4" /> Transcribe
                  </>
                )}
              </Button>
            </>
          }
        >
          {/* Picking a clip needs no model, so it is not gated on one. */}
          <SampleClips
            selected={sample}
            loadingId={loadingSample}
            disabled={busy || running}
            onSelect={selectSample}
          />

          <AudioTake
            recording={recording}
            stream={stream}
            clip={clip}
            sampleRate={sampleRate}
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
                : "Press Start listening and speak, or load a clip and press Transcribe"
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
