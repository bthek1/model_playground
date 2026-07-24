// Automatic Speech Recognition — real-time, in-browser voice-to-text. A
// Whisper/Moonshine ONNX model runs client-side in a Web Worker (WebGPU, with a
// WASM fallback); the mic is captured continuously and the transcript updates
// live as you speak. Audio is decoded to 16 kHz mono via the Web Audio API. The
// take is retained and visualized as a waveform (live while recording, static
// after) and can be replayed, downloaded, or re-transcribed with another model.
// See docs/plans/in-progress/audio-models-in-browser.md.

import { createFileRoute } from "@tanstack/react-router";
import {
  AudioLines,
  Download,
  FlaskConical,
  Loader2,
  Mic,
  Pause,
  Play,
  Square,
  Upload,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { decodeToMono, play, toWavBlob } from "@/audio/io";
import { AUDIO_SAMPLES, type AudioSample } from "@/audio/samples";
import { ASR_MODELS, DEFAULT_ASR_MODEL } from "@/audio/types";
import { formatDuration } from "@/audio/waveform";
import { ModelStatus } from "@/components/audio/ModelStatus";
import { LiveWaveform, Waveform } from "@/components/audio/Waveform";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useLiveAsr } from "@/hooks/useLiveAsr";

export const Route = createFileRoute("/asr")({
  component: AsrPage,
});

function AsrPage() {
  const [model, setModel] = useState(DEFAULT_ASR_MODEL);
  const {
    ready,
    loading,
    progress,
    backend,
    recording,
    running,
    stream,
    clip,
    sampleRate,
    text,
    error,
    start,
    stop,
    transcribeClip,
  } = useLiveAsr(model);

  const [decoding, setDecoding] = useState(false);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [sample, setSample] = useState<AudioSample | null>(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const [playState, setPlayState] = useState<"idle" | "playing" | "paused">("idle");
  const fileRef = useRef<HTMLInputElement>(null);
  const playbackRef = useRef<AudioContext | null>(null);

  const busy = recording || decoding || loadingSample != null;
  const shownError = ioError ?? error;

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
        if (!res.ok) throw new Error(`Couldn't fetch ${s.label} (HTTP ${res.status})`);
        await transcribeClip(await decodeToMono(await res.arrayBuffer()));
      } catch (err) {
        setIoError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingSample(null);
      }
    })();
  };

  // Play ⇄ Pause/Resume via the AudioContext transport. A fresh play() starts
  // from the top; suspend()/resume() pause/resume in place; the clip's natural
  // end resets to idle.
  const onPlayPause = () => {
    if (!clip) return;
    const ctx = playbackRef.current;
    if (playState === "playing" && ctx) {
      void ctx.suspend();
      setPlayState("paused");
    } else if (playState === "paused" && ctx) {
      void ctx.resume();
      setPlayState("playing");
    } else {
      const started = play(clip, sampleRate, () => {
        if (playbackRef.current === started) {
          playbackRef.current = null;
          setPlayState("idle");
        }
      });
      playbackRef.current = started;
      setPlayState("playing");
    }
  };

  const stopPlayback = () => {
    const ctx = playbackRef.current;
    playbackRef.current = null;
    setPlayState("idle");
    if (ctx && ctx.state !== "closed") void ctx.close();
  };

  const onDownload = () => {
    if (!clip) return;
    const url = URL.createObjectURL(toWavBlob(clip, sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = "recording.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  // A new take (or unmount) invalidates any in-progress playback.
  useEffect(() => {
    setPlayState("idle");
    const ref = playbackRef;
    return () => {
      const ctx = ref.current;
      ref.current = null;
      if (ctx && ctx.state !== "closed") void ctx.close();
    };
  }, [clip]);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Mic className="size-6" /> Automatic Speech Recognition
        </h1>
        <p className="text-sm text-muted-foreground">
          Real-time voice-to-text, entirely in your browser — a Whisper/Moonshine
          ONNX model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker.
          Nothing is uploaded to a server. Pick a model, press{" "}
          <span className="font-medium">Start listening</span>, and speak.
        </p>
      </div>

      {/* Model picker */}
      <div className="flex flex-wrap gap-2">
        {ASR_MODELS.map((m) => (
          <Button
            key={m.id}
            variant={m.id === model ? "default" : "outline"}
            size="sm"
            disabled={busy}
            onClick={() => setModel(m.id)}
            title={m.hint}
          >
            {m.label}
          </Button>
        ))}
      </div>

      {/* Model load status */}
      <ModelStatus loading={loading} ready={ready} backend={backend} progress={progress} />

      {/* Transport controls */}
      <div className="flex flex-wrap items-center gap-2">
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
      </div>

      {shownError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {shownError}
        </div>
      )}

      {/* Sample clips — a model health check with known-good audio. Run one and
          compare the model's output against the reference transcript. */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <FlaskConical className="size-4" /> Test with a sample clip
          </CardTitle>
          <CardDescription>
            Not sure the model works? Run a known clip and compare the output
            below with the reference. Garbage here means the model, not your mic.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {AUDIO_SAMPLES.map((s) => (
              <Button
                key={s.id}
                variant={sample?.id === s.id ? "default" : "outline"}
                size="sm"
                disabled={!ready || busy || running}
                onClick={() => runSample(s)}
                title={s.hint}
              >
                {loadingSample === s.id ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <FlaskConical className="size-4" />
                )}
                {s.label}
              </Button>
            ))}
          </div>
          {sample && (
            <div className="rounded-md border bg-muted/30 px-3 py-2 text-sm">
              <p className="text-xs text-muted-foreground">{sample.hint}</p>
              {sample.reference ? (
                <p className="mt-1">
                  <span className="font-medium text-muted-foreground">
                    Reference:{" "}
                  </span>
                  {sample.reference}
                </p>
              ) : (
                <p className="mt-1 text-muted-foreground">
                  Long-form clip — no fixed reference; check that the transcript
                  reads as coherent English with sensible timestamps.
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Audio — live waveform while recording, the retained take after */}
      {(recording || clip) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AudioLines className="size-4" /> Audio
              {clip && !recording && (
                <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">
                  {formatDuration(clip.length, sampleRate)} · 16 kHz mono
                </span>
              )}
            </CardTitle>
            <CardDescription>
              {recording
                ? "Live microphone signal"
                : "Your take — replay it, download it, or run another model on it"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {recording && stream ? (
              <LiveWaveform stream={stream} />
            ) : (
              clip && (
                <>
                  <Waveform samples={clip} />
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={onPlayPause}>
                      {playState === "playing" ? (
                        <>
                          <Pause className="size-4" /> Pause
                        </>
                      ) : (
                        <>
                          <Play className="size-4" />{" "}
                          {playState === "paused" ? "Resume" : "Play"}
                        </>
                      )}
                    </Button>
                    {playState !== "idle" && (
                      <Button variant="outline" size="sm" onClick={stopPlayback}>
                        <Square className="size-4" /> Stop
                      </Button>
                    )}
                    <Button variant="outline" size="sm" onClick={onDownload}>
                      <Download className="size-4" /> Download WAV
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!ready || busy || running}
                      onClick={() => void transcribeClip(clip)}
                    >
                      {running ? (
                        <>
                          <Loader2 className="size-4 animate-spin" /> Transcribing…
                        </>
                      ) : (
                        <>
                          <AudioLines className="size-4" /> Transcribe clip
                        </>
                      )}
                    </Button>
                  </div>
                </>
              )
            )}
          </CardContent>
        </Card>
      )}

      {/* Live transcript */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
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
          </CardTitle>
          <CardDescription>
            {recording
              ? "Updating live as you speak"
              : text
                ? "Final transcript"
                : "Press Start listening and speak, or upload a clip"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="min-h-16 text-sm leading-relaxed whitespace-pre-wrap">
            {text.trim() || (
              <span className="text-muted-foreground">
                {recording ? "…" : "Your transcript will appear here."}
              </span>
            )}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
