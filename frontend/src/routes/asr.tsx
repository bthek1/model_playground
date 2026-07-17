// Automatic Speech Recognition — real-time, in-browser voice-to-text. A
// Whisper/Moonshine ONNX model runs client-side in a Web Worker (WebGPU, with a
// WASM fallback); the mic is captured continuously and the transcript updates
// live as you speak. Audio is decoded to 16 kHz mono via the Web Audio API. The
// take is retained and visualized as a waveform (live while recording, static
// after) and can be replayed, downloaded, or re-transcribed with another model.
// See docs/plans/in-progress/audio-models-in-browser.md.

import { createFileRoute } from "@tanstack/react-router";
import { AudioLines, Download, Loader2, Mic, Play, Square, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { decodeToMono, play, toWavBlob } from "@/audio/io";
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
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const playbackRef = useRef<AudioContext | null>(null);

  const busy = recording || decoding;
  const shownError = ioError ?? error;

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    void (async () => {
      setIoError(null);
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

  const onPlay = () => {
    if (!clip) return;
    void playbackRef.current?.close(); // restart from the top on replay
    playbackRef.current = play(clip, sampleRate);
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

  // Stop playback when the page unmounts.
  useEffect(() => {
    return () => void playbackRef.current?.close();
  }, []);

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
                    <Button variant="outline" size="sm" onClick={onPlay}>
                      <Play className="size-4" /> Play
                    </Button>
                    <Button variant="outline" size="sm" onClick={onDownload}>
                      <Download className="size-4" /> Download WAV
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!ready || busy}
                      onClick={() => void transcribeClip(clip)}
                    >
                      <AudioLines className="size-4" /> Transcribe clip
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
