// Audio to Audio — DeepFilterNet3 speech enhancement, entirely in the browser.
// Upload or record a noisy clip, denoise it on the GPU (WebGPU) or CPU (WASM),
// then A/B the before and after and download the result.
//
// Two things set this route apart from the other audio pages:
//   * 48 kHz, not 16 kHz. DeepFilterNet3 is natively 48 kHz — `decodeToMono` and
//     `recordMic` are both told so explicitly. Do not let a 16 kHz assumption
//     leak in from the ASR code; resampling to 16 kHz would throw away exactly
//     the band the model is trained to repair.
//   * There is no Transformers.js pipeline behind it. The DSP is ours
//     (src/audio/enhance/), validated against the reference implementation.
//
// See docs/plans/completed/audio-to-audio-deepfilternet.md.

import { createFileRoute } from "@tanstack/react-router";
import { AudioWaveform, Download, Loader2, Mic, Play, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  DEFAULT_ENHANCE_MODEL,
  ENHANCE_MODELS,
  SAMPLE_RATE,
} from "@/audio/enhance/types";
import { decodeToMono, play, recordMic, toWavBlob } from "@/audio/io";
import { ModelPicker } from "@/components/audio/ModelPicker";
import { ModelStatus } from "@/components/audio/ModelStatus";
import { Waveform } from "@/components/audio/Waveform";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useEnhance } from "@/hooks/useEnhance";

export const Route = createFileRoute("/audio-to-audio")({
  component: AudioToAudioPage,
});

const RECORD_SECONDS = 6;

function AudioToAudioPage() {
  const [model, setModel] = useState(DEFAULT_ENHANCE_MODEL);
  const { idle, loading, ready, progress, backend, running, result, error, load, retry, run } =
    useEnhance(model);

  const [input, setInput] = useState<Float32Array | null>(null);
  const [preparing, setPreparing] = useState<null | "file" | "mic">(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const playbackRef = useRef<AudioContext | null>(null);

  // Stop playback when the page goes away — an AudioContext outlives the route.
  useEffect(() => () => void playbackRef.current?.close(), []);

  const busy = running || preparing !== null;
  const shownError = ioError ?? error;

  function playClip(samples: Float32Array) {
    void playbackRef.current?.close();
    playbackRef.current = play(samples, SAMPLE_RATE);
  }

  async function enhanceFrom(
    source: () => Promise<Float32Array>,
    kind: "file" | "mic",
  ) {
    setIoError(null);
    setPreparing(kind);
    try {
      const audio = await source();
      // `run` transfers the buffer to the worker, which detaches it — keep our
      // own copy so the "before" waveform and A/B playback still have samples.
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
    void enhanceFrom(
      async () => decodeToMono(await file.arrayBuffer(), SAMPLE_RATE),
      "file",
    );
  };

  const download = () => {
    if (!result) return;
    const url = URL.createObjectURL(toWavBlob(result.audio, result.sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = "enhanced.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <AudioWaveform className="size-6" /> Audio to Audio
        </h1>
        <p className="text-sm text-muted-foreground">
          Remove background noise from speech with DeepFilterNet3, at its native
          48&nbsp;kHz. The model runs on your GPU (WebGPU) or CPU (WASM) in a Web
          Worker — the audio never leaves your machine.
        </p>
      </div>

      <ModelPicker
        models={ENHANCE_MODELS}
        value={model}
        onChange={(m) => setModel(m.id)}
        disabled={busy || loading}
      />

      {/* LOAD — nothing downloads until asked (model-page-pattern.md §1.2). */}
      {idle && (
        <Button onClick={load} data-testid="load-model">
          Load model
        </Button>
      )}
      {error && !loading && (
        <Button variant="outline" onClick={retry}>
          Retry load
        </Button>
      )}
      <ModelStatus loading={loading} ready={ready} backend={backend} progress={progress} />

      {/* RUN */}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          disabled={!ready || busy}
          onClick={() =>
            void enhanceFrom(() => recordMic(RECORD_SECONDS, SAMPLE_RATE), "mic")
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

        {running && (
          <span className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Enhancing…
          </span>
        )}
      </div>

      {shownError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {shownError}
        </div>
      )}

      {/* OUTPUT — before and after, side by side, so the difference is audible
          and visible rather than asserted. */}
      {input && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Before / after</CardTitle>
            <CardDescription>
              Same clip, {(input.length / SAMPLE_RATE).toFixed(1)}s at 48&nbsp;kHz.
              Play both and listen to the noise floor.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <ClipRow
              title="Noisy input"
              samples={input}
              onPlay={() => playClip(input)}
              tone="text-muted-foreground"
            />
            {result && (
              <ClipRow
                title="Enhanced"
                samples={result.audio}
                onPlay={() => playClip(result.audio)}
                tone="text-primary"
                action={
                  <Button variant="outline" size="sm" onClick={download}>
                    <Download className="size-4" /> WAV
                  </Button>
                }
              />
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ClipRow({
  title,
  samples,
  onPlay,
  tone,
  action,
}: {
  title: string;
  samples: Float32Array;
  onPlay: () => void;
  tone: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-medium">{title}</span>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onPlay}>
            <Play className="size-4" /> Play
          </Button>
          {action}
        </div>
      </div>
      <Waveform samples={samples} className={tone} />
    </div>
  );
}
