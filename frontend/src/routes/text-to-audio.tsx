// Text to Audio — text-prompted music generation with MusicGen-small, running
// client-side in the TTS Web Worker (text → audio is the same modality).
//
// This route is deliberately **gated**: MusicGen is 571 MB quantized and
// autoregressive (seconds of compute per second of audio), so nothing is
// downloaded until the user explicitly opts in. Longer-form music and the
// diffusion models (AudioLDM, Stable Audio) have no browser path — those belong
// on a server. See docs/plans/in-progress/audio-models-in-browser.md.

import { createFileRoute } from "@tanstack/react-router";
import {
  AudioLines,
  Download,
  FlaskConical,
  Loader2,
  Music,
  Play,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { play, toWavBlob } from "@/audio/io";
import { sizeEstimate } from "@/audio/size";
import {
  DEFAULT_MUSIC_MODEL,
  DEFAULT_SECONDS,
  MAX_SECONDS,
  MUSIC_MODELS,
  tokensForSeconds,
} from "@/audio/textToAudio";
import { ModelPicker } from "@/components/audio/ModelPicker";
import { ModelStatus } from "@/components/audio/ModelStatus";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useTts } from "@/hooks/useTts";

export const Route = createFileRoute("/text-to-audio")({
  component: TextToAudioPage,
});

const DEFAULT_PROMPT = "lo-fi hip hop with a mellow piano loop";

function TextToAudioPage() {
  // `useTts` starts downloading on mount, so the generator is only mounted once
  // the user has accepted the cost.
  const [enabled, setEnabled] = useState(false);

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Music className="size-6" /> Text to Audio
        </h1>
        <p className="text-sm text-muted-foreground">
          Generate short music clips from a text prompt, entirely in your
          browser. MusicGen runs on your GPU (WebGPU) or CPU (WASM) in a Web
          Worker — nothing is uploaded.
        </p>
      </div>

      {enabled ? (
        <MusicGenerator />
      ) : (
        <ExperimentalGate onEnable={() => setEnabled(true)} />
      )}
    </div>
  );
}

/**
 * The opt-in. Everything expensive about this route is stated before anything
 * is fetched: the download size, and that generation is far from real time.
 */
function ExperimentalGate({ onEnable }: { onEnable: () => void }) {
  const model = MUSIC_MODELS[0];
  const size = sizeEstimate(model.params, model.bytes);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FlaskConical className="size-4" /> Experimental — and slow
        </CardTitle>
        <CardDescription>
          Read this before starting the download.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
          <li>
            <span className="font-medium text-foreground">
              {size.label.replace("≈", "")}
            </span>{" "}
            of model weights — many times larger than the speech models on the
            other audio pages. Cached after the first download.
          </li>
          <li>
            MusicGen is <span className="font-medium text-foreground">autoregressive</span>:
            it generates roughly 50 audio tokens per second of compute, so a{" "}
            {DEFAULT_SECONDS}-second clip takes tens of seconds on CPU.
          </li>
          <li>
            Quality is a toy compared with a server-side model. Longer-form music
            and the diffusion models (AudioLDM, Stable Audio) have no in-browser
            path at all.
          </li>
        </ul>
        <Button onClick={onEnable}>
          <Music className="size-4" /> Download the model and continue
        </Button>
      </CardContent>
    </Card>
  );
}

function MusicGenerator() {
  const [model, setModel] = useState(DEFAULT_MUSIC_MODEL);
  const { ready, loading, progress, backend, result, running, error, synthesize } =
    useTts(model);

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [seconds, setSeconds] = useState(DEFAULT_SECONDS);
  const playbackRef = useRef<AudioContext | null>(null);

  const onGenerate = () => {
    void (async () => {
      try {
        const out = await synthesize(prompt, {
          maxNewTokens: tokensForSeconds(seconds),
        });
        void playbackRef.current?.close();
        playbackRef.current = play(out.audio, out.sampleRate);
      } catch {
        /* surfaced via the hook's error state */
      }
    })();
  };

  const onReplay = () => {
    if (!result) return;
    void playbackRef.current?.close();
    playbackRef.current = play(result.audio, result.sampleRate);
  };

  const onDownload = () => {
    if (!result) return;
    const url = URL.createObjectURL(toWavBlob(result.audio, result.sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = "music.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => () => void playbackRef.current?.close(), []);

  const duration = result ? result.audio.length / result.sampleRate : 0;

  return (
    <>
      <ModelPicker
        models={MUSIC_MODELS}
        value={model}
        onChange={(m) => setModel(m.id)}
        disabled={running}
      />

      <ModelStatus loading={loading} ready={ready} backend={backend} progress={progress} />

      <div className="space-y-1.5">
        <Label htmlFor="prompt">Prompt</Label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="w-full max-w-2xl rounded-md border bg-background px-3 py-2 text-sm"
          placeholder="e.g. an upbeat 8-bit chiptune loop"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="seconds">
          Length: <span className="tabular-nums">{seconds}s</span>
        </Label>
        <input
          id="seconds"
          type="range"
          min={1}
          max={MAX_SECONDS}
          step={1}
          value={seconds}
          onChange={(e) => setSeconds(Number(e.target.value))}
          className="block w-full max-w-xs"
        />
        <p className="text-xs text-muted-foreground">
          Generation time grows with length — expect well over {seconds}s of
          waiting on CPU.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={!ready || running || !prompt.trim()} onClick={onGenerate}>
          {running ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Generating…
            </>
          ) : (
            <>
              <Music className="size-4" /> Generate
            </>
          )}
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AudioLines className="size-4" /> Generated audio
              <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">
                {duration.toFixed(1)}s · {(result.sampleRate / 1000).toFixed(0)} kHz
              </span>
            </CardTitle>
            <CardDescription>Replay it or download the WAV.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={onReplay}>
              <Play className="size-4" /> Play
            </Button>
            <Button variant="outline" size="sm" onClick={onDownload}>
              <Download className="size-4" /> Download WAV
            </Button>
          </CardContent>
        </Card>
      )}
    </>
  );
}
