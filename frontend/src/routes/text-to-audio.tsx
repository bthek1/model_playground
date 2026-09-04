// Text to Audio — text-prompted music generation with MusicGen-small, running
// client-side in the TTS Web Worker (text → audio is the same modality).
//
// This route is deliberately **gated**: MusicGen is 571 MB quantized and
// autoregressive (seconds of compute per second of audio). That gate used to be
// a mount trick — the generator was only mounted once the user opted in, because
// `useTts` began downloading on mount. The `idle` state (model-page-pattern.md
// §2) makes that unnecessary: the hook is always mounted, nothing downloads, and
// the warning simply sits in the LOAD slot above the Load button.
//
// Longer-form music and the diffusion models (AudioLDM, Stable Audio) have no
// browser path — those belong on a server.

import { createFileRoute } from "@tanstack/react-router";
import {
  AudioLines,
  Download,
  FlaskConical,
  Loader2,
  Music,
  Play,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { play, toWavBlob } from "@/audio/io";
import { sizeEstimate } from "@/audio/size";
import {
  DEFAULT_MUSIC_MODEL,
  DEFAULT_SECONDS,
  MAX_SECONDS,
  MUSIC_MODELS,
  tokensForSeconds,
} from "@/audio/textToAudio";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
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

/**
 * The opt-in notice. Everything expensive about this route is stated before
 * anything is fetched — and unlike the picker's size line, the cost here is as
 * much compute as bandwidth, which is why this route says more than the others.
 */
function ExperimentalNotice({ sizeLabel }: { sizeLabel: string }) {
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
      <CardContent className="text-sm">
        <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
          <li>
            <span className="font-medium text-foreground">{sizeLabel}</span> of
            model weights — many times larger than the speech models on the other
            audio pages. Cached after the first download.
          </li>
          <li>
            MusicGen is{" "}
            <span className="font-medium text-foreground">autoregressive</span>:
            it generates roughly 50 audio tokens per second of compute, so a{" "}
            {DEFAULT_SECONDS}-second clip takes tens of seconds on CPU.
          </li>
          <li>
            Quality is a toy compared with a server-side model. Longer-form music
            and the diffusion models (AudioLDM, Stable Audio) have no in-browser
            path at all.
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}

function TextToAudioPage() {
  const [model, setModel] = useState(DEFAULT_MUSIC_MODEL);
  const {
    status,
    ready,
    loading,
    progress,
    backend,
    result,
    running,
    error,
    synthesize,
    load,
    retry,
  } = useTts(model, false);

  const meta = useMemo(
    () => MUSIC_MODELS.find((m) => m.id === model) ?? MUSIC_MODELS[0],
    [model],
  );
  const size = useMemo(
    () => sizeEstimate(meta.params, meta.bytes),
    [meta],
  );

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
        /* surfaced via the hook's error state, in the OUTPUT slot */
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
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  return (
    <ModelPage
      icon={Music}
      title="Text to Audio"
      description="Generate short music clips from a text prompt, entirely in your browser. MusicGen runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — nothing is uploaded."
      select={
        <ModelPicker
          models={MUSIC_MODELS}
          value={model}
          onChange={(m) => setModel(m.id)}
          disabled={loading || running}
        />
      }
      load={
        <div className="space-y-3">
          {status === "idle" && (
            <ExperimentalNotice sizeLabel={size.label.replace("≈", "")} />
          )}
          <ModelStatus
            status={status}
            backend={backend}
            progress={progress}
            error={loadError}
            onLoad={load}
            onRetry={retry}
          />
        </div>
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load the model to generate audio."
          controls={
            <Button
              disabled={!ready || running || !prompt.trim()}
              onClick={onGenerate}
            >
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
          }
        >
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
        </InputPanel>
      }
      output={
        <OutputPanel
          title={
            <>
              <AudioLines className="size-4" /> Generated audio
            </>
          }
          description="Replay it or download the WAV."
          meta={
            result
              ? `${duration.toFixed(1)}s · ${(result.sampleRate / 1000).toFixed(0)} kHz`
              : undefined
          }
          running={running}
          runningLabel="Generating… this takes tens of seconds on CPU."
          error={runError}
          empty="Describe a clip and press Generate — the audio appears here."
          actions={
            <>
              <Button variant="outline" size="sm" onClick={onReplay}>
                <Play className="size-4" /> Play
              </Button>
              <Button variant="outline" size="sm" onClick={onDownload}>
                <Download className="size-4" /> Download WAV
              </Button>
            </>
          }
        >
          {result && (
            <p className="text-sm text-muted-foreground">
              {duration.toFixed(1)} seconds of audio at{" "}
              {(result.sampleRate / 1000).toFixed(0)} kHz.
            </p>
          )}
        </OutputPanel>
      }
    />
  );
}
