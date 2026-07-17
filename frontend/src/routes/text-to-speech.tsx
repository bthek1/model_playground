// Text to Speech — in-browser speech synthesis. Kokoro-82M (kokoro-js) is the
// default; MMS-VITS and SpeechT5 run via the Transformers.js `text-to-speech`
// pipeline. Synthesis runs client-side in the TTS Web Worker (WebGPU, WASM
// fallback); the result plays through the Web Audio API and downloads as WAV.
// See docs/plans/in-progress/audio-models-in-browser.md.

import { createFileRoute } from "@tanstack/react-router";
import { AudioLines, Download, Loader2, Play, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { play, toWavBlob } from "@/audio/io";
import { DEFAULT_TTS_MODEL, TTS_MODELS } from "@/audio/tts";
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

export const Route = createFileRoute("/text-to-speech")({
  component: TextToSpeechPage,
});

const DEFAULT_TEXT =
  "Text to speech runs entirely in your browser — no server, no upload.";

function TextToSpeechPage() {
  const [model, setModel] = useState(DEFAULT_TTS_MODEL);
  const { ready, loading, progress, backend, result, running, error, synthesize } =
    useTts(model);

  const meta = useMemo(
    () => TTS_MODELS.find((m) => m.id === model) ?? TTS_MODELS[0],
    [model],
  );
  const [voice, setVoice] = useState(meta.voices?.[0]?.id);
  const [text, setText] = useState(DEFAULT_TEXT);
  const playbackRef = useRef<AudioContext | null>(null);

  const onSpeak = () => {
    void (async () => {
      try {
        const out = await synthesize(text, voice ? { voice } : undefined);
        void playbackRef.current?.close();
        playbackRef.current = play(out.audio, out.sampleRate); // hear it immediately
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
    a.download = "speech.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Stop playback when the page unmounts.
  useEffect(() => {
    return () => void playbackRef.current?.close();
  }, []);

  const seconds = result ? result.audio.length / result.sampleRate : 0;

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Volume2 className="size-6" /> Text to Speech
        </h1>
        <p className="text-sm text-muted-foreground">
          Turn text into speech entirely in your browser — the model runs on your
          GPU (WebGPU) or CPU (WASM) in a Web Worker. Nothing is uploaded. Pick a
          model and voice, type something, and press Speak.
        </p>
      </div>

      {/* Model picker */}
      <div className="flex flex-wrap gap-2">
        {TTS_MODELS.map((m) => (
          <Button
            key={m.id}
            variant={m.id === model ? "default" : "outline"}
            size="sm"
            disabled={running}
            onClick={() => {
              setModel(m.id);
              setVoice(m.voices?.[0]?.id);
            }}
            title={m.hint}
          >
            {m.label}
          </Button>
        ))}
      </div>

      <ModelStatus loading={loading} ready={ready} backend={backend} progress={progress} />

      {/* Voice picker (Kokoro only) */}
      {meta.voices && (
        <div className="space-y-1.5">
          <Label htmlFor="voice">Voice</Label>
          <select
            id="voice"
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            className="block w-full max-w-xs rounded-md border bg-background px-3 py-2 text-sm"
          >
            {meta.voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Text input */}
      <div className="space-y-1.5">
        <Label htmlFor="tts-text">Text</Label>
        <textarea
          id="tts-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          className="w-full max-w-2xl rounded-md border bg-background px-3 py-2 text-sm"
          placeholder="Type something to say…"
        />
      </div>

      {/* Transport */}
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={!ready || running || !text.trim()} onClick={onSpeak}>
          {running ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Synthesising…
            </>
          ) : (
            <>
              <Volume2 className="size-4" /> Speak
            </>
          )}
        </Button>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AudioLines className="size-4" /> Speech
              <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">
                {seconds.toFixed(1)}s · {(result.sampleRate / 1000).toFixed(0)} kHz
              </span>
            </CardTitle>
            <CardDescription>
              Generated audio — replay it or download the WAV.
            </CardDescription>
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
    </div>
  );
}
