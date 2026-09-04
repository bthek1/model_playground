// Text to Speech — in-browser speech synthesis. Kokoro-82M (kokoro-js) is the
// default; MMS-VITS and SpeechT5 run via the Transformers.js `text-to-speech`
// pipeline. Synthesis runs client-side in the TTS Web Worker (WebGPU, WASM
// fallback); the result plays through the Web Audio API and downloads as WAV.
//
// Reference implementation of the four-slot page pattern
// (docs/standards/model-page-pattern.md): the route is layout + task glue, and
// every stage — Select, Load, Input, Output — is a shared component.

import { createFileRoute } from "@tanstack/react-router";
import { AudioLines, Download, Loader2, Play, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { play, toWavBlob } from "@/audio/io";
import { sizeEstimate } from "@/audio/size";
import { DEFAULT_TTS_MODEL, TTS_MODELS } from "@/audio/tts";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useTts } from "@/hooks/useTts";

export const Route = createFileRoute("/text-to-speech")({
  component: TextToSpeechPage,
});

const DEFAULT_TEXT =
  "Text to speech runs entirely in your browser — no server, no upload.";

function TextToSpeechPage() {
  const [model, setModel] = useState(DEFAULT_TTS_MODEL);
  // autoLoad: false — the weights are the user's bandwidth (§1.2). Nothing is
  // downloaded until they press Load in the LOAD slot.
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
    a.download = "speech.wav";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Stop playback when the page unmounts.
  useEffect(() => {
    return () => void playbackRef.current?.close();
  }, []);

  const seconds = result ? result.audio.length / result.sampleRate : 0;
  // A load failure belongs in LOAD; anything else came from a run, so it belongs
  // in OUTPUT (§4).
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  return (
    <ModelPage
      icon={Volume2}
      title="Text to Speech"
      description="Turn text into speech entirely in your browser — the model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker. Nothing is uploaded."
      select={
        <ModelPicker
          models={TTS_MODELS}
          value={model}
          onChange={(m) => {
            setModel(m.id);
            setVoice(m.voices?.[0]?.id);
          }}
          disabled={loading || running}
        />
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          progress={progress}
          error={loadError}
          size={sizeEstimate(meta.params)}
          onLoad={load}
          onRetry={retry}
        />
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load a model to synthesise speech."
          controls={
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
          }
        >
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
        </InputPanel>
      }
      output={
        <OutputPanel
          title={
            <>
              <AudioLines className="size-4" /> Speech
            </>
          }
          description="Generated audio — replay it or download the WAV."
          meta={
            result
              ? `${seconds.toFixed(1)}s · ${(result.sampleRate / 1000).toFixed(0)} kHz`
              : undefined
          }
          running={running}
          runningLabel="Synthesising…"
          error={runError}
          empty="Press Speak and the generated audio appears here, ready to play or download."
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
              {seconds.toFixed(1)} seconds of audio at{" "}
              {(result.sampleRate / 1000).toFixed(0)} kHz.
            </p>
          )}
        </OutputPanel>
      }
    />
  );
}
