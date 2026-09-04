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
import { useEffect, useMemo, useRef, useState } from "react";

import {
  DEFAULT_ENHANCE_MODEL,
  ENHANCE_MODELS,
  SAMPLE_RATE,
} from "@/audio/enhance/types";
import { decodeToMono, play, recordMic, toWavBlob } from "@/audio/io";
import { sizeEstimate } from "@/audio/size";
import { Waveform } from "@/components/audio/Waveform";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useEnhance } from "@/hooks/useEnhance";

export const Route = createFileRoute("/audio-to-audio")({
  component: AudioToAudioPage,
});

const RECORD_SECONDS = 6;

function AudioToAudioPage() {
  const [model, setModel] = useState(DEFAULT_ENHANCE_MODEL);
  const {
    status,
    loading,
    ready,
    progress,
    backend,
    running,
    result,
    error,
    load,
    retry,
    run,
  } = useEnhance(model);

  const meta = useMemo(
    () => ENHANCE_MODELS.find((m) => m.id === model) ?? ENHANCE_MODELS[0],
    [model],
  );

  const [input, setInput] = useState<Float32Array | null>(null);
  const [preparing, setPreparing] = useState<null | "file" | "mic">(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const playbackRef = useRef<AudioContext | null>(null);

  // Stop playback when the page goes away — an AudioContext outlives the route.
  useEffect(() => () => void playbackRef.current?.close(), []);

  const busy = running || preparing !== null;
  // Each error in the slot that produced it (§4): capture failures in RUN, a
  // load failure in LOAD, an enhancement failure in OUTPUT.
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

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
    <ModelPage
      icon={AudioWaveform}
      title="Audio to Audio"
      description={
        <>
          Remove background noise from speech with DeepFilterNet3, at its native
          48&nbsp;kHz. The model runs on your GPU (WebGPU) or CPU (WASM) in a Web
          Worker — the audio never leaves your machine.
        </>
      }
      select={
        <ModelPicker
          models={ENHANCE_MODELS}
          value={model}
          onChange={(m) => setModel(m.id)}
          disabled={busy || loading}
        />
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          progress={progress}
          error={loadError}
          size={sizeEstimate(meta.params, meta.bytes)}
          onLoad={load}
          onRetry={retry}
          disabled={busy}
        />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load the model to enhance a clip."
          controls={
            <>
              <Button
                disabled={!ready || busy}
                onClick={() =>
                  void enhanceFrom(
                    () => recordMic(RECORD_SECONDS, SAMPLE_RATE),
                    "mic",
                  )
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
        />
      }
      output={
        <OutputPanel
          title="Before / after"
          description={
            input
              ? `Same clip, ${(input.length / SAMPLE_RATE).toFixed(1)}s at 48 kHz. Play both and listen to the noise floor.`
              : undefined
          }
          running={running}
          runningLabel="Enhancing…"
          error={runError}
          empty="Record or upload a noisy clip — the original and the denoised version appear here, side by side."
        >
          {input && (
            <div className="space-y-5">
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
            </div>
          )}
        </OutputPanel>
      }
    />
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
