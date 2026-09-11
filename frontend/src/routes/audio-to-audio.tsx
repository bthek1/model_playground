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
// See docs/guides/adding-a-model.md §9 (a bare ONNX graph).

import { createFileRoute } from "@tanstack/react-router";
import { AudioWaveform, Download, Loader2, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  DEFAULT_ENHANCE_MODEL,
  ENHANCE_MODELS,
  SAMPLE_RATE,
} from "@/audio/enhance/types";
import { play, toWavBlob } from "@/audio/io";
import { AUDIO_SAMPLES } from "@/audio/samples";
import { AudioSourcePanel } from "@/components/audio/AudioSourcePanel";
import { Waveform } from "@/components/audio/Waveform";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useAudioPick } from "@/hooks/useAudioPick";
import { useEnhance } from "@/hooks/useEnhance";
import {
  useCacheRefresh,
  useModelSelection,
} from "@/model/useModelSelection";

export const Route = createFileRoute("/audio-to-audio")({
  component: AudioToAudioPage,
});

const RECORD_SECONDS = 6;

function AudioToAudioPage() {
  // DeepFilterNet3 fetches its graph through ONNX Runtime, which uses the HTTP
  // cache rather than the Cache Storage bucket the probe reads — so this route
  // keeps the persisted *selection* but never reports a model as cached, and
  // therefore never auto-resumes. That is the conservative failure direction:
  // it asks one extra time rather than spending bandwidth unasked.
  const session = useModelSelection({
    routeKey: "audio-to-audio",
    models: ENHANCE_MODELS,
    fallback:
      ENHANCE_MODELS.find((m) => m.id === DEFAULT_ENHANCE_MODEL) ??
      ENHANCE_MODELS[0],
  });
  const model = session.model.id;
  const {
    status,
    loading,
    ready,
    loadProgress,
    loadedInMs,
    backend,
    running,
    result,
    error,
    load,
    retry,
    cancel,
    run,
  } = useEnhance(model);
  useCacheRefresh(session, ready);

  // 48 kHz, not the 16 kHz default — DeepFilterNet3 is natively 48 kHz and
  // resampling down would discard the band it exists to repair.
  const input = useAudioPick({ sampleRate: SAMPLE_RATE });
  // The clip the current "before/after" pair belongs to, captured at run time,
  // so picking a new input doesn't retitle an older result's comparison.
  const [enhanced, setEnhanced] = useState<Float32Array | null>(null);
  const playbackRef = useRef<AudioContext | null>(null);

  // Stop playback when the page goes away — an AudioContext outlives the route.
  useEffect(() => () => void playbackRef.current?.close(), []);

  const busy = running || input.preparing !== null;
  // Each error in the slot that produced it (§4): capture failures in RUN, a
  // load failure in LOAD, an enhancement failure in OUTPUT.
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  function playClip(samples: Float32Array) {
    void playbackRef.current?.close();
    playbackRef.current = play(samples, SAMPLE_RATE);
  }

  // The only trigger. `take()` hands the worker a copy — it detaches the buffer
  // it is given, and the "before" waveform plus A/B playback need the samples
  // to survive the run.
  const enhanceCurrent = () => {
    const audio = input.take();
    if (!audio) return;
    input.clearError();
    setEnhanced(audio.slice());
    void run(audio).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
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
          error={input.error}
          disabledHint="Load the model to enhance a clip. You can pick one first."
          controls={
            <Button
              disabled={!ready || busy || !input.clip}
              onClick={enhanceCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Enhancing…
                </>
              ) : (
                <>
                  <AudioWaveform className="size-4" /> Enhance
                </>
              )}
            </Button>
          }
        >
          <AudioSourcePanel
            clip={input.clip}
            preparing={input.preparing}
            samples={AUDIO_SAMPLES}
            sampleHint="Samples — clean studio speech, so the interesting test is your own noisy recording."
            onFile={input.pickFile}
            onSample={input.pickSample}
            onRecord={input.record}
            recordSeconds={RECORD_SECONDS}
            busy={busy}
          />
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Before / after"
          description={
            enhanced
              ? `Same clip, ${(enhanced.length / SAMPLE_RATE).toFixed(1)}s at 48 kHz. Play both and listen to the noise floor.`
              : undefined
          }
          running={running}
          runningLabel="Enhancing…"
          error={runError}
          empty="Pick a noisy clip, then press Enhance — the original and the denoised version appear here, one above the other."
        >
          {enhanced && (
            <div className="space-y-5">
              <ClipRow
                title="Noisy input"
                samples={enhanced}
                onPlay={() => playClip(enhanced)}
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
