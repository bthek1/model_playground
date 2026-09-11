// Audio Classification — in-browser sound tagging. Fixed-label models (AST,
// wav2vec2-KS) return top-k tags via `audio-classification`; CLAP scores the clip
// against your own free-text prompts via `zero-shot-audio-classification`. Both
// run client-side in the generic pipeline Web Worker (WebGPU, WASM fallback);
// audio is decoded to 16 kHz mono.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Tags } from "lucide-react";
import { useState } from "react";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
  DEFAULT_ZERO_SHOT_LABELS,
} from "@/audio/classification";
import type { ClassLabel } from "@/audio/pipelineTypes";
import { AUDIO_SAMPLES } from "@/audio/samples";
import { AudioSourcePanel } from "@/components/audio/AudioSourcePanel";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAudioClassifier } from "@/hooks/useAudioClassifier";
import { useAudioPick } from "@/hooks/useAudioPick";
import {
  useCacheRefresh,
  useModelSelection,
} from "@/model/useModelSelection";

export const Route = createFileRoute("/audio-classification")({
  component: AudioClassificationPage,
});

const RECORD_SECONDS = 5;

function AudioClassificationPage() {
  const session = useModelSelection({
    routeKey: "audio-classification",
    models: CLASSIFIER_MODELS,
    fallback:
      CLASSIFIER_MODELS.find((m) => m.id === DEFAULT_CLASSIFIER_MODEL) ??
      CLASSIFIER_MODELS[0],
  });
  const model = session.model.id;
  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    error,
    isZeroShot,
    result,
    classify,
    load,
    retry,
    cancel,
  } = useAudioClassifier(model);
  useCacheRefresh(session, ready);

  const [labelsText, setLabelsText] = useState(
    DEFAULT_ZERO_SHOT_LABELS.join("\n"),
  );
  const input = useAudioPick();

  const busy = running || input.preparing !== null;
  const labels = labelsText
    .split(/[\n,]/)
    .map((l) => l.trim())
    .filter(Boolean);

  // The only trigger. The clip is already decoded and held by `useAudioPick`,
  // so this re-runs the same audio as often as the user likes — after editing
  // CLAP's prompts, or after switching to a different checkpoint.
  const classifyCurrent = () => {
    const audio = input.take();
    if (!audio) return;
    input.clearError();
    void classify(audio, labels).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  // Capture failures (mic denied, undecodable file) belong in RUN; the model's
  // own errors split between LOAD and OUTPUT by status (§4).
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  return (
    <ModelPage
      icon={Tags}
      title="Audio Classification"
      description="Tag a sound entirely in your browser. Fixed-label models return the most likely tags; CLAP scores the clip against your own text prompts. The model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — nothing is uploaded."
      select={
        <ModelPicker
          models={CLASSIFIER_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
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
          disabledHint="Load a model to classify a sound. You can pick a clip first."
          controls={
            <Button
              disabled={!ready || busy || !input.clip || (isZeroShot && labels.length === 0)}
              onClick={classifyCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Classifying…
                </>
              ) : (
                <>
                  <Tags className="size-4" /> Classify
                </>
              )}
            </Button>
          }
        >
          <AudioSourcePanel
            clip={input.clip}
            preparing={input.preparing}
            samples={AUDIO_SAMPLES}
            sampleHint="Samples — speech clips, so a general sound tagger should land on a speech tag and CLAP should prefer a speech prompt."
            onFile={input.pickFile}
            onSample={input.pickSample}
            onRecord={input.record}
            recordSeconds={RECORD_SECONDS}
            busy={busy}
          >
            {isZeroShot && (
              <div className="flex min-h-0 flex-col space-y-1.5">
                <Label htmlFor="labels">
                  Labels to score against (one per line)
                </Label>
                <textarea
                  id="labels"
                  value={labelsText}
                  onChange={(e) => setLabelsText(e.target.value)}
                  rows={5}
                  className="min-h-28 w-full resize-none rounded-md border bg-background px-3 py-2 text-sm"
                  placeholder="a dog barking&#10;rain falling&#10;a car engine"
                />
                {/* Editing prompts changes the answer, so it must not change it
                    silently: CLAP is re-scored on the next Classify, not on
                    this keystroke. */}
                <p className="text-xs text-muted-foreground">
                  Edit the prompts, then press Classify to re-score the same clip.
                </p>
              </div>
            )}
          </AudioSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Predictions"
          description={
            isZeroShot
              ? "Similarity to each prompt"
              : "Most likely tags, highest score first"
          }
          running={running}
          runningLabel="Classifying…"
          error={runError}
          empty="Pick a clip, then press Classify — the ranked tags appear here."
        >
          {result && result.length > 0 && (
            <ul className="space-y-2">
              {result.map((p) => (
                <ScoreRow key={p.label} label={p.label} score={p.score} />
              ))}
            </ul>
          )}
        </OutputPanel>
      }
    />
  );
}

function ScoreRow({ label, score }: ClassLabel) {
  const pct = Math.round(score * 100);
  return (
    <li className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate">{label}</span>
        <span className="shrink-0 font-mono text-xs text-muted-foreground">
          {pct}%
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary"
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
    </li>
  );
}
