// Audio Classification — in-browser sound tagging. Fixed-label models (AST,
// wav2vec2-KS) return top-k tags via `audio-classification`; CLAP scores the clip
// against your own free-text prompts via `zero-shot-audio-classification`. Both
// run client-side in the generic pipeline Web Worker (WebGPU, WASM fallback);
// audio is decoded to 16 kHz mono.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { Tags } from "lucide-react";
import { useState } from "react";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
  DEFAULT_ZERO_SHOT_LABELS,
} from "@/audio/classification";
import { AUDIO_SAMPLES } from "@/audio/samples";
import { AudioSourcePanel } from "@/components/audio/AudioSourcePanel";
import { RunButton } from "@/components/model/RunButton";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ScoreList } from "@/components/model/ScoreList";
import { Label } from "@/components/ui/label";
import { useAudioClassifier } from "@/hooks/useAudioClassifier";
import { useAudioPick } from "@/hooks/useAudioPick";
import { useModelSelection } from "@/model/useModelSelection";
import { useTaskSlots } from "@/model/useTaskSlots";

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
  const task = useAudioClassifier(model);
  const {
    ready,
    running,
    isZeroShot,
    result,
    classify,
  } = task;

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
  const slots = useTaskSlots(session, task, {
    models: CLASSIFIER_MODELS,
    busy,
  });
  const { runError } = slots;

  return (
    <ModelPage
      icon={Tags}
      title="Audio Classification"
      description="Tag a sound entirely in your browser. Fixed-label models return the most likely tags; CLAP scores the clip against your own text prompts. The model runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — nothing is uploaded."
      select={
        <ModelPicker {...slots.picker} />
      }
      load={
        <ModelStatus {...slots.status} />
      }
      run={
        <InputPanel
          ready={ready}
          error={input.error}
          disabledHint="Load a model to classify a sound. You can pick a clip first."
          controls={
            <RunButton
              disabled={!ready || busy || !input.clip || (isZeroShot && labels.length === 0)}
              onRun={classifyCurrent}
              icon={Tags}
              running={running}
              runningLabel="Classifying…"
            >
              Classify
            </RunButton>
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
            // `quiet`: this page reports the ranking without the near-tie note.
            <ScoreList scores={result} format="percent" quiet />
          )}
        </OutputPanel>
      }
    />
  );
}
