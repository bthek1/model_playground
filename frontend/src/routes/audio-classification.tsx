// Audio Classification — in-browser sound tagging. Fixed-label models (AST,
// wav2vec2-KS) return top-k tags via `audio-classification`; CLAP scores the clip
// against your own free-text prompts via `zero-shot-audio-classification`. Both
// run client-side in the generic pipeline Web Worker (WebGPU, WASM fallback);
// audio is decoded to 16 kHz mono.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Mic, Tags, Upload } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
  DEFAULT_ZERO_SHOT_LABELS,
} from "@/audio/classification";
import { decodeToMono, recordMic } from "@/audio/io";
import type { ClassLabel } from "@/audio/pipelineTypes";
import { sizeEstimate } from "@/audio/size";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAudioClassifier } from "@/hooks/useAudioClassifier";

export const Route = createFileRoute("/audio-classification")({
  component: AudioClassificationPage,
});

const RECORD_SECONDS = 5;

function AudioClassificationPage() {
  const [model, setModel] = useState(DEFAULT_CLASSIFIER_MODEL);
  const {
    status,
    ready,
    loading,
    progress,
    backend,
    running,
    error,
    isZeroShot,
    result,
    classify,
    load,
    retry,
  } = useAudioClassifier(model, false);

  const [labelsText, setLabelsText] = useState(
    DEFAULT_ZERO_SHOT_LABELS.join("\n"),
  );
  const [preparing, setPreparing] = useState<null | "file" | "mic">(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const meta = useMemo(
    () => CLASSIFIER_MODELS.find((m) => m.id === model) ?? CLASSIFIER_MODELS[0],
    [model],
  );

  const busy = running || preparing !== null;
  const labels = labelsText
    .split(/[\n,]/)
    .map((l) => l.trim())
    .filter(Boolean);

  async function runOn(
    source: () => Promise<Float32Array>,
    kind: "file" | "mic",
  ) {
    setIoError(null);
    setPreparing(kind);
    try {
      const audio = await source();
      await classify(audio, labels);
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
    void runOn(async () => decodeToMono(await file.arrayBuffer()), "file");
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
          onChange={(m) => setModel(m.id)}
          disabled={loading || busy}
        />
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          progress={progress}
          error={loadError}
          onLoad={load}
          onRetry={retry}
          disabled={busy}
        />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to classify a sound."
          controls={
            <>
              <Button
                disabled={!ready || busy}
                onClick={() => void runOn(() => recordMic(RECORD_SECONDS), "mic")}
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
        >
          {isZeroShot && (
            <div className="flex min-h-0 flex-1 flex-col space-y-1.5">
              <Label htmlFor="labels">
                Labels to score against (one per line)
              </Label>
              <textarea
                id="labels"
                value={labelsText}
                onChange={(e) => setLabelsText(e.target.value)}
                rows={5}
                className="min-h-28 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
                placeholder="a dog barking&#10;rain falling&#10;a car engine"
              />
            </div>
          )}
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
          empty="Record or upload a clip and the ranked tags appear here."
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
