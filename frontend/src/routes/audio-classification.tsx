// Audio Classification — in-browser sound tagging. Fixed-label models (AST,
// wav2vec2-KS) return top-k tags via `audio-classification`; CLAP scores the clip
// against your own free-text prompts via `zero-shot-audio-classification`. Both
// run client-side in the generic pipeline Web Worker (WebGPU, WASM fallback);
// audio is decoded to 16 kHz mono. See docs/plans/…/audio-models-in-browser.md.

import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Mic, Tags, Upload } from "lucide-react";
import { useRef, useState } from "react";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
  DEFAULT_ZERO_SHOT_LABELS,
} from "@/audio/classification";
import { decodeToMono, recordMic } from "@/audio/io";
import type { ClassLabel } from "@/audio/pipelineTypes";
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
import { useAudioClassifier } from "@/hooks/useAudioClassifier";

export const Route = createFileRoute("/audio-classification")({
  component: AudioClassificationPage,
});

const RECORD_SECONDS = 5;

function AudioClassificationPage() {
  const [model, setModel] = useState(DEFAULT_CLASSIFIER_MODEL);
  const { ready, loading, progress, backend, running, error, isZeroShot, result, classify } =
    useAudioClassifier(model);

  const [labelsText, setLabelsText] = useState(DEFAULT_ZERO_SHOT_LABELS.join("\n"));
  const [preparing, setPreparing] = useState<null | "file" | "mic">(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = running || preparing !== null;
  const shownError = ioError ?? error;
  const labels = labelsText
    .split(/[\n,]/)
    .map((l) => l.trim())
    .filter(Boolean);

  async function runOn(source: () => Promise<Float32Array>, kind: "file" | "mic") {
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

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Tags className="size-6" /> Audio Classification
        </h1>
        <p className="text-sm text-muted-foreground">
          Tag a sound entirely in your browser. Fixed-label models return the most
          likely tags; CLAP scores the clip against your own text prompts. The model
          runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — nothing is
          uploaded.
        </p>
      </div>

      {/* Model picker */}
      <div className="flex flex-wrap gap-2">
        {CLASSIFIER_MODELS.map((m) => (
          <Button
            key={m.id}
            variant={m.id === model ? "default" : "outline"}
            size="sm"
            disabled={busy}
            onClick={() => setModel(m.id)}
            title={m.hint}
          >
            {m.label}
          </Button>
        ))}
      </div>

      <ModelStatus loading={loading} ready={ready} backend={backend} progress={progress} />

      {/* Zero-shot: candidate prompts */}
      {isZeroShot && (
        <div className="space-y-1.5">
          <Label htmlFor="labels">Labels to score against (one per line)</Label>
          <textarea
            id="labels"
            value={labelsText}
            onChange={(e) => setLabelsText(e.target.value)}
            rows={5}
            className="w-full max-w-lg rounded-md border bg-background px-3 py-2 text-sm"
            placeholder="a dog barking&#10;rain falling&#10;a car engine"
          />
        </div>
      )}

      {/* Transport controls */}
      <div className="flex flex-wrap items-center gap-2">
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

        {running && (
          <span className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Classifying…
          </span>
        )}
      </div>

      {shownError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {shownError}
        </div>
      )}

      {/* Predictions */}
      {result && result.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Predictions</CardTitle>
            <CardDescription>
              {isZeroShot
                ? "Similarity to each prompt"
                : "Most likely tags, highest score first"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {result.map((p) => (
                <ScoreRow key={p.label} label={p.label} score={p.score} />
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
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
