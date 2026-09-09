// Image Classification — the first Computer Vision route, and the simplest
// instance of the four-slot pattern in the app: an image in, a ranked label list
// out, no decode step at all. It runs client-side in the generic vision Web
// Worker (WebGPU, WASM fallback) — the picture never leaves the machine.
//
// One deliberate decision in the OUTPUT slot: **the top five, never the argmax.**
// A 0.31 / 0.29 near-tie is the case worth seeing, and a single confident-looking
// label is exactly what hides it. The margin between first and second is printed
// for the same reason.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { ImageIcon, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useImageClassifier } from "@/hooks/useImageClassifier";
import type { ClassLabel } from "@/model/types";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_IMAGE_CLASSIFIER,
  IMAGE_CLASSIFIER_MODELS,
} from "@/vision/classification";
import { fromFile, fromUrl } from "@/vision/image";
import { IMAGE_SAMPLES, type ImageSample } from "@/vision/samples";

export const Route = createFileRoute("/image-classification")({
  component: ImageClassificationPage,
});

interface Picked {
  image: RawImage;
  /** Object URL or sample URL, for the preview. */
  previewUrl: string;
  /** True when `previewUrl` is ours to revoke. */
  owned: boolean;
  name: string;
}

function ImageClassificationPage() {
  const session = useModelSelection({
    routeKey: "image-classification",
    models: IMAGE_CLASSIFIER_MODELS,
    fallback:
      IMAGE_CLASSIFIER_MODELS.find((m) => m.id === DEFAULT_IMAGE_CLASSIFIER) ??
      IMAGE_CLASSIFIER_MODELS[0],
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
    result,
    error,
    load,
    retry,
    cancel,
    run,
  } = useImageClassifier(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [picked, setPicked] = useState<Picked | null>(null);
  const [preparing, setPreparing] = useState<null | "file" | "sample">(null);
  const [ioError, setIoError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // One object URL alive at a time, and none after unmount: a preview URL that
  // outlives its <img> pins the decoded bitmap in memory for the tab's lifetime.
  const ownedUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    },
    [],
  );

  const busy = running || preparing !== null;
  // Each error in the slot that produced it (§4): a failed decode belongs to
  // RUN, a failed download to LOAD, a failed inference to OUTPUT.
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  function adopt(next: Picked) {
    if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    ownedUrl.current = next.owned ? next.previewUrl : null;
    setPicked(next);
  }

  async function pick(
    open: () => Promise<Picked>,
    kind: "file" | "sample",
  ): Promise<void> {
    setIoError(null);
    setPreparing(kind);
    try {
      const next = await open();
      adopt(next);
      // Classify straight away when a model is live; otherwise the picture sits
      // in the preview and the Classify button lights up once it is.
      if (ready) await run(next.image);
    } catch (e) {
      setIoError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreparing(null);
    }
  }

  const onFile = (file: File | undefined) => {
    if (!file) return;
    void pick(
      async () => ({
        image: await fromFile(file),
        previewUrl: URL.createObjectURL(file),
        owned: true,
        name: file.name,
      }),
      "file",
    );
  };

  const onSample = (sample: ImageSample) =>
    void pick(
      async () => ({
        image: await fromUrl(sample.url),
        previewUrl: sample.url,
        owned: false,
        name: sample.label,
      }),
      "sample",
    );

  const classifyCurrent = () => {
    if (!picked) return;
    setIoError(null);
    void run(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  const margin =
    result && result.length > 1 ? result[0].score - result[1].score : null;

  return (
    <ModelPage
      icon={ImageIcon}
      title="Image Classification"
      description={
        <>
          Label a picture entirely in your browser. An ImageNet-1k classifier
          runs on your GPU (WebGPU) or CPU (WASM) in a Web Worker — the image is
          never uploaded.
        </>
      }
      labels={{ output: "Predictions" }}
      select={
        <ModelPicker
          models={IMAGE_CLASSIFIER_MODELS}
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
          restoring={session.restoring}
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
          error={ioError}
          disabledHint="Load a model to classify an image. You can pick a picture first."
          controls={
            <>
              <Button
                disabled={!ready || busy || !picked}
                onClick={classifyCurrent}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Classifying…
                  </>
                ) : (
                  <>
                    <ImageIcon className="size-4" /> Classify
                  </>
                )}
              </Button>

              <Button
                variant="outline"
                disabled={busy}
                onClick={() => fileRef.current?.click()}
              >
                {preparing === "file" ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Decoding…
                  </>
                ) : (
                  <>
                    <Upload className="size-4" /> Upload image
                  </>
                )}
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                aria-label="Upload an image"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  e.target.value = ""; // allow re-selecting the same file
                  onFile(file);
                }}
              />
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                onFile(e.dataTransfer.files?.[0]);
              }}
              className="flex min-h-40 flex-1 items-center justify-center overflow-hidden rounded-lg border border-dashed bg-muted/20 p-2"
            >
              {picked ? (
                <img
                  src={picked.previewUrl}
                  alt={`Selected input: ${picked.name}`}
                  className="max-h-72 max-w-full rounded object-contain"
                />
              ) : (
                <p className="px-4 text-center text-sm text-balance text-muted-foreground">
                  Drop an image here, upload one, or start from a sample below.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">
                Samples — the first two are easy, the rest are the cases where a
                top-1 label starts to mislead.
              </p>
              <div className="flex flex-wrap gap-2">
                {IMAGE_SAMPLES.map((sample) => (
                  <Button
                    key={sample.id}
                    variant="outline"
                    size="sm"
                    disabled={busy}
                    title={sample.hint}
                    onClick={() => onSample(sample)}
                  >
                    {sample.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Predictions"
          description="Top five, highest score first — a near-tie at the top is the thing worth seeing."
          meta={
            margin != null ? (
              <span className="tabular-nums">
                top-2 margin {margin.toFixed(2)}
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Classifying…"
          error={runError}
          empty="Pick an image and its five most likely labels appear here, with their scores."
        >
          {result && result.length > 0 && (
            <div className="space-y-4">
              {margin != null && margin < 0.1 && (
                <p className="text-xs text-amber-600 dark:text-amber-500">
                  The top two are within {margin.toFixed(2)} of each other — this
                  model is not confident, whatever the first row says.
                </p>
              )}
              <ul className="space-y-2">
                {result.map((p) => (
                  <ScoreRow key={p.label} label={p.label} score={p.score} />
                ))}
              </ul>
            </div>
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
        <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
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
