// Image to Text — captioning, OCR and grounding. The first vision route with a
// **generative decoder**, which changes three things about the page:
//
//  1. **A run takes seconds, not milliseconds.** The OUTPUT slot has a running
//     state with an elapsed counter (`OutputPanel` already does that past two
//     seconds), and there is no live camera mode on this route — a webcam feed
//     at one frame per three seconds is not a demo, it is a hang with pictures.
//  2. **The mode selector is driven by capability flags, not by a fixed list.**
//     Only Florence-2 has four modes. A task token a model has never seen does
//     not error: it produces a confident, fluent, unrelated sentence. So the UI
//     offers exactly what the selected checkpoint declares, and switching model
//     resets an unsupported mode rather than sending it.
//  3. **Two output shapes from one task.** The text modes render through
//     `components/Markdown.tsx`; the grounding mode renders boxes on a canvas.
//     Routing them by the mode's own `isBoxMode` rather than by sniffing the
//     result keeps a truncated generation from being drawn as prose.
//
// Florence-2 is gated to WebGPU by the picker rather than being allowed to fail
// at load: four graphs and an autoregressive decoder on WASM is tens of seconds
// per caption, and a page that looks broken is worse than a model that is
// visibly unavailable.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, Type } from "lucide-react";
import { useCallback, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import { Button } from "@/components/ui/button";
import { useImagePick } from "@/hooks/useImagePick";
import { useImageToText } from "@/hooks/useImageToText";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  CAPTION_MODELS,
  DEFAULT_CAPTION_MODEL,
  MAX_INFERENCE_SIDE,
  MODE_HINTS,
  MODE_LABELS,
  type CaptionMode,
} from "@/vision/caption/types";
import {
  colorForLabel,
  drawBoxes,
  drawPixels,
  type Detection,
} from "@/vision/draw";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES, TEXT_SAMPLES } from "@/vision/samples";

/** The shared five, plus the two whose subject is printed text. */
const SAMPLES = [...IMAGE_SAMPLES, ...TEXT_SAMPLES];

export const Route = createFileRoute("/image-to-text")({
  component: ImageToTextPage,
});

function ImageToTextPage() {
  const backendProbe = useBackendProbe();
  const session = useModelSelection({
    routeKey: "image-to-text",
    models: CAPTION_MODELS,
    fallback:
      CAPTION_MODELS.find((m) => m.id === DEFAULT_CAPTION_MODEL) ??
      CAPTION_MODELS[0],
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
    modes,
    mode,
    setMode,
    load,
    retry,
    cancel,
    run,
  } = useImageToText(model, session.autoLoad);
  useCacheRefresh(session, ready);

  // The frame the model saw. Boxes come back in its pixels, so the canvas has
  // to be the same picture at the same size.
  const [frame, setFrame] = useState<RawImage | null>(null);

  const describe = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setFrame(small);
      await run(small);
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        const small = await downscale(next.image, MAX_INFERENCE_SIDE);
        setFrame(small);
        if (ready) await run(small);
      },
    });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const describeCurrent = () => {
    if (!picked) return;
    clearError();
    void describe(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Type}
      title="Image to Text"
      description={
        <>
          Caption a picture, read the text in it, or find and name what is in it
          — one download, four modes, all on your own GPU.
        </>
      }
      labels={{ output: "Answer" }}
      aside={
        result ? (
          <span
            data-testid="generate-ms"
            className="font-mono text-xs text-muted-foreground tabular-nums"
          >
            {(result.ms / 1000).toFixed(1)}s on {backend ?? "…"}
          </span>
        ) : undefined
      }
      select={
        <ModelPicker
          models={CAPTION_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
          backend={backendProbe}
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
          disabledHint="Load a model to describe a picture. You can pick one first."
          controls={
            <Button
              disabled={!ready || busy || !picked}
              onClick={describeCurrent}
              title={`Run ${MODE_LABELS[mode]}`}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Generating…
                </>
              ) : (
                <>
                  {/* "Generate", not the mode's own name: the mode selector a
                      few lines up already carries that label, and two buttons
                      reading "Caption" is ambiguous to a screen reader and to a
                      test. The mode still travels — in the button's title, in
                      the panel copy, and in the OUTPUT header. */}
                  <Type className="size-4" /> Generate
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={SAMPLES}
            sampleHint="Samples — the street shot has the most for a caption to get wrong; the last two are for OCR."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          >
            <div className="space-y-1.5">
              <span className="text-sm font-medium" id="mode-label">
                Mode
              </span>
              <div
                className="flex flex-wrap gap-2"
                role="group"
                aria-labelledby="mode-label"
                data-testid="modes"
              >
                {modes.map((option: CaptionMode) => (
                  <Button
                    key={option}
                    variant="outline"
                    size="sm"
                    aria-pressed={option === mode}
                    disabled={busy}
                    onClick={() => setMode(option)}
                    title={MODE_HINTS[option]}
                  >
                    {MODE_LABELS[option]}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {MODE_HINTS[mode]}
                {modes.length === 1 && (
                  <>
                    {" "}
                    This checkpoint only captions — the four-mode model is
                    Florence-2.
                  </>
                )}
              </p>
              <p className="text-xs text-muted-foreground">
                {/* The one expectation this page has to set. Every other vision
                    route answers in milliseconds; this one decodes a token at a
                    time. */}
                This is the one vision task with a generative decoder: expect
                seconds, and more of them for a detailed caption.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Answer"
          description="What the model generated, in the mode you asked for."
          meta={
            result ? (
              <span>{MODE_LABELS[result.mode]}</span>
            ) : undefined
          }
          running={running}
          runningLabel="Generating…"
          error={runError}
          empty="Pick a picture and the model's answer — a caption, the text it can read, or named boxes — appears here."
        >
          {result &&
            (result.kind === "boxes" ? (
              frame && (
                <GroundingView detections={result.detections} source={frame} />
              )
            ) : (
              <TextView text={result.text} />
            ))}
        </OutputPanel>
      }
    />
  );
}

function TextView({ text }: { text: string }) {
  if (!text) {
    return (
      <p className="text-sm text-muted-foreground">
        The model generated nothing. On OCR that usually means there is no
        legible text in the picture — try the invoice or a photo of a sign.
      </p>
    );
  }
  return <Markdown data-testid="answer-text">{text}</Markdown>;
}

function GroundingView({
  detections,
  source,
}: {
  detections: readonly Detection[];
  source: RawImage;
}) {
  const paint = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawPixels(ctx, source);
      drawBoxes(ctx, detections);
    },
    [detections, source],
  );

  return (
    <div className="space-y-3">
      <OverlayCanvas
        width={source.width}
        height={source.height}
        draw={paint}
        label={`Grounding: ${detections.length} object${detections.length === 1 ? "" : "s"}`}
        testId="grounding-canvas"
      />
      {detections.length > 0 ? (
        <ul className="space-y-1 text-sm" data-testid="grounding-list">
          {detections.map((d, i) => (
            <li key={`${d.label}-${i}`} className="flex items-center gap-2">
              <span
                aria-hidden
                className="size-3 shrink-0 rounded-sm"
                style={{ background: colorForLabel(d.label) }}
              />
              <span className="truncate">{d.label}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          The model named nothing in this picture.
        </p>
      )}
      <p className="text-xs text-muted-foreground">
        {/* Not a detector's confidence — Florence-2 writes a location out as
            tokens, so there is no score to show, and pretending otherwise would
            invent a number. */}
        Grounding has no confidence score: the model writes each box out as
        tokens rather than ranking candidates. Every box here is one it
        committed to.
      </p>
    </div>
  );
}
