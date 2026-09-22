// Document Question Answering — photograph a document, ask about it, get the
// answer back out of the page. Roadmap §3.3.
//
// Three things set this page apart from its siblings.
//
//  1. **It does not downscale, and that is deliberate.** Every other
//     vision-family route caps the source before inference. Donut's processor
//     resizes to a fixed 2560x1920 and *pads* rather than upscales, so the
//     encoder's cost is constant and a smaller source only means fewer real
//     pixels of print. Capping here would trade legibility for nothing. See
//     `MAX_SOURCE_SIDE` in `multimodal/docvqa/types.ts` — the cap that remains
//     is a memory bound at the processor's own dimension, not preprocessing.
//  2. **The answer is extracted, not reasoned.** Donut copies a span off the
//     page. Presenting it as though the model understood the document would
//     overclaim, so the page says what it actually does — the
//     `/video-classification` precedent, where framing the limitation is a
//     correctness requirement rather than decoration.
//  3. **"No answer" is an ordinary outcome with no error attached.** The
//     pipeline returns `{ answer: null }` when its `<s_answer>` match misses, so
//     OUTPUT renders that case explicitly instead of showing a blank panel on
//     exactly the documents the model found hardest.
//
// This is also the page with the strongest privacy argument in the app, and it
// says so plainly: people photograph payslips, medical letters and bank
// statements, and this answers questions about them without the image leaving
// the device. That is the reason the whole architecture was chosen, and this is
// the one route where a user can feel it.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { FileText, Loader2, Search } from "lucide-react";
import { useCallback, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { Button } from "@/components/ui/button";
import { useDocVqa } from "@/hooks/useDocVqa";
import { useImagePick } from "@/hooks/useImagePick";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_DOCVQA_MODEL,
  DOCVQA_MODELS,
  MAX_SOURCE_SIDE,
} from "@/multimodal/docvqa/types";
import { downscale } from "@/vision/image";
import { DOCUMENT_SAMPLES, TEXT_SAMPLES } from "@/vision/samples";

/** Documents first — the book cover and advertisement are the harder edge. */
const SAMPLES = [...DOCUMENT_SAMPLES, ...TEXT_SAMPLES];

/**
 * Preset questions. INPUT, like the samples: tapping one fills the box and runs
 * nothing.
 *
 * Phrased as field lookups rather than open questions, because that is what an
 * extractive model can actually answer — a preset the model cannot serve would
 * teach the user the wrong thing about the page.
 */
const PRESETS = [
  "What is the invoice number?",
  "What is the total?",
  "What is the date?",
  "Who is it addressed to?",
] as const;

const DEFAULT_QUESTION = PRESETS[0];

export const Route = createFileRoute("/document-question-answering")({
  component: DocumentQuestionAnsweringPage,
});

function DocumentQuestionAnsweringPage() {
  const backendProbe = useBackendProbe();
  const session = useModelSelection({
    routeKey: "document-question-answering",
    models: DOCVQA_MODELS,
    fallback:
      DOCVQA_MODELS.find((m) => m.id === DEFAULT_DOCVQA_MODEL) ??
      DOCVQA_MODELS[0],
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
  } = useDocVqa(model);
  useCacheRefresh(session, ready);

  // The question. Held state: editing it costs nothing until GENERATE.
  const [question, setQuestion] = useState<string>(DEFAULT_QUESTION);

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick();

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;
  const canRun = ready && !busy && picked != null && question.trim().length > 0;

  const ask = useCallback(
    async (document: RawImage, asked: string) => {
      // NOT preprocessing, and not the usual downscale: `MAX_SOURCE_SIDE` is the
      // processor's own longest dimension, so this is lossless with respect to
      // what the model sees and exists only to bound the decoded bitmap and the
      // structured clone. The processor still does the real resize.
      const bounded = await downscale(document, MAX_SOURCE_SIDE);
      await run(bounded, asked);
    },
    [run],
  );

  const askCurrent = () => {
    if (!picked || !question.trim()) return;
    clearError();
    void ask(picked.image, question.trim()).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={FileText}
      title="Document Question Answering"
      description={
        <>
          Ask a question about a photographed document and get the answer read
          straight off the page — no OCR step, and no upload.
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
          models={DOCVQA_MODELS}
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
          disabledHint="Load the model to question a document. You can pick one and write your question first."
          controls={
            <Button
              disabled={!canRun}
              onClick={askCurrent}
              title="Ask the document"
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Reading…
                </>
              ) : (
                <>
                  <Search className="size-4" /> Generate
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={SAMPLES}
            sampleHint="Samples — the invoice has the clearest fields; the last two are pictures with type rather than documents."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          >
            <div className="space-y-1.5">
              <label className="text-sm font-medium" htmlFor="docvqa-question">
                Question
              </label>
              <textarea
                id="docvqa-question"
                data-testid="question-input"
                rows={2}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask about a field on the page…"
                className="w-full min-w-0 resize-y rounded-lg border border-input bg-transparent px-2.5 py-1.5 text-base transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 md:text-sm dark:bg-input/30"
              />
              <div
                className="flex flex-wrap gap-2"
                role="group"
                aria-label="Preset questions"
                data-testid="presets"
              >
                {PRESETS.map((preset) => (
                  <Button
                    key={preset}
                    variant="outline"
                    size="sm"
                    aria-pressed={preset === question}
                    onClick={() => setQuestion(preset)}
                    title="Fill the question box — this does not run the model"
                  >
                    {preset}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground" data-testid="privacy-note">
                {/* The reason the whole architecture was chosen, on the one page
                    where the user can feel it. Plain sentence, not a tooltip. */}
                The document never leaves your device. Weights are downloaded to
                this tab and the page is read here — which is the point on a
                payslip, a medical letter or a bank statement.
              </p>
              <p className="text-xs text-muted-foreground" data-testid="resolution-note">
                {/* Every other vision route in this repo caps its source. This
                    one must not, and the reason is worth stating rather than
                    leaving as an unexplained inconsistency. */}
                Upload the sharpest version you have. This is the one page that
                does not shrink your image first: the model pads rather than
                enlarges, so a smaller picture costs legibility without running
                any faster.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Answer"
          description="The span the model read off the page, for the question you asked."
          running={running}
          runningLabel="Reading the document…"
          error={runError}
          empty="Pick a document, ask about a field on it, and the answer appears here."
        >
          {result && <AnswerView result={result} />}
        </OutputPanel>
      }
    />
  );
}

function AnswerView({
  result,
}: {
  result: { answer: string | null; question: string };
}) {
  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground" data-testid="answer-asked">
        Asked: {result.question}
      </p>

      {result.answer ? (
        <p className="text-lg font-medium break-words" data-testid="answer-text">
          {result.answer}
        </p>
      ) : (
        // Not an error, and not a blank panel. The pipeline's `<s_answer>` match
        // missed, which is what this model does when it cannot find the field.
        <p className="text-sm text-muted-foreground" data-testid="answer-none">
          The model did not find an answer on this page. That is its normal
          "no" — it looks for a field matching the question and returns nothing
          rather than guessing. Try naming the field as the document labels it,
          or a sharper photograph.
        </p>
      )}

      <p className="text-xs text-muted-foreground" data-testid="extractive-note">
        {/* The honest description of what just happened. Donut copies a span; it
            does not reason over the document, and a page that implied otherwise
            would be overclaiming. */}
        This is extraction, not reasoning: the model finds the text on the page
        that answers the question and copies it. It cannot add up a column or
        compare two figures.
      </p>
    </div>
  );
}
