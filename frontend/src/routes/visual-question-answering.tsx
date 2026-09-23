// Visual Question Answering — **§3.1's engine, asked a different question.**
//
// This page downloads nothing new, adds no worker and adds no catalogue. It
// imports `useVlm` exactly as `/image-text-to-text` does, and the *only*
// difference between the two route files is how the question is shaped before it
// is sent. That constraint is the point rather than an accident: two routes over
// one engine is cheap and useful, and two routes that drift into two catalogues,
// two hooks and two sets of copy is the hazard. **If this page needs a change to
// the hook, the change belongs in `useVlm` and `/image-text-to-text` gets it
// too.**
//
// Why it is a route at all, rather than a mode on the existing page: neither
// classical VQA model has a usable export (`dandelin/vilt-b32-finetuned-vqa`,
// `Salesforce/blip-vqa-base` — no ONNX), so VQA in a browser *is* prompting a
// general VLM. The navigational argument is the one `/link-prediction` made
// against folding into `/graph`: a user looking for Visual Question Answering
// does not think to click "Image Text to Text", and the taxonomy has both rows
// because the Hub has both tags.
//
// What this page adds that §3.1 does not have:
//
//  1. **Answer length is the whole demonstration.** The same weights and the
//     same picture, with and without "answer in one word", produce visibly
//     different output. It is a toggle over the *prompt* — not a second model,
//     not a second run.
//  2. **The composed prompt is on screen.** A page that rewrites the prompt
//     behind the user's back is the `hypothesis_template` problem again: the
//     user compares two prompts while the model is shown two others. So
//     `composePrompt` returns the exact string, the INPUT slot displays it
//     before the click, and OUTPUT labels the answer with it afterwards.
//  3. **Flipping the toggle runs nothing.** It looks like a filter, which is
//     precisely why it is the §1.6 rule's hardest case on this page — it changes
//     what the *next* GENERATE will send, and costs nothing until then.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { HelpCircle, Loader2, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { Button } from "@/components/ui/button";
import { useImagePick } from "@/hooks/useImagePick";
import { useVlm } from "@/hooks/useVlm";
import { supportsShaderF16 } from "@/model/backend";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { composePrompt } from "@/multimodal/prompt";
import {
  DEFAULT_VLM_MODEL,
  MAX_INFERENCE_SIDE,
  VLM_MODELS,
} from "@/multimodal/types";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES, TEXT_SAMPLES } from "@/vision/samples";

const SAMPLES = [...IMAGE_SAMPLES, ...TEXT_SAMPLES];

/**
 * Preset questions, chosen for VQA rather than for description: each has a short
 * factual answer, so the terse toggle has something to be terse *about*. They
 * are INPUT — tapping one fills the box and runs nothing.
 */
const PRESETS = [
  "What animal is this?",
  "How many people are there?",
  "What colour is the main object?",
  "What does the text say?",
] as const;

const DEFAULT_QUESTION = PRESETS[0];

export const Route = createFileRoute("/visual-question-answering")({
  component: VisualQuestionAnsweringPage,
});

function VisualQuestionAnsweringPage() {
  // `requireShaderF16` because every model here loads `q4f16`: an adapter
  // without that feature loads the weights and then fails on the first
  // operator. Same gate as `/image-text-to-text`, same catalogue.
  const backendProbe = useBackendProbe({ requireShaderF16: true });
  const session = useModelSelection({
    routeKey: "visual-question-answering",
    models: VLM_MODELS,
    fallback:
      VLM_MODELS.find((m) => m.id === DEFAULT_VLM_MODEL) ?? VLM_MODELS[0],
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
    partial,
    error,
    load,
    retry,
    cancel,
    run,
  } = useVlm(model);
  useCacheRefresh(session, ready);

  // Both held state. Editing the question and flipping the toggle are INPUT:
  // neither costs anything until GENERATE.
  const [question, setQuestion] = useState<string>(DEFAULT_QUESTION);
  const [terse, setTerse] = useState(false);
  // The **composed** prompt the current answer was actually asked — captured
  // inside the run, so editing the box or flipping the toggle afterwards cannot
  // relabel a result already on screen.
  const [asked, setAsked] = useState<string | null>(null);

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick();

  const composed = useMemo(
    () => composePrompt(question, { terse }),
    [question, terse],
  );

  // The probe folds "no GPU" and "a GPU without f16" into one answer, which is
  // right for gating and wrong for explaining.
  const [hasF16, setHasF16] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    void supportsShaderF16().then((ok) => live && setHasF16(ok));
    return () => {
      live = false;
    };
  }, []);
  const f16Missing = backendProbe === "wasm" && hasF16 === false;

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;
  const canRun = ready && !busy && picked != null && composed.prompt.length > 0;

  const ask = useCallback(
    async (image: RawImage, prompt: string, maxNewTokens: number) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setAsked(prompt);
      await run(small, prompt, maxNewTokens);
    },
    [run],
  );

  const askCurrent = () => {
    if (!picked || !composed.prompt) return;
    clearError();
    void ask(picked.image, composed.prompt, composed.maxNewTokens).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={HelpCircle}
      title="Visual Question Answering"
      description={
        <>
          Ask a picture a question with a short answer. Same vision-language
          model as{" "}
          <span className="font-medium">Image Text to Text</span> — the
          difference is the prompt, and this page shows you exactly what it
          sends.
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
        <div className="space-y-3">
          <ModelPicker
            models={VLM_MODELS}
            value={model}
            onChange={session.setModel}
            disabled={loading || busy}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
            backend={backendProbe}
          />
          {f16Missing && (
            <p
              data-testid="f16-note"
              className="text-xs leading-snug text-amber-600 dark:text-amber-500"
            >
              This browser has a GPU, but its adapter does not support
              half-precision (<code>shader-f16</code>) in shaders. These models are
              4-bit with f16 activations, so they would download and then fail on
              the first operator — the page disables them rather than charging you
              for that.
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            {/* The catalogue is shared, so the weights are too: a model loaded
                on one page is already in the browser cache for the other. */}
            The same two checkpoints as Image Text to Text. If you loaded one
            there, it is already cached here.
          </p>
        </div>
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
          disabledHint="Load a model to ask a question. You can pick a picture and write the question first."
          controls={
            <Button disabled={!canRun} onClick={askCurrent} title="Ask the model">
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Generating…
                </>
              ) : (
                <>
                  <Sparkles className="size-4" /> Generate
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={SAMPLES}
            sampleHint="Samples — the first few have a single obvious answer; the last two have printed text to read."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          >
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-sm font-medium" htmlFor="vqa-question">
                  Question
                </label>
                <textarea
                  id="vqa-question"
                  data-testid="question-input"
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask something with a short answer…"
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
              </div>

              {/* The demonstration. A toggle over the prompt, not a filter over
                  a result and not a second model — so it runs nothing on flip,
                  and the line below says so rather than leaving the user to
                  guess from a page that did not move. */}
              <div className="space-y-1.5 rounded-lg border bg-muted/20 p-3">
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    data-testid="terse-toggle"
                    checked={terse}
                    onChange={(e) => setTerse(e.target.checked)}
                  />
                  Ask for a one-word answer
                </label>
                <p className="text-xs text-muted-foreground">
                  Nothing runs when you flip this — it changes what the next
                  Generate sends. Same weights, same picture, a different
                  question.
                </p>
              </div>

              {/* Exactly what will be sent, before it is sent. */}
              <div className="space-y-1">
                <p className="text-xs font-medium text-muted-foreground">
                  Sent to the model
                </p>
                <p
                  data-testid="composed-prompt"
                  className="rounded-lg border border-dashed px-2.5 py-1.5 font-mono text-xs break-words"
                >
                  {composed.prompt || (
                    <span className="text-muted-foreground italic">
                      Write a question above.
                    </span>
                  )}
                </p>
                <p className="text-xs text-muted-foreground">
                  {composed.instructed
                    ? `Capped at ${composed.maxNewTokens} new tokens, and the instruction was added by this page.`
                    : terse
                      ? `Capped at ${composed.maxNewTokens} new tokens. You already asked for a short answer, so nothing was added — instructing a small decoder twice makes it answer the instruction.`
                      : `Capped at ${composed.maxNewTokens} new tokens. Your question, unchanged.`}
                </p>
              </div>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Answer"
          description="What the model wrote, from the picture and the prompt above."
          meta={
            result ? (
              <span>
                {result.tokens} tokens · {(result.encodeMs / 1000).toFixed(1)}s
                encoding
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Generating…"
          error={runError}
          empty="Pick a picture, ask a question, and the model's answer appears here."
        >
          {(partial != null || result != null) && (
            <AnswerView partial={partial} text={result?.text} asked={asked} />
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * The OUTPUT body across a run's three states — identical in shape to
 * `/image-text-to-text`'s, with one difference: the label is the **composed**
 * prompt rather than the raw box, because that is what the model was given.
 */
function AnswerView({
  partial,
  text,
  asked,
}: {
  partial: { stage: "encoding" } | { stage: "generating"; text: string } | null;
  text?: string;
  asked: string | null;
}) {
  if (partial?.stage === "encoding") {
    return (
      <p
        data-testid="answer-encoding"
        className="flex items-center gap-2 text-sm text-muted-foreground"
      >
        <Loader2 className="size-4 animate-spin" />
        Encoding the image — the model reads the whole picture before it writes
        anything.
      </p>
    );
  }

  const body = partial?.stage === "generating" ? partial.text : text;
  if (body == null) return null;

  return (
    <div className="space-y-3">
      {asked && (
        <p className="text-xs text-muted-foreground" data-testid="answer-asked">
          Asked: {asked}
        </p>
      )}
      {body ? (
        // The testid goes on a wrapper: `Markdown` takes only
        // `{children, className}` and silently drops everything else.
        <div data-testid="answer-text">
          <Markdown>{body}</Markdown>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          The model generated nothing. Try a more specific question, or a picture
          with more in it.
        </p>
      )}
      {partial == null && (
        <p className="text-xs text-muted-foreground">
          These are small models: they answer confidently whether or not they can
          actually see what you asked about. Check the answer against the
          picture.
        </p>
      )}
    </div>
  );
}
