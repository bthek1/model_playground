// Image Text to Text — a picture and a question in, an answer in prose out. The
// first route in the **Multimodal** category (roadmap §3.1), and the first
// vision-language model in the app.
//
// Three things make this page different from every vision route before it:
//
//  1. **The input is a conversation, not a value.** A picture alone is not a
//     request here — the model needs a question, and the question is held state
//     that costs nothing until GENERATE is pressed. Typing in the prompt box,
//     picking a preset question and choosing an image are all INPUT: none of
//     them runs anything.
//  2. **The pause before the first token is seconds, and it is labelled.** The
//     image encoder runs to completion before a single token exists. An
//     unlabelled pause is indistinguishable from a hang, so OUTPUT shows
//     "Encoding the image…" and then streams the answer as it arrives. Both are
//     `partial` messages inside one run — Machine A stays `ready` and `running`
//     stays an inflight count throughout.
//  3. **The answer is generated, so it can be wrong fluently.** A VLM that
//     cannot see the thing you asked about does not say so; it writes a
//     confident sentence. The page says this next to the output rather than
//     letting a plausible paragraph imply more than it should.
//
// Both checkpoints are gated to WebGPU by the picker rather than being allowed
// to fail at load — an autoregressive decoder on WASM is seconds per token, and
// a page that looks broken is worse than a model that is visibly unavailable.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, MessagesSquare, Sparkles } from "lucide-react";
import { useCallback, useState } from "react";

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
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_VLM_MODEL,
  MAX_INFERENCE_SIDE,
  VLM_MODELS,
} from "@/multimodal/types";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES, TEXT_SAMPLES } from "@/vision/samples";

const SAMPLES = [...IMAGE_SAMPLES, ...TEXT_SAMPLES];

/**
 * Preset questions. These are INPUT — tapping one fills the box and runs
 * nothing, the same rule the sample pictures follow.
 *
 * They are chosen to be answerable from a picture in one or two sentences, and
 * to show the range: description, counting, reading, and a judgement the model
 * will sometimes get confidently wrong.
 */
const PRESETS = [
  "What is in this picture?",
  "How many people are there?",
  "What does the text say?",
  "What is unusual about this scene?",
] as const;

const DEFAULT_PROMPT = PRESETS[0];

export const Route = createFileRoute("/image-text-to-text")({
  component: ImageTextToTextPage,
});

function ImageTextToTextPage() {
  const backendProbe = useBackendProbe();
  const session = useModelSelection({
    routeKey: "image-text-to-text",
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

  // The question. Held state: editing it costs nothing until GENERATE.
  const [prompt, setPrompt] = useState<string>(DEFAULT_PROMPT);
  // The question the current answer was actually asked — captured inside the
  // run, so editing the box afterwards cannot relabel a result on screen.
  const [asked, setAsked] = useState<string | null>(null);

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick();

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;
  const canRun = ready && !busy && picked != null && prompt.trim().length > 0;

  const ask = useCallback(
    async (image: RawImage, question: string) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      setAsked(question);
      await run(small, question);
    },
    [run],
  );

  const askCurrent = () => {
    if (!picked || !prompt.trim()) return;
    clearError();
    void ask(picked.image, prompt.trim()).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={MessagesSquare}
      title="Image Text to Text"
      description={
        <>
          Ask a question about a picture and get an answer in prose — a vision
          encoder and a language decoder, both running on your own GPU.
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
          models={VLM_MODELS}
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
          disabledHint="Load a model to ask about a picture. You can pick one and write your question first."
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
            sampleHint="Samples — the street shot has the most to ask about; the last two have printed text."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
          >
            <div className="space-y-1.5">
              <label className="text-sm font-medium" htmlFor="vlm-prompt">
                Question
              </label>
              <textarea
                id="vlm-prompt"
                data-testid="prompt-input"
                rows={2}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask something about the picture…"
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
                    aria-pressed={preset === prompt}
                    onClick={() => setPrompt(preset)}
                    title="Fill the question box — this does not run the model"
                  >
                    {preset}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {/* The expectation this page has to set, and the reason the
                    OUTPUT slot has an encoding state at all. */}
                The picture is encoded before the first word appears, so expect a
                pause and then an answer that streams in. A larger model is
                slower and better — that comparison is the point of the two
                choices above.
              </p>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Answer"
          description="What the model wrote, from the picture and your question."
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
          {/* Gated here rather than inside `AnswerView`: `OutputPanel` decides
              "is there a result" from whether `children` is null/false, so an
              element that merely *renders* null still counts as one and the
              empty state never shows. */}
          {(partial != null || result != null) && (
            <AnswerView partial={partial} text={result?.text} asked={asked} />
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * The OUTPUT body across all three of a run's states.
 *
 * `partial` takes precedence over `result` so the two can never be on screen at
 * once — the hook clears it the moment the result lands. Returning `null` when
 * there is neither is what lets `OutputPanel` fall back to its empty state, so
 * this must not render an empty wrapper instead.
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
        // The testid goes on a wrapper, not on `Markdown`: that component takes
        // only `{children, className}` and drops everything else, so a
        // `data-testid` passed to it is silently dead. (`/image-to-text` passes
        // one too, and it has never resolved — its test queries `slot-4`.)
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
          {/* A generated answer is fluent whether or not it is right. Saying so
              is cheaper than letting a confident paragraph imply more than a
              256M model can deliver. */}
          These are small models: they answer confidently whether or not they can
          actually see what you asked about. Check the answer against the
          picture.
        </p>
      )}
    </div>
  );
}
