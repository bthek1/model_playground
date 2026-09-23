// Question Answering — extractive QA, where **the answer is a range in the
// passage** and the page shows it as one.
//
// That choice is the page, not its styling. An extractive model's answer *is* a
// span; quoting it as a standalone string is a lossy rendering that makes a
// confidently wrong answer look authoritative, while the same answer shown
// where it came from is obviously wrong at a glance. So OUTPUT marks the
// passage and never prints the answer alone as the primary result.
//
// **The model cannot say "I don't know", and saying so is a requirement of this
// page rather than a footnote.** SQuAD 1.1 heads always answer; the squad2
// checkpoints that can abstain have no ONNX export, so the behaviour is not
// merely unshipped, it is unavailable. The note is keyed off
// `QaModel.canAbstain` so it disappears on its own the day an abstaining export
// arrives — and one sample ships with a question its passage cannot answer, so
// the claim is demonstrated and not merely asserted. It is the same class of
// standing disclaimer `/video-classification` carries, and like that one it is
// pinned by a test so it cannot be deleted as decoration.
//
// The score sits beside the span **always**. It is the only signal there is,
// and hiding it makes a 0.03 answer look like a 0.99 one — though the
// unanswerable sample is there to show that a high score does not mean the
// question was answerable either.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { HelpCircle, Loader2, MessageCircleQuestion } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { SpanOverlay } from "@/components/text/SpanOverlay";
import { Button } from "@/components/ui/button";
import { useQa } from "@/hooks/useQa";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  ANSWER_LABEL,
  DEFAULT_QA_MODEL,
  QA_MODELS,
  QA_SAMPLES,
} from "@/text/catalogue";
import { highlight } from "@/text/highlight";
import type { QaAnswer } from "@/text/qa/types";

export const Route = createFileRoute("/question-answering")({
  component: QuestionAnsweringPage,
});

/**
 * One answer plus the exact inputs that produced it, captured inside the run.
 *
 * The passage here is what OUTPUT renders — never the textarea's current value.
 * Editing the box after a run must not restyle a finished result, and with a
 * highlight on screen the failure is worse than a stale label: the offsets
 * would be applied to a string they were not computed against, and the mark
 * would land on the wrong words.
 */
interface RunRecord {
  question: string;
  context: string;
  answer: QaAnswer;
}

function QuestionAnsweringPage() {
  const session = useModelSelection({
    routeKey: "question-answering",
    models: QA_MODELS,
    fallback: QA_MODELS.find((m) => m.id === DEFAULT_QA_MODEL) ?? QA_MODELS[0],
  });
  const model = session.model;

  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    error,
    load,
    retry,
    cancel,
    run,
  } = useQa(model.id);
  useCacheRefresh(session, ready);

  const [context, setContext] = useState(QA_SAMPLES[0].context);
  const [question, setQuestion] = useState(QA_SAMPLES[0].question);
  const [ran, setRan] = useState<RunRecord | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // The highlight is derived from the captured passage and the captured
  // offsets, so it is pinned to the pair that produced it.
  const marked = useMemo(() => {
    if (!ran || ran.answer.start == null || ran.answer.end == null) return null;
    return highlight(ran.context, [
      {
        start: ran.answer.start,
        end: ran.answer.end,
        label: ANSWER_LABEL,
        score: ran.answer.score,
      },
    ]);
  }, [ran]);

  const canAsk =
    ready && !running && context.trim().length > 0 && question.trim().length > 0;

  async function ask() {
    if (!canAsk) return;
    const capturedQuestion = question;
    const capturedContext = context;
    const answer = await run(capturedQuestion, capturedContext);
    setRan({
      question: capturedQuestion,
      context: capturedContext,
      answer,
    });
  }

  function pickSample(sample: (typeof QA_SAMPLES)[number]) {
    setContext(sample.context);
    setQuestion(sample.question);
  }

  return (
    <ModelPage
      icon={MessageCircleQuestion}
      title="Question Answering"
      description={
        <>
          Ask a question about a passage and watch the model point at the answer
          — entirely in your browser, on your GPU (WebGPU) or CPU (WASM). This
          is <span className="font-medium">extractive</span> QA: the model does
          not write an answer, it selects a range of your text. Nothing is
          uploaded.
        </>
      }
      labels={{ run: "Passage & question", output: "Answer" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={QA_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Fine-tuned on {model.domain}. It is the only extractive reader with
            an ONNX export — the SQuAD 2.0 checkpoints that can decline to
            answer have none, which is why the note below is a standing one.
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
          disabled={running}
        />
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load a model to ask a question. You can write the passage and the question first."
          controls={
            <Button
              disabled={!canAsk}
              onClick={() => void ask().catch(() => {})}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Reading…
                </>
              ) : (
                <>
                  <HelpCircle className="size-4" /> Answer
                </>
              )}
            </Button>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <div className="flex min-h-0 flex-1 flex-col gap-1.5">
              <label htmlFor="qa-context" className="text-xs font-medium">
                Passage
              </label>
              <textarea
                id="qa-context"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                rows={7}
                className="min-h-36 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
                placeholder="Paste the text the answer is somewhere inside…"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="qa-question" className="text-xs font-medium">
                Question
              </label>
              <input
                id="qa-question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                placeholder="What do you want to know?"
              />
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — picking one fills both boxes and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {QA_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => pickSample(s)}
                    title={s.hint}
                    data-testid={
                      s.unanswerable ? "sample-unanswerable" : undefined
                    }
                  >
                    {s.label}
                    {s.unanswerable && (
                      <span className="ml-1 text-[0.6rem] tracking-wide uppercase opacity-70">
                        no answer
                      </span>
                    )}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Answer"
          description={
            <>
              The model selects a range of your passage. It is marked in place,
              with its confidence — the answer is a location, not a sentence the
              model wrote.
              {!model.canAbstain && (
                // Standing, and in the slot's description rather than beside
                // the result, so it is on screen *before* the first answer as
                // well as under every one after it. A caveat that only appears
                // once you already believe the answer has arrived too late.
                <span
                  data-testid="no-abstain-note"
                  className="mt-2 block text-foreground"
                >
                  <span className="font-medium">
                    This model cannot say “I don&apos;t know”.
                  </span>{" "}
                  It was trained on SQuAD 1.1, where every question had an
                  answer in its passage, so it always returns a span — even when
                  the passage does not address the question, and a high score
                  does not mean it did. Try the{" "}
                  <span className="font-medium">
                    “A question the passage cannot answer”
                  </span>{" "}
                  sample. The SQuAD 2.0 checkpoints that can decline have no
                  ONNX export, so the behaviour is unavailable in a browser
                  rather than merely unshipped.
                </span>
              )}
            </>
          }
          meta={
            ran ? (
              <span data-testid="answer-score" className="tabular-nums">
                {ran.answer.score.toFixed(3)}
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Reading…"
          error={runError}
          empty="Write a passage and a question, then press Answer — the model's answer appears highlighted in your own text, with the score beside it."
        >
          {ran && (
            <div className="space-y-4">
              <p
                data-testid="ran-question"
                className="text-sm font-medium text-balance"
              >
                {ran.question}
              </p>

              {marked ? (
                // The range travels to the DOM beside the mark it produced.
                // Asserting the highlighted *words* is not enough on a passage
                // that says a name twice: the right string at the wrong
                // occurrence looks exactly like a correct answer.
                <div
                  data-testid="answer-range"
                  data-start={ran.answer.start ?? undefined}
                  data-end={ran.answer.end ?? undefined}
                >
                  <SpanOverlay
                    result={marked}
                    // One span, and the page names it directly above — the
                    // inline type tag would repeat "ANSWER" into the middle of
                    // a sentence the user is trying to read.
                    showLabels={false}
                    data-testid="answer-span"
                  />
                </div>
              ) : (
                // The alignment failed, so there is no honest highlight to
                // draw. The answer still stands; `wordPieceOffsets` refuses to
                // approximate a range, and a mark on the wrong words would be
                // indistinguishable from a working page.
                <div className="space-y-2" data-testid="answer-unaligned">
                  <p className="text-sm">
                    <span className="rounded bg-entity-1/15 px-1">
                      {ran.answer.decoded}
                    </span>
                  </p>
                  <p className="text-xs text-amber-600 dark:text-amber-500">
                    The answer could not be located in your passage — this
                    checkpoint's tokenizer does not line up with these
                    characters, so it is quoted rather than highlighted.
                  </p>
                </div>
              )}

              {ran.answer.truncated && (
                <p
                  data-testid="answer-truncated"
                  className="text-xs text-amber-600 dark:text-amber-500"
                >
                  The passage is longer than the model's 512-token window, and
                  the tail was never read. The answer comes only from the part
                  above.
                </p>
              )}

            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
