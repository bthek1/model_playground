// Text Classification — the Natural Language Processing category's first route,
// and the smallest useful page in the app: a string in, a score list out. No
// decode step, no transport problem, no preprocessing. That absence is why it
// establishes `src/text/` rather than a heavier page doing it.
//
// The page's subject is the **head-to-head**. Three sentiment classifiers,
// three training domains; running one sentence through two of them shows that
// "domain-matched" is a claim about a *specific* domain, and that a model
// outside its domain is confidently wrong rather than uncertain. The comparison
// is a **second LOAD, not a free toggle** — a second model is a second
// download, and the page quotes it before the click.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Type } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ScoreList } from "@/components/text/ScoreList";
import { Button } from "@/components/ui/button";
import { useTextClassifier } from "@/hooks/useTextClassifier";
import type { ClassLabel } from "@/model/types";
import { formatBytes } from "@/model/size";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  CLASSIFIER_SAMPLES,
  DEFAULT_TEXT_CLASSIFIER,
  TEXT_CLASSIFIER_MODELS,
  type TextClassifierModel,
} from "@/text/catalogue";

export const Route = createFileRoute("/text-classification")({
  component: TextClassificationPage,
});

/** One model's answer, captured at run time so OUTPUT cannot be relabelled. */
interface Answer {
  model: TextClassifierModel;
  scores: ClassLabel[];
}

interface RunRecord {
  /** The exact string that was classified — not the textarea's current value. */
  text: string;
  primary: Answer;
  compare: Answer | null;
}

function TextClassificationPage() {
  const session = useModelSelection({
    routeKey: "text-classification",
    models: TEXT_CLASSIFIER_MODELS,
    fallback:
      TEXT_CLASSIFIER_MODELS.find((m) => m.id === DEFAULT_TEXT_CLASSIFIER) ??
      TEXT_CLASSIFIER_MODELS[0],
  });
  const model = session.model;

  const primary = useTextClassifier(model.id);
  useCacheRefresh(session, primary.ready);

  // The head-to-head's second model. `null` until the user opts in, because it
  // is a second download — see `compareCost` below, which is on screen before
  // the LOAD button it describes.
  const [compareId, setCompareId] = useState<string | null>(null);
  const compareModel =
    TEXT_CLASSIFIER_MODELS.find((m) => m.id === compareId) ?? null;
  const compare = useTextClassifier(compareModel?.id ?? model.id);

  const [text, setText] = useState(CLASSIFIER_SAMPLES[0].text);
  const [ran, setRan] = useState<RunRecord | null>(null);

  const running = primary.running || compare.running;
  const busy = running;

  // Each error in the slot that produced it (§4): a failed download belongs to
  // LOAD, a failed inference to OUTPUT.
  const loadError = primary.status === "error" ? primary.error : null;
  const runError = primary.status === "error" ? null : primary.error;

  const compareLive = compareModel != null && compare.ready;

  async function classify() {
    if (!primary.ready || text.trim().length === 0) return;
    const captured = text;
    const [a, b] = await Promise.all([
      primary.run(captured),
      compareLive ? compare.run(captured) : Promise.resolve(null),
    ]);
    setRan({
      text: captured,
      primary: { model, scores: a },
      compare: compareLive && b ? { model: compareModel, scores: b } : null,
    });
  }

  return (
    <ModelPage
      icon={Type}
      title="Text Classification"
      description={
        <>
          Score a sentence with a sentiment classifier running entirely in your
          browser — on your GPU (WebGPU) or CPU (WASM), in a Web Worker. The
          text is never uploaded. Load a second model to see the same sentence
          read two different ways.
        </>
      }
      labels={{ run: "Text", output: "Scores" }}
      select={
        <div className="space-y-4">
          <ModelPicker
            models={TEXT_CLASSIFIER_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={primary.loading || busy}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />

          <div className="space-y-1.5 border-t pt-3">
            <p className="text-xs font-medium">Compare against</p>
            <p className="text-xs leading-snug text-muted-foreground">
              The same sentence, read by a model trained on different writing.
              This is a <strong>second download and a second model in memory</strong>,
              not a view of the result you already have.
            </p>
            <div className="flex flex-wrap gap-1.5 pt-0.5">
              <Button
                size="sm"
                variant={compareId == null ? "default" : "outline"}
                disabled={busy}
                onClick={() => setCompareId(null)}
              >
                None
              </Button>
              {TEXT_CLASSIFIER_MODELS.filter((m) => m.id !== model.id).map(
                (m) => (
                  <Button
                    key={m.id}
                    size="sm"
                    variant={compareId === m.id ? "default" : "outline"}
                    disabled={busy}
                    onClick={() => setCompareId(m.id)}
                    title={`${m.hint} — ${describeCost(m)}`}
                  >
                    {m.label}
                  </Button>
                ),
              )}
            </div>
            {compareModel && (
              <p
                data-testid="compare-cost"
                className="pt-0.5 text-xs text-muted-foreground"
              >
                {compareModel.label} is a further {describeCost(compareModel)}.
              </p>
            )}
          </div>
        </div>
      }
      load={
        <div className="space-y-4">
          <ModelStatus
            status={primary.status}
            backend={primary.backend}
            loadProgress={primary.loadProgress}
            loadedInMs={primary.loadedInMs}
            cached={session.isCached}
            error={loadError}
            onLoad={session.onLoad(primary.load)}
            onCancel={session.onCancel(primary.cancel)}
            onRetry={primary.retry}
            disabled={busy}
          />

          {compareModel && (
            <div
              className="space-y-1.5 border-t pt-3"
              data-testid="compare-load"
            >
              <p className="text-xs font-medium">
                Comparison — {compareModel.label}
              </p>
              <ModelStatus
                status={compare.status}
                backend={compare.backend}
                loadProgress={compare.loadProgress}
                loadedInMs={compare.loadedInMs}
                cached={session.cached.has(compareModel.id)}
                error={compare.status === "error" ? compare.error : null}
                onLoad={compare.load}
                onCancel={compare.cancel}
                onRetry={compare.retry}
                disabled={busy}
              />
            </div>
          )}
        </div>
      }
      run={
        <InputPanel
          ready={primary.ready}
          disabledHint="Load a model to classify. You can write the sentence first."
          controls={
            <Button
              disabled={!primary.ready || busy || text.trim().length === 0}
              onClick={() => void classify().catch(() => {})}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Classifying…
                </>
              ) : (
                <>
                  <Type className="size-4" />
                  {compareLive ? "Classify with both" : "Classify"}
                </>
              )}
            </Button>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="tc-text" className="sr-only">
              Text to classify
            </label>
            <textarea
              id="tc-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
              className="min-h-28 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Type or paste a sentence…"
            />
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — chosen so the three models disagree. Picking one fills
                the box; it does not run anything.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {CLASSIFIER_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={busy}
                    onClick={() => setText(s.text)}
                    title={s.hint}
                  >
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Scores"
          description="Every label the head can emit, highest first — a near-tie is the information a single label hides."
          running={running}
          runningLabel="Classifying…"
          error={runError}
          empty="Write a sentence and press Classify — every label this model can emit appears here with its score."
        >
          {ran && (
            <div className="space-y-4">
              <blockquote
                data-testid="ran-text"
                className="border-l-2 pl-3 text-sm text-muted-foreground italic"
              >
                {ran.text}
              </blockquote>

              <div
                className={
                  ran.compare ? "grid gap-5 sm:grid-cols-2" : "grid gap-5"
                }
              >
                <AnswerCard answer={ran.primary} testId="answer-primary" />
                {ran.compare && (
                  <AnswerCard answer={ran.compare} testId="answer-compare" />
                )}
              </div>

              {ran.compare && (
                <p className="text-xs leading-snug text-muted-foreground">
                  Same sentence, two training domains. Where they disagree, the
                  model trained on this kind of writing is usually the one to
                  believe — and neither of them tells you which that is.
                </p>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

function AnswerCard({ answer, testId }: { answer: Answer; testId: string }) {
  return (
    <div className="space-y-2" data-testid={testId}>
      <p className="text-xs font-medium">
        {answer.model.label}
        <span className="ml-1.5 font-normal text-muted-foreground">
          trained on {answer.model.domain}
        </span>
      </p>
      <ScoreList scores={answer.scores} data-testid={`${testId}-scores`} />
    </div>
  );
}

/** The download, quoted for both backends — the guardrail, before the click. */
function describeCost(m: TextClassifierModel): string {
  return `${formatBytes(m.bytes.webgpu ?? 0)} on WebGPU · ${formatBytes(
    m.bytes.wasm ?? 0,
  )} on WASM`;
}
