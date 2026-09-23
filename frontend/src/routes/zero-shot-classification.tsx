// Zero-Shot Classification — your own labels, typed here, with no fine-tuning
// and no head trained on them. An NLI model asks, once per label, "does this
// text entail *this* hypothesis?", and the answers are normalised into a score
// list.
//
// Two things about that sentence are the page, and both are invisible if the UI
// does not say them:
//
//   **N labels cost N forward passes.** There is no batching anywhere in the
//   pipeline — it is a `for` loop over the hypotheses. Ten labels is ten
//   inferences on one press, and on BART-large that is a wait a user will read
//   as a hang. The count is derived from the label list as it is edited, so
//   editing labels still spends nothing and the cost is on screen before the
//   click.
//
//   **The hypothesis template is part of the input, so it is on screen.** The
//   pipeline applies "This example is {}." unless told otherwise — the same
//   trap /zero-shot-image-classification found with "This is a photo of {}".
//   A page that templates silently is comparing a prompt the user cannot see,
//   so the template is a field, the composed hypothesis for the first label is
//   rendered underneath it, and both travel into OUTPUT with the result.
//
// `multi_label` is the third control and the one that looks like a filter. It
// changes the arithmetic — a softmax across labels versus an independent
// entailment-vs-contradiction softmax per label — and cannot be re-derived
// from scores already in hand, so flipping it runs nothing and the next
// GENERATE is a real second inference. The page says that beside the switch,
// exactly as /video-text-to-text does for its reverse toggle.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, Loader2, Tags, X } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ScoreList } from "@/components/text/ScoreList";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useZeroShotText } from "@/hooks/useZeroShotText";
import type { ClassLabel } from "@/model/types";
import { formatBytes, isHeavyDownload, sizeEstimate } from "@/model/size";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_ZERO_SHOT_TEXT,
  ZERO_SHOT_SAMPLES,
  ZERO_SHOT_TEXT_MODELS,
} from "@/text/catalogue";
import {
  BARE_HYPOTHESIS_TEMPLATE,
  composeHypothesis,
  DEFAULT_HYPOTHESIS_TEMPLATE,
  formatLabels,
  parseLabels,
  runCost,
  scoredIndependently,
  templateProblem,
} from "@/text/zeroShot";

export const Route = createFileRoute("/zero-shot-classification")({
  component: ZeroShotClassificationPage,
});

/**
 * Everything the answer on screen is an answer *to*, captured inside the run.
 *
 * All four fields are editable after a result lands, and every one of them
 * changes what the scores mean. A caption read from live state relabels a
 * finished result the moment the user types — the same rule
 * /image-text-to-text follows for its question.
 */
interface RunRecord {
  text: string;
  labels: string[];
  template: string;
  multiLabel: boolean;
  scores: ClassLabel[];
  /** Wall-clock for the whole run — N passes, not one. */
  ms: number;
}

function ZeroShotClassificationPage() {
  const session = useModelSelection({
    routeKey: "zero-shot-classification",
    models: ZERO_SHOT_TEXT_MODELS,
    fallback:
      ZERO_SHOT_TEXT_MODELS.find((m) => m.id === DEFAULT_ZERO_SHOT_TEXT) ??
      ZERO_SHOT_TEXT_MODELS[0],
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
  } = useZeroShotText(model.id);
  useCacheRefresh(session, ready);

  const [text, setText] = useState(ZERO_SHOT_SAMPLES[0].text);
  const [labelText, setLabelText] = useState(
    formatLabels(ZERO_SHOT_SAMPLES[0].labels),
  );
  const [template, setTemplate] = useState(DEFAULT_HYPOTHESIS_TEMPLATE);
  const [multiLabel, setMultiLabel] = useState(false);
  const [ran, setRan] = useState<RunRecord | null>(null);
  // The pass count of the run **in flight**, captured with everything else.
  // Reading it off the live label list instead would let the progress line
  // change under a run the user has already started — the same relabelling
  // mistake OUTPUT is careful about, one state earlier.
  const [pending, setPending] = useState<number | null>(null);

  // Every one of these is a pure derivation over what the user has typed. None
  // of them touches the model, which is what makes editing free.
  const labels = useMemo(() => parseLabels(labelText), [labelText]);
  const cost = useMemo(() => runCost(labels), [labels]);
  const problem = templateProblem(template);
  const preview = labels[0]
    ? composeHypothesis(template, labels[0])
    : composeHypothesis(template, "…");

  const size = sizeEstimate(model.params, model.bytes);
  const heavy = isHeavyDownload(model.bytes);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const canRun =
    ready && !running && text.trim().length > 0 && labels.length > 0 && !problem;

  async function classify() {
    if (!canRun) return;
    const captured = {
      text,
      labels,
      template,
      multiLabel,
    };
    const started = performance.now();
    setPending(captured.labels.length);
    try {
      const scores = await run({
        text: captured.text,
        labels: captured.labels,
        multiLabel: captured.multiLabel,
        hypothesisTemplate: captured.template,
      });
      setRan({ ...captured, scores, ms: performance.now() - started });
    } finally {
      setPending(null);
    }
  }

  function removeLabel(label: string) {
    setLabelText(formatLabels(labels.filter((l) => l !== label)));
  }

  return (
    <ModelPage
      icon={Tags}
      title="Zero-Shot Classification"
      description={
        <>
          Classify text against labels you invent here — no training, no
          fine-tuning, no head that has ever seen them. An entailment model runs
          in your browser, on your GPU (WebGPU) or CPU (WASM), and it runs{" "}
          <strong>once per label</strong>.
        </>
      }
      labels={{ run: "Text & labels", output: "Scores" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={ZERO_SHOT_TEXT_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Fine-tuned on <span className="font-medium">{model.domain}</span> —
            these are entailment models, so “what are the classes?” is a
            question you answer, not the checkpoint.
          </p>
        </div>
      }
      load={
        <div className="space-y-3">
          {heavy && !session.isCached && (
            <HeavyModelNotice
              label={model.label}
              sizeLabel={size.label}
              webgpu={formatBytes(model.bytes.webgpu ?? 0)}
            />
          )}
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
        </div>
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load a model to classify. Write the text and the labels first — none of it costs anything."
          controls={
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
              <Button
                disabled={!canRun}
                onClick={() => void classify().catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Classifying…
                  </>
                ) : (
                  <>
                    <Tags className="size-4" /> Classify
                  </>
                )}
              </Button>
              {/* The cost, in the model's units, derived from the list above —
                  which is why editing labels is free and pressing this is not. */}
              <span
                data-testid="pass-count"
                className={
                  cost.many
                    ? "text-xs font-medium text-amber-600 dark:text-amber-500"
                    : "text-xs text-muted-foreground"
                }
              >
                {cost.labels === 0
                  ? "No labels yet — add at least one."
                  : `${cost.labels} ${cost.labels === 1 ? "label" : "labels"} = ${cost.passes} forward ${
                      cost.passes === 1 ? "pass" : "passes"
                    }${cost.many ? " — this one will take a while." : ""}`}
              </span>
            </div>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-4">
            <div className="space-y-1.5">
              <label htmlFor="zs-text" className="text-xs font-medium">
                Text to classify
              </label>
              <textarea
                id="zs-text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                className="min-h-24 w-full resize-none rounded-md border bg-background px-3 py-2 text-sm"
                placeholder="Paste a ticket, a sentence, a paragraph…"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="zs-labels" className="text-xs font-medium">
                Labels — one per line, or comma separated
              </label>
              <textarea
                id="zs-labels"
                value={labelText}
                onChange={(e) => setLabelText(e.target.value)}
                rows={4}
                className="min-h-20 w-full resize-none rounded-md border bg-background px-3 py-2 font-mono text-sm"
                placeholder={"billing\noutage\nfeature request"}
              />
              <p className="text-xs leading-snug text-muted-foreground">
                Bare nouns — <code>billing</code>, not{" "}
                <code>a billing issue</code>. The template below supplies the
                sentence around them. Order does not matter: each label is its
                own independent pass, and the scores come back ranked.
              </p>
              {labels.length > 0 && (
                <div
                  className="flex flex-wrap gap-1.5 pt-0.5"
                  data-testid="label-chips"
                >
                  {labels.map((label) => (
                    <span
                      key={label}
                      className="inline-flex items-center gap-1 rounded-full border bg-muted/50 py-0.5 pr-1 pl-2.5 text-xs"
                    >
                      {label}
                      <button
                        type="button"
                        aria-label={`Remove ${label}`}
                        onClick={() => removeLabel(label)}
                        className="rounded-full p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                      >
                        <X className="size-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-1.5 border-t pt-3">
              <label htmlFor="zs-template" className="text-xs font-medium">
                Hypothesis template
              </label>
              <input
                id="zs-template"
                value={template}
                onChange={(e) => setTemplate(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-1.5 font-mono text-sm"
                spellCheck={false}
              />
              <div className="flex flex-wrap gap-1.5">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setTemplate(DEFAULT_HYPOTHESIS_TEMPLATE)}
                >
                  Default
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setTemplate(BARE_HYPOTHESIS_TEMPLATE)}
                >
                  Label only
                </Button>
              </div>
              {problem ? (
                <p
                  data-testid="template-problem"
                  role="alert"
                  className="flex items-start gap-1.5 text-xs text-destructive"
                >
                  <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                  {problem}
                </p>
              ) : (
                <p className="text-xs leading-snug text-muted-foreground">
                  The model is asked whether your text entails this, once per
                  label:{" "}
                  <span
                    data-testid="composed-hypothesis"
                    className="font-mono text-foreground"
                  >
                    “{preview}”
                  </span>
                </p>
              )}
            </div>

            <div className="space-y-1.5 border-t pt-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  variant={multiLabel ? "default" : "outline"}
                  aria-pressed={multiLabel}
                  data-testid="multi-label-toggle"
                  onClick={() => setMultiLabel((m) => !m)}
                >
                  Multi-label {multiLabel ? "on" : "off"}
                </Button>
                <span className="text-xs text-muted-foreground">
                  {multiLabel
                    ? "Each label scored on its own — several can be high."
                    : "One softmax across the labels — the scores sum to 1."}
                </span>
              </div>
              <p className="text-xs leading-snug text-muted-foreground">
                This changes the arithmetic, not the model, but the numbers it
                needs are not in a result already on screen —{" "}
                <strong>so flipping it runs nothing</strong>, and the next
                Classify is a second real inference.
              </p>
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — each one brings its own label set. Picking one fills
                both boxes and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {ZERO_SHOT_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => {
                      setText(s.text);
                      setLabelText(formatLabels(s.labels));
                    }}
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
          description="Every label you gave it, ranked — and the prompt it was actually asked, because that is half the answer."
          meta={
            ran ? (
              <span data-testid="ran-cost">
                {ran.labels.length} {ran.labels.length === 1 ? "pass" : "passes"}{" "}
                · {(ran.ms / 1000).toFixed(1)}s
              </span>
            ) : undefined
          }
          running={running}
          runningLabel={`Running ${pending ?? cost.passes} ${
            (pending ?? cost.passes) === 1 ? "pass" : "passes"
          }…`}
          error={runError}
          empty="Write some text, give it labels of your own, and press Classify — the model scores each label as a separate entailment question."
        >
          {ran && (
            <div className="space-y-4">
              <blockquote
                data-testid="ran-text"
                className="border-l-2 pl-3 text-sm text-muted-foreground italic"
              >
                {ran.text}
              </blockquote>

              <ScoreList
                scores={ran.scores}
                caption={
                  <>
                    Scored against{" "}
                    <span data-testid="ran-labels" className="font-medium">
                      {ran.labels.join(", ")}
                    </span>{" "}
                    using{" "}
                    <span data-testid="ran-template" className="font-mono">
                      “{ran.template}”
                    </span>
                    .
                  </>
                }
              />

              <p
                data-testid="scoring-note"
                className="text-xs leading-snug text-muted-foreground"
              >
                {scoredIndependently(ran.labels, ran.multiLabel)
                  ? ran.labels.length === 1
                    ? "One label, so there is nothing to normalise across: it was scored on its own, entailment against contradiction, whatever the multi-label switch said."
                    : "Multi-label: each score is that label's entailment against its own contradiction, so they do not sum to 1 and several can be high at once."
                  : "Single-label: one softmax over every label's entailment logit, so these sum to 1 — a label can only rise by pushing another down."}
              </p>
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * The second opt-in a heavy entry gets on top of the picker's size line.
 *
 * `isHeavyDownload` is a size predicate from `model/size.ts`, shared with
 * `/depth` rather than re-derived — and the reason it is a size test and not a
 * model id is that /depth's original id check would have been deleted along
 * with Depth Pro, taking the gate with it.
 */
function HeavyModelNotice({
  label,
  sizeLabel,
  webgpu,
}: {
  label: string;
  sizeLabel: string;
  webgpu: string;
}) {
  return (
    <Card data-testid="heavy-model-notice">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <AlertTriangle className="size-4" /> {label} — {webgpu} on WebGPU
        </CardTitle>
        <CardDescription>Read this before starting the download.</CardDescription>
      </CardHeader>
      <CardContent className="text-sm">
        <ul className="list-disc space-y-1.5 pl-5 text-muted-foreground">
          <li>
            <span className="font-medium text-foreground">{sizeLabel}</span> of
            weights — fifteen times the smallest model on this page, backend for
            backend, and cached only once it has finished. Nothing downloads
            until you press Load.
          </li>
          <li>
            It is also <span className="font-medium text-foreground">slow
            per label</span>, and this page runs one pass per label: six labels
            is six passes through a 407M-parameter model.
          </li>
          <li>
            What you get for it: the strongest entailment model of the four, and
            the one published zero-shot numbers are usually quoted from.
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
