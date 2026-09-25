// Summarization — and the **lead-3 baseline shipped as part of the output**,
// because on this page the baseline is the most useful thing on screen.
//
// Three decisions, in order of how much they matter:
//
//   **The baseline is the empty state.** Before anything is downloaded, OUTPUT
//   already shows the first three sentences of the article and says that this is
//   what the model has to beat. On a news article — the genre every one of these
//   checkpoints was fine-tuned on — that is a genuinely hard baseline, and
//   watching a 284 MB model fail to beat `text.split(/(?<=[.!?])\s/).slice(0, 3)`
//   is the lesson. A page that shows only the model's output invites the
//   opposite conclusion by omission.
//
//   **This page exists because of one measurement.** Every configuration of
//   DistilBART except q8-on-WebGPU is over the app's size bar, and its q8 WASM
//   session **will not open** (the fourth family to hit that ONNX Runtime bug).
//   So distilbart is GPU-only here, and `Xenova/t5-small` — which the plan did
//   not consider — is the floor that keeps a CPU path. It is a much weaker
//   summarizer, which on this page is a feature.
//
//   **`max_new_tokens` and `min_length` re-run.** They change the generation,
//   not a view of it, so editing them spends nothing and the page says the next
//   GENERATE is a real second inference.
//
// The faithfulness check is a second model (27–50 MB, borrowed from
// `/zero-shot-classification`) with its own opt-in and its own LOAD, scored
// **per summary sentence** — an aggregate hides the one fabricated clause, which
// is the whole thing being looked for.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { FileText, Loader2, ShieldQuestion } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useEntailment, type Entailment } from "@/hooks/useEntailment";
import { useSummarize } from "@/hooks/useSummarize";
import { formatBytes, sizeEstimate } from "@/model/size";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  ARTICLE_SAMPLES,
  DEFAULT_SUMMARIZER,
  FAITHFULNESS_MODEL,
  LEAD_N,
  SUMMARIZER_MODELS,
  SUMMARY_MAX_TOKENS,
  SUMMARY_MIN_TOKENS,
  ZERO_SHOT_TEXT_MODELS,
} from "@/text/catalogue";
import { lead, sentenceCount, splitSentences, wordCount } from "@/text/lead3";

export const Route = createFileRoute("/summarization")({
  component: SummarizationPage,
});

/** One finished summary, captured inside the run that produced it. */
interface RunRecord {
  /** The article as it was when the button was pressed. */
  article: string;
  summary: string;
  /** The baseline for **that** article, so the two can never drift apart. */
  baseline: string;
  modelLabel: string;
  maxNewTokens: number;
  minLength: number;
}

function SummarizationPage() {
  const session = useModelSelection({
    routeKey: "summarization",
    models: SUMMARIZER_MODELS,
    fallback:
      SUMMARIZER_MODELS.find((m) => m.id === DEFAULT_SUMMARIZER) ??
      SUMMARIZER_MODELS[0],
  });
  const model = session.model;
  // Two of the three entries have no CPU path at all, so the probe is not
  // decoration here: it turns a row the machine cannot run into a disabled row
  // with the reason on it, instead of a download that fails at the end.
  const backendProbe = useBackendProbe();

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
  } = useSummarize(model.id);
  useCacheRefresh(session, ready);

  // The faithfulness model: opt-in, so `null` until the user asks for it.
  const [checking, setChecking] = useState(false);
  const faithful = useEntailment();
  const faithfulMeta = ZERO_SHOT_TEXT_MODELS.find(
    (m) => m.id === FAITHFULNESS_MODEL,
  );

  const [article, setArticle] = useState(ARTICLE_SAMPLES[0].text);
  const [maxNewTokens, setMaxNewTokens] = useState(SUMMARY_MAX_TOKENS);
  const [minLength, setMinLength] = useState(SUMMARY_MIN_TOKENS);
  const [ran, setRan] = useState<RunRecord | null>(null);
  const [scored, setScored] = useState<Entailment[] | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  /** The baseline for whatever is in the box right now. Pure, free, live. */
  const preview = useMemo(
    () => ({
      text: lead(article, LEAD_N),
      sentences: sentenceCount(article),
      words: wordCount(article),
    }),
    [article],
  );

  async function summarize() {
    if (!ready) return;
    const captured = article.trim();
    if (!captured) return;
    const out = await run({ text: captured, maxNewTokens, minLength });
    setRan({
      article: captured,
      summary: out,
      // Captured with the summary, so editing the box afterwards cannot leave
      // a summary of one article beside a baseline from another.
      baseline: lead(captured, LEAD_N),
      modelLabel: model.label,
      maxNewTokens,
      minLength,
    });
    setScored(null);
  }

  async function check() {
    if (!ran || !faithful.ready) return;
    const sentences = splitSentences(ran.summary).map((s) => s.text);
    if (sentences.length === 0) return;
    setScored(await faithful.run(ran.article, sentences));
  }

  const summarySentences = ran ? splitSentences(ran.summary).length : 0;

  return (
    <ModelPage
      icon={FileText}
      title="Summarization"
      description={
        <>
          Shorten an article in your browser — and compare it against the first{" "}
          {LEAD_N} sentences of that same article, which is a harder baseline
          than it sounds. News is written so the answer comes first.
        </>
      }
      labels={{ run: "Article", output: "Summary" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={SUMMARIZER_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            backend={backendProbe}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Fine-tuned on {model.domain}, which is what its output will sound
            like.
          </p>
          {model.backends?.length === 1 && (
            <p
              data-testid="gpu-only-note"
              className="text-xs leading-snug text-muted-foreground"
            >
              GPU only. This checkpoint's CPU session cannot be quantized — it
              does not open at all — and the unquantized build is{" "}
              {formatBytes(742_766_503)}, past what this app will download. The
              T5 entry is the CPU path.
            </p>
          )}

          <div className="space-y-1.5 border-t pt-2">
            <p className="text-xs font-medium">Faithfulness check</p>
            <p className="text-xs leading-snug text-muted-foreground">
              Score each summary sentence against the article with an entailment
              model. It is a{" "}
              <strong>second download and a second model in memory</strong>
              {faithfulMeta && (
                <>
                  {" — "}
                  {
                    sizeEstimate(faithfulMeta.params, faithfulMeta.bytes).label
                  }
                </>
              )}
              .
            </p>
            <div className="flex flex-wrap gap-1.5">
              <Button
                size="sm"
                variant={checking ? "outline" : "default"}
                aria-pressed={!checking}
                onClick={() => setChecking(false)}
              >
                Off
              </Button>
              <Button
                size="sm"
                variant={checking ? "default" : "outline"}
                aria-pressed={checking}
                data-testid="enable-faithfulness"
                onClick={() => setChecking(true)}
              >
                Add it
              </Button>
            </div>
          </div>
        </div>
      }
      load={
        <div className="space-y-3">
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
          {checking && (
            // A second model gets a second LOAD, in the LOAD slot — the
            // `/text-classification` head-to-head's shape.
            <div className="space-y-1.5 border-t pt-3" data-testid="faithful-load">
              <p className="text-xs font-medium text-muted-foreground">
                Faithfulness — {faithfulMeta?.label ?? FAITHFULNESS_MODEL}
              </p>
              <ModelStatus
                status={faithful.status}
                backend={faithful.backend}
                loadProgress={faithful.loadProgress}
                loadedInMs={faithful.loadedInMs}
                cached={session.cached.has(FAITHFULNESS_MODEL)}
                error={faithful.status === "error" ? faithful.error : null}
                onLoad={faithful.load}
                onCancel={faithful.cancel}
                onRetry={faithful.retry}
                disabled={faithful.running}
              />
            </div>
          )}
        </div>
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load a model to summarize. The lead-3 baseline already works — it needs no model."
          controls={
            <>
              <Button
                disabled={!ready || running || article.trim().length === 0}
                onClick={() => void summarize().catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Summarizing…
                  </>
                ) : (
                  <>
                    <FileText className="size-4" /> Summarize
                  </>
                )}
              </Button>
              <span
                data-testid="rerun-note"
                className="text-xs text-muted-foreground"
              >
                Changing the lengths below runs nothing — the next Summarize is a
                real second inference.
              </span>
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="sm-text" className="sr-only">
              Article to summarize
            </label>
            <textarea
              id="sm-text"
              value={article}
              onChange={(e) => setArticle(e.target.value)}
              rows={10}
              className="min-h-40 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Paste an article…"
            />
            <p className="text-xs text-muted-foreground tabular-nums">
              {preview.words} words · {preview.sentences} sentences
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label
                  htmlFor="sm-max"
                  className="text-xs font-medium"
                >
                  Max new tokens: <span className="tabular-nums">{maxNewTokens}</span>
                </label>
                <input
                  id="sm-max"
                  type="range"
                  min={20}
                  max={300}
                  step={10}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div className="space-y-1">
                <label htmlFor="sm-min" className="text-xs font-medium">
                  Min length: <span className="tabular-nums">{minLength}</span>
                </label>
                <input
                  id="sm-min"
                  type="range"
                  min={0}
                  max={120}
                  step={5}
                  value={minLength}
                  onChange={(e) => setMinLength(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — picking one fills the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {ARTICLE_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => setArticle(s.text)}
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
          title="Summary"
          description={`Beside the first ${LEAD_N} sentences of the same article, which is what a summarizer has to beat.`}
          meta={
            ran ? (
              <span>
                {ran.modelLabel} · {wordCount(ran.summary)}w
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Summarizing…"
          error={runError}
          /* The baseline **is** the empty state: it needs no model, so it is on
             screen before anything is downloaded, already saying what the model
             will be measured against. */
          empty={
            <span className="block w-full space-y-2 text-left">
              <span className="block text-xs font-medium text-foreground">
                Lead-{LEAD_N} baseline — no model, no download
              </span>
              <span
                data-testid="baseline-preview"
                className="block text-sm leading-relaxed text-foreground"
              >
                {preview.text || "Paste an article to see its first sentences."}
              </span>
              <span className="block text-xs text-muted-foreground">
                {preview.text
                  ? `${wordCount(preview.text)} words, quoted verbatim. Load a model and press Summarize to put a neural summary beside this.`
                  : ""}
              </span>
            </span>
          }
        >
          {ran && (
            <div className="space-y-5" data-testid="summarized">
              <div className="space-y-2">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <p className="text-sm font-medium">{ran.modelLabel}</p>
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    {wordCount(ran.summary)} words · max {ran.maxNewTokens} · min{" "}
                    {ran.minLength}
                  </span>
                </div>
                <p
                  data-testid="summary-text"
                  className="text-sm leading-relaxed whitespace-pre-wrap"
                >
                  {ran.summary}
                </p>
              </div>

              <div className="space-y-2 border-t pt-4">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <p className="text-sm font-medium">
                    Lead-{LEAD_N} baseline — no model
                  </p>
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    {wordCount(ran.baseline)} words
                  </span>
                </div>
                <p
                  data-testid="baseline-text"
                  className="text-sm leading-relaxed whitespace-pre-wrap text-muted-foreground"
                >
                  {ran.baseline}
                </p>
                <p className="text-xs leading-snug text-muted-foreground/80">
                  Quoted verbatim from the article, by a sentence splitter that
                  gets “Dr. Smith” wrong on purpose — a real splitter would be
                  more code than the model's own hook, and it is not this page's
                  subject.
                </p>
              </div>

              {checking && (
                <div className="space-y-2 border-t pt-4" data-testid="faithfulness">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm font-medium">Faithfulness</p>
                    <Button
                      size="sm"
                      variant="outline"
                      data-testid="check-faithfulness"
                      disabled={
                        !faithful.ready ||
                        faithful.running ||
                        summarySentences === 0
                      }
                      onClick={() => void check().catch(() => {})}
                    >
                      {faithful.running ? (
                        <>
                          <Loader2 className="size-4 animate-spin" /> Checking…
                        </>
                      ) : (
                        <>
                          <ShieldQuestion className="size-4" /> Check{" "}
                          {summarySentences} sentence
                          {summarySentences === 1 ? "" : "s"}
                        </>
                      )}
                    </Button>
                  </div>
                  <p className="text-xs leading-snug text-muted-foreground">
                    One forward pass per summary sentence, scored against the
                    article. Per sentence rather than once for the whole summary
                    — an aggregate hides the one fabricated clause, which is
                    exactly what this is looking for.
                  </p>
                  {scored && (
                    <ul className="space-y-2" data-testid="faithfulness-scores">
                      {scored.map((s, i) => (
                        <li key={i} className="space-y-1">
                          <div className="flex items-baseline justify-between gap-3">
                            <span className="text-xs leading-snug">
                              {s.sentence}
                            </span>
                            <span className="shrink-0 font-mono text-xs tabular-nums">
                              {s.score.toFixed(2)}
                            </span>
                          </div>
                          <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
                            <div
                              className="h-full rounded bg-primary"
                              style={{ width: `${s.score * 100}%` }}
                            />
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                  <p className="text-xs leading-snug text-muted-foreground/80">
                    The honest caveat: this entailment model was trained on
                    single-sentence premises, so a whole article is out of
                    distribution and is truncated at 512 tokens. A low score is
                    evidence worth reading, not a verdict.
                  </p>
                  {faithful.error && (
                    <p className="text-xs text-destructive">{faithful.error}</p>
                  )}
                </div>
              )}

              <p
                data-testid="ran-article"
                className="border-t pt-3 text-xs leading-snug text-muted-foreground italic"
              >
                Summarized: “{ran.article.slice(0, 120)}
                {ran.article.length > 120 ? "…" : ""}”
              </p>
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
