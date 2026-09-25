// Sentence Similarity — the feature page with a cosine on the end, and a
// separate route rather than a toggle on it because it is a different question
// and a different Hub tag. (The same argument that keeps
// `/visual-question-answering` out of `/image-text-to-text`.)
//
// What the page is built to show, in order of how much it matters:
//
//   **A number needs its neighbours.** 0.82 means nothing on its own. So the
//   page scores a *set* of pairs and ranks them, and the sample pairs are chosen
//   to span the range — a paraphrase, an unrelated pair, a lexical-overlap trap,
//   and a negation.
//
//   **The negation pair is the honest part.** "The flight was cancelled" and
//   "The flight was not cancelled" score very high on every model here. That is
//   a limitation of sentence embeddings, not a bug in this page, and it is the
//   single most useful thing a similarity demo can show — so it ships as a
//   sample with the expectation stated next to it.
//
//   **Truncation re-derives.** Both sides are in hand, so cutting to 128
//   dimensions re-scores on the main thread and spends nothing — and because
//   only one of the five entries is Matryoshka-trained, the page reports what
//   the cut cost rather than claiming it was free.
//
//   **One side changing re-embeds one side.** The cache is keyed by the exact
//   text (`text/embed.ts`), so editing B and pressing Compare again is one
//   forward pass, not two.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { GitCompareArrows, Loader2 } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { VectorStrip } from "@/components/text/VectorStrip";
import { Button } from "@/components/ui/button";
import { useTextEmbed } from "@/hooks/useTextEmbed";
import { cosine } from "@/model/similarity";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_EMBED_MODEL,
  EMBED_MODELS,
  PAIR_SAMPLES,
  type PairSample,
} from "@/text/catalogue";
import { truncate, truncationSteps } from "@/text/embed";

export const Route = createFileRoute("/sentence-similarity")({
  component: SentenceSimilarityPage,
});

/** One scored pair, captured inside the run that produced it. */
interface ScoredPair {
  id: string;
  label: string;
  a: string;
  b: string;
  hint?: string;
  va: Float32Array;
  vb: Float32Array;
}

interface RunRecord {
  pairs: ScoredPair[];
  modelLabel: string;
  dim: number;
}

function SentenceSimilarityPage() {
  const session = useModelSelection({
    routeKey: "sentence-similarity",
    models: EMBED_MODELS,
    fallback:
      EMBED_MODELS.find((m) => m.id === DEFAULT_EMBED_MODEL) ?? EMBED_MODELS[0],
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
    cached,
  } = useTextEmbed(model.id);
  useCacheRefresh(session, ready);

  const [a, setA] = useState(PAIR_SAMPLES[0].a);
  const [b, setB] = useState(PAIR_SAMPLES[0].b);
  const [ran, setRan] = useState<RunRecord | null>(null);
  const [keep, setKeep] = useState<number | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const steps = useMemo(
    () => truncationSteps(ran?.dim ?? model.dim),
    [ran, model.dim],
  );

  /**
   * Scores at the current width. Pure: both vectors are in hand, so the
   * truncation buttons re-rank without a model. Truncating **and
   * renormalising** is what makes a 128-d cosine comparable to a 768-d one.
   */
  const scored = useMemo(() => {
    if (!ran) return null;
    const width = keep ?? ran.dim;
    const rows = ran.pairs.map((p) => {
      const ta = truncate(p.va, width);
      const tb = truncate(p.vb, width);
      return {
        pair: p,
        score: cosine(ta.vector, tb.vector),
        kept: Math.min(ta.kept, tb.kept),
        va: ta.vector,
        vb: tb.vector,
      };
    });
    return {
      width,
      rows,
      /** Best first — the ranking is the output, not any single number. */
      ranked: [...rows].sort((x, y) => y.score - x.score),
    };
  }, [ran, keep]);

  /** Embed and score a set of pairs — one batched call for every side at once. */
  async function compare(pairs: readonly PairSample[]) {
    if (!ready) return;
    const usable = pairs.filter((p) => p.a.trim() && p.b.trim());
    if (usable.length === 0) return;
    const texts = usable.flatMap((p) => [p.a.trim(), p.b.trim()]);
    const vectors = await run(texts);
    setRan({
      pairs: usable.map((p, i) => ({
        id: p.id,
        label: p.label,
        a: p.a.trim(),
        b: p.b.trim(),
        hint: p.hint,
        va: vectors[i * 2],
        vb: vectors[i * 2 + 1],
      })),
      modelLabel: model.label,
      dim: vectors[0]?.length ?? model.dim,
    });
  }

  const yours: PairSample = {
    id: "yours",
    label: "Your pair",
    a,
    b,
    hint: "",
  };

  return (
    <ModelPage
      icon={GitCompareArrows}
      title="Sentence Similarity"
      description={
        <>
          How close two sentences are, as a cosine between their embeddings —
          computed in your browser. The score is only meaningful against other
          scores, so the page ranks a set of pairs rather than showing one number
          on its own.
        </>
      }
      labels={{ run: "Two sentences", output: "Similarity" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={EMBED_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            <span data-testid="embed-dim">{model.dim}</span> dimensions, pooled
            by{" "}
            <code data-testid="embed-pooling" className="rounded bg-muted px-1 font-mono">
              {model.pooling}
            </code>
            . Switching model discards the vectors already computed — a 384-d
            MiniLM vector and a 768-d BGE vector are not comparable, and neither
            are two of the same width from different checkpoints.
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
          disabledHint="Load a model to compare. You can write both sentences first."
          controls={
            <>
              <Button
                disabled={!ready || running || !a.trim() || !b.trim()}
                onClick={() => void compare([yours]).catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Comparing…
                  </>
                ) : (
                  <>
                    <GitCompareArrows className="size-4" /> Compare
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                disabled={!ready || running}
                onClick={() => void compare(PAIR_SAMPLES).catch(() => {})}
                title="One batched call — two forward passes per pair."
              >
                Score all {PAIR_SAMPLES.length} sample pairs
              </Button>
              {cached > 0 && (
                <span
                  data-testid="embed-cache"
                  className="text-xs text-muted-foreground"
                >
                  {cached} remembered — changing one side re-embeds one side.
                </span>
              )}
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <div className="space-y-1.5">
              <label htmlFor="ss-a" className="text-xs font-medium">
                Sentence A
              </label>
              <textarea
                id="ss-a"
                value={a}
                onChange={(e) => setA(e.target.value)}
                rows={2}
                className="w-full resize-none rounded-md border bg-background px-3 py-2 text-sm"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="ss-b" className="text-xs font-medium">
                Sentence B
              </label>
              <textarea
                id="ss-b"
                value={b}
                onChange={(e) => setB(e.target.value)}
                rows={2}
                className="w-full resize-none rounded-md border bg-background px-3 py-2 text-sm"
              />
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Sample pairs — picking one fills both boxes and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {PAIR_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => {
                      setA(s.a);
                      setB(s.b);
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
          title="Similarity"
          description="Cosine between the two embeddings, in [−1, 1]. Ranked, because a single score has no scale of its own."
          meta={
            ran ? (
              <span>
                {ran.modelLabel} · {scored?.width}-d
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Embedding both sides…"
          error={runError}
          empty="Write two sentences and press Compare — or score the four sample pairs at once, which is the only way the number has a scale."
        >
          {ran && scored && (
            <div className="space-y-5" data-testid="similarity">
              <ul className="space-y-3">
                {scored.ranked.map(({ pair, score }) => (
                  <li
                    key={pair.id}
                    className="space-y-1"
                    data-testid={`pair-${pair.id}`}
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-xs font-medium">{pair.label}</span>
                      <span
                        className="font-mono text-sm tabular-nums"
                        data-testid={`score-${pair.id}`}
                      >
                        {score.toFixed(3)}
                      </span>
                    </div>
                    {/* A bar, mapped from [-1, 1] so a negative cosine is
                        visibly different from a small positive one. */}
                    <div className="h-1.5 w-full overflow-hidden rounded bg-muted">
                      <div
                        className="h-full rounded bg-primary"
                        style={{ width: `${((score + 1) / 2) * 100}%` }}
                      />
                    </div>
                    <p className="text-xs leading-snug text-muted-foreground">
                      <span className="italic">“{pair.a}”</span>
                      {" vs "}
                      <span className="italic">“{pair.b}”</span>
                    </p>
                    {pair.hint && (
                      <p className="text-xs leading-snug text-muted-foreground/80">
                        {pair.hint}
                      </p>
                    )}
                  </li>
                ))}
              </ul>

              <div className="space-y-2 border-t pt-4">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <p className="text-sm font-medium">Truncation</p>
                  <span className="text-xs text-muted-foreground">
                    Re-scores from the vectors above. Runs nothing.
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {steps.map((n) => {
                    const active = scored.width === n;
                    return (
                      <Button
                        key={n}
                        size="sm"
                        variant={active ? "default" : "outline"}
                        aria-pressed={active}
                        data-testid={`truncate-${n}`}
                        onClick={() => setKeep(n === ran.dim ? null : n)}
                      >
                        {n}-d
                      </Button>
                    );
                  })}
                </div>
                <p
                  data-testid="truncate-kept"
                  className="text-xs leading-snug text-muted-foreground"
                >
                  At {scored.width}-d the retained vectors hold at least{" "}
                  <strong className="tabular-nums">
                    {(Math.min(...scored.rows.map((r) => r.kept)) * 100).toFixed(1)}%
                  </strong>{" "}
                  of their full length, renormalised before scoring.{" "}
                  {model.matryoshka
                    ? "This checkpoint was Matryoshka-trained, so the ranking should survive the cut."
                    : "This checkpoint was not Matryoshka-trained, so whether the ranking survives is a measurement, not a promise."}
                </p>
              </div>

              <div className="space-y-3 border-t pt-4">
                <p className="text-sm font-medium">The vectors themselves</p>
                {scored.rows.slice(0, 1).map(({ pair, va, vb }) => (
                  <div key={pair.id} className="space-y-3">
                    <VectorStrip
                      vector={va}
                      data-testid="strip-a"
                      label={<span className="text-muted-foreground">A</span>}
                    />
                    <VectorStrip
                      vector={vb}
                      data-testid="strip-b"
                      label={<span className="text-muted-foreground">B</span>}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
