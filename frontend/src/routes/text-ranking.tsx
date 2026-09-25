// Text Ranking — **all four retrieval stages, client-side**, over a corpus the
// user pastes in: BM25, dense retrieval, hybrid RRF, and cross-encoder
// reranking. Two of the four need no model at all.
//
// The page is arranged around its cost model, because that is the thing being
// taught:
//
//   **BM25 and RRF re-derive.** Changing `k1`, `b` or RRF's `k` re-scores from
//   a held index on the main thread and spends nothing — the same rule `/vad`'s
//   threshold follows. Only two buttons spend.
//
//   **Embedding the corpus is N forward passes, once.** Every query after that
//   is a dot product. That is dense retrieval's actual claim, and it is why the
//   two triggers are separate: a new corpus costs N passes, a new query costs
//   one.
//
//   **Reranking is one forward pass per candidate, and cannot be
//   precomputed.** A cross-encoder scores `{ text, text_pair }` — a *pair* — so
//   there is no per-document vector to cache, which is exactly why it reranks a
//   shortlist of 20 rather than the whole corpus. The pass count sits beside the
//   trigger.
//
//   **Four columns, not one final list.** The page's subject is how the four
//   stages *disagree*; a single fused answer hides it. The rank movement between
//   stages is the output.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ListOrdered, Loader2, Search } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useRanking } from "@/hooks/useRanking";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_RANKING_PAIR,
  RANKING_CORPUS,
  RANKING_PAIRS,
  RANKING_SAMPLES,
  RANK_SHOW,
  RERANK_TOP_K,
} from "@/text/catalogue";
import {
  buildIndex,
  DEFAULT_BM25,
  search,
  type Bm25Params,
} from "@/text/bm25";
import { DEFAULT_RRF_K, fuse, rankOf } from "@/text/rrf";

export const Route = createFileRoute("/text-ranking")({
  component: TextRankingPage,
});

/** One press of Search, captured so the columns can never disagree. */
interface RunRecord {
  query: string;
  corpus: string[];
  dense: number[];
  reranked: number[];
  /** Cross-encoder scores, for the rerank column's numbers. */
  rerankScores: Map<number, number>;
  pairLabel: string;
}

function parseCorpus(raw: string): string[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function TextRankingPage() {
  const session = useModelSelection({
    routeKey: "text-ranking",
    models: RANKING_PAIRS,
    fallback:
      RANKING_PAIRS.find((p) => p.id === DEFAULT_RANKING_PAIR) ??
      RANKING_PAIRS[0],
  });
  const pairEntry = session.model;

  const {
    pair,
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
    embedding,
    embedded,
    embedCorpus,
    denseSearch,
    rerank,
    clear,
  } = useRanking(pairEntry.id);
  useCacheRefresh(session, ready);

  const [corpusText, setCorpusText] = useState(RANKING_CORPUS.join("\n"));
  const [query, setQuery] = useState(RANKING_SAMPLES[0].query);
  const [bm25, setBm25] = useState<Bm25Params>(DEFAULT_BM25);
  const [rrfK, setRrfK] = useState(DEFAULT_RRF_K);
  const [ran, setRan] = useState<RunRecord | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const corpus = useMemo(() => parseCorpus(corpusText), [corpusText]);

  /**
   * BM25 over whatever is in the box, at whatever `k1`/`b` are set to. Pure,
   * synchronous, and needs no model — which is why it is live rather than
   * behind a button.
   */
  const bm25Ranked = useMemo(() => {
    if (corpus.length === 0 || !query.trim()) return [];
    return search(buildIndex(corpus), query, bm25).map((s) => s.doc);
  }, [corpus, query, bm25]);

  /**
   * The hybrid column, fused from the two rank lists in hand. Also pure — so
   * dragging RRF's `k` re-fuses and spends nothing.
   */
  const hybrid = useMemo(() => {
    if (!ran) return [];
    // BM25 is recomputed against the **captured** corpus, not the live one, so
    // all four columns are about the same documents.
    const lexical = search(buildIndex(ran.corpus), ran.query, bm25).map(
      (s) => s.doc,
    );
    return fuse([lexical, ran.dense], rrfK).map((f) => f.doc);
  }, [ran, bm25, rrfK]);

  const capturedBm25 = useMemo(() => {
    if (!ran) return [];
    return search(buildIndex(ran.corpus), ran.query, bm25).map((s) => s.doc);
  }, [ran, bm25]);

  async function runSearch() {
    if (!ready) return;
    const capturedQuery = query.trim();
    const capturedCorpus = corpus;
    if (!capturedQuery || capturedCorpus.length === 0) return;

    const dense = (await denseSearch(capturedQuery, capturedCorpus)).map(
      (s) => s.doc,
    );
    const lexical = search(
      buildIndex(capturedCorpus),
      capturedQuery,
      bm25,
    ).map((s) => s.doc);
    // The reranker sees the hybrid shortlist: reranking the dense list alone
    // would hide whatever BM25 found that the embedder missed, which on this
    // page's own sample corpus is the interesting half.
    const shortlist = fuse([lexical, dense], rrfK)
      .map((f) => f.doc)
      .slice(0, RERANK_TOP_K);
    const scores = await rerank(capturedQuery, capturedCorpus, shortlist);

    setRan({
      query: capturedQuery,
      corpus: capturedCorpus,
      dense,
      reranked: scores.map((s) => s.doc),
      rerankScores: new Map(scores.map((s) => [s.doc, s.score])),
      pairLabel: pair.label,
    });
  }

  const rerankCount = Math.min(RERANK_TOP_K, corpus.length);

  return (
    <ModelPage
      icon={ListOrdered}
      title="Text Ranking"
      description={
        <>
          The whole retrieval stack, in your browser: keyword search, dense
          embeddings, a hybrid fusion of the two, and a cross-encoder rerank.{" "}
          <strong>Two of the four stages need no model at all</strong>, which is
          the most useful thing this page has to say.
        </>
      }
      labels={{ select: "Pair", run: "Corpus & query", output: "Four rankings" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={RANKING_PAIRS}
            value={pairEntry.id}
            onChange={(next) => {
              session.setModel(next);
              // Vectors from another checkpoint are not comparable with these.
              clear();
            }}
            disabled={loading || running}
            cached={session.cached}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Two models: <strong>{pair.embedder.label}</strong> embeds the corpus
            once, and <strong>{pair.reranker.label}</strong> scores the shortlist
            a query at a time. The size above is the{" "}
            <strong>combined</strong> download — both are live at once, which is
            a deliberate exception to this app's one-model-at-a-time rule.
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
          disabledHint="Load the pair to use the neural stages. BM25 below already works — it needs no model."
          controls={
            <>
              <Button
                variant="outline"
                disabled={!ready || running || corpus.length === 0}
                data-testid="embed-corpus"
                onClick={() => void embedCorpus(corpus).catch(() => {})}
              >
                {embedding ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Embedding{" "}
                    {embedding.done}/{embedding.total}…
                  </>
                ) : (
                  <>Embed the corpus ({corpus.length} passes)</>
                )}
              </Button>
              <Button
                disabled={!ready || running || corpus.length === 0 || !query.trim()}
                data-testid="search"
                onClick={() => void runSearch().catch(() => {})}
              >
                {running && !embedding ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Searching…
                  </>
                ) : (
                  <>
                    <Search className="size-4" /> Search
                  </>
                )}
              </Button>
              <span
                data-testid="cost-note"
                className="basis-full text-xs text-muted-foreground"
              >
                Embedding is <strong>{corpus.length}</strong> forward passes,
                once — every query after it is a dot product. Search costs 1 for
                the query plus <strong>{rerankCount}</strong> for the rerank,
                because a cross-encoder scores a <em>pair</em> and cannot be
                precomputed. {embedded > 0 && `${embedded} vectors cached.`}
              </span>
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <div className="space-y-1">
              <label htmlFor="tr-query" className="text-xs font-medium">
                Query
              </label>
              <input
                id="tr-query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
            </div>

            <div className="space-y-1">
              <label htmlFor="tr-corpus" className="text-xs font-medium">
                Corpus — one document per line ({corpus.length})
              </label>
              <textarea
                id="tr-corpus"
                value={corpusText}
                onChange={(e) => setCorpusText(e.target.value)}
                rows={8}
                className="min-h-32 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 font-mono text-xs"
              />
            </div>

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Queries — picking one fills the field and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {RANKING_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => setQuery(s.query)}
                    title={s.hint}
                  >
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* BM25 needs no model, so its parameters are live from the first
                render — and they re-score rather than re-run. */}
            <div className="space-y-2 border-t pt-3">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                <p className="text-xs font-medium">
                  BM25 parameters — no model, no cost
                </p>
                <span className="text-xs text-muted-foreground">
                  Re-scores as you drag. Runs nothing.
                </span>
              </div>
              <div className="grid gap-x-4 gap-y-2 sm:grid-cols-3">
                <Range
                  id="tr-k1"
                  label="k1 (saturation)"
                  value={bm25.k1}
                  min={0}
                  max={3}
                  step={0.1}
                  onChange={(v) => setBm25((p) => ({ ...p, k1: v }))}
                />
                <Range
                  id="tr-b"
                  label="b (length norm.)"
                  value={bm25.b}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => setBm25((p) => ({ ...p, b: v }))}
                />
                <Range
                  id="tr-rrf"
                  label="RRF k"
                  value={rrfK}
                  min={1}
                  max={200}
                  step={1}
                  onChange={setRrfK}
                />
              </div>
            </div>

            {/* The lexical stage is useful before anything is downloaded, so it
                is shown here rather than withheld until OUTPUT has four
                columns. */}
            <div className="space-y-1 border-t pt-3">
              <p className="text-xs font-medium">
                BM25 right now (no model loaded needed)
              </p>
              <ol
                data-testid="bm25-live"
                className="space-y-0.5 text-xs text-muted-foreground"
              >
                {bm25Ranked.slice(0, 3).map((doc, i) => (
                  <li key={doc} className="truncate">
                    {i + 1}. {corpus[doc]}
                  </li>
                ))}
                {bm25Ranked.length === 0 && <li>Type a query and a corpus.</li>}
              </ol>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Four rankings"
          description="How the four stages disagree is the point — a single fused answer hides it, so all four are shown with the movement between them."
          meta={ran ? <span>{ran.pairLabel}</span> : undefined}
          running={running}
          runningLabel={embedding ? "Embedding the corpus…" : "Searching…"}
          error={runError}
          empty="Embed the corpus, then press Search — the four stages appear side by side, with each document's movement between them."
        >
          {ran && (
            <div className="space-y-4" data-testid="rankings">
              <p className="text-xs text-muted-foreground italic">
                “{ran.query}”
              </p>
              <div className="grid gap-4 xl:grid-cols-2">
                <Column
                  title="1 · BM25"
                  note="Keyword overlap. No model."
                  testId="col-bm25"
                  order={capturedBm25}
                  corpus={ran.corpus}
                />
                <Column
                  title="2 · Dense"
                  note="Cosine between embeddings."
                  testId="col-dense"
                  order={ran.dense}
                  corpus={ran.corpus}
                  previous={capturedBm25}
                />
                <Column
                  title="3 · Hybrid (RRF)"
                  note={`Rank fusion of 1 and 2, k=${rrfK}. No model.`}
                  testId="col-hybrid"
                  order={hybrid}
                  corpus={ran.corpus}
                  previous={ran.dense}
                />
                <Column
                  title="4 · Reranked"
                  note={`Cross-encoder, ${ran.reranked.length} passes.`}
                  testId="col-rerank"
                  order={ran.reranked}
                  corpus={ran.corpus}
                  previous={hybrid}
                  scores={ran.rerankScores}
                />
              </div>
              <p className="text-xs leading-snug text-muted-foreground">
                Columns 1 and 3 are arithmetic — no model ran for either.
                Column 2 is one forward pass for the query against vectors
                already in hand. Column 4 is{" "}
                <strong>{ran.reranked.length} forward passes</strong>, one per
                candidate, because a cross-encoder scores a query and a document
                <em>together</em> and so cannot be precomputed. That asymmetry
                is the architecture, and it is why reranking is applied to a
                shortlist rather than to a corpus.
              </p>
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/** One stage's ranking, with each document's movement from the previous stage. */
function Column({
  title,
  note,
  testId,
  order,
  corpus,
  previous,
  scores,
}: {
  title: string;
  note: string;
  testId: string;
  order: readonly number[];
  corpus: readonly string[];
  /** The stage before this one, for the movement arrows. */
  previous?: readonly number[];
  scores?: Map<number, number>;
}) {
  const before = previous ? rankOf(previous) : null;
  return (
    <div className="space-y-1.5" data-testid={testId}>
      <div className="space-y-0.5">
        <p className="text-xs font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{note}</p>
      </div>
      <ol className="space-y-1">
        {order.slice(0, RANK_SHOW).map((doc, i) => {
          const was = before?.get(doc);
          const moved = was == null ? null : was - i;
          return (
            <li key={doc} className="flex gap-1.5 text-xs leading-snug">
              <span className="w-4 shrink-0 text-right tabular-nums text-muted-foreground">
                {i + 1}
              </span>
              {moved != null && moved !== 0 && (
                <span
                  className={
                    moved > 0
                      ? "shrink-0 tabular-nums text-emerald-600 dark:text-emerald-500"
                      : "shrink-0 tabular-nums text-amber-600 dark:text-amber-500"
                  }
                  title={`Moved ${Math.abs(moved)} place${Math.abs(moved) === 1 ? "" : "s"} ${moved > 0 ? "up" : "down"}`}
                >
                  {moved > 0 ? "▲" : "▼"}
                  {Math.abs(moved)}
                </span>
              )}
              <span className="min-w-0 flex-1 truncate" title={corpus[doc]}>
                {corpus[doc]}
              </span>
              {scores && (
                <span className="shrink-0 font-mono tabular-nums text-muted-foreground">
                  {(scores.get(doc) ?? 0).toFixed(2)}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** A labelled range with its value shown. */
function Range({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="text-xs font-medium">
        {label}: <span className="tabular-nums">{value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
