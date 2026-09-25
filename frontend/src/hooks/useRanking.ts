// `/text-ranking` — all four retrieval stages, two of which need no model.
//
// This hook is where the page's cost model lives, and it has exactly two
// spending operations:
//
//   **Embedding the corpus** is N forward passes, once. Every subsequent query
//   is then a dot product against vectors already in hand, which is the claim
//   dense retrieval actually makes and the reason it is worth its download.
//   `useTextEmbed`'s cache is keyed on the exact composed string, so a corpus
//   re-submitted with one line changed re-embeds **one line**.
//
//   **Reranking** is one forward pass per candidate. A cross-encoder scores
//   `{ text, text_pair }` — a *pair*, not a concatenation — so unlike an
//   embedding it cannot be precomputed per document, which is exactly why it
//   reranks a shortlist rather than the corpus. That cost is visible in a
//   browser, and making it visible is the argument for the page.
//
// BM25 and RRF spend nothing at all and are not here: they are pure functions
// over a held index (`text/bm25.ts`, `text/rrf.ts`) and the route calls them
// directly, so `k1`, `b` and RRF's `k` re-score on the main thread.
//
// **Two models are live at once**, which is the declared exception `/pose`
// established — but not the way `/pose` does it, and the difference is the
// reason `combineProgress` exists. `/pose` loads its pair inside **one** worker,
// so one engine owns both models and a combined teardown is something it can
// get wrong; the plan's `Promise.allSettled` bullet was written for that shape.
// Here each half is **its own worker**, so the engine rule ("one model live at a
// time") holds per worker unmodified, each engine disposes its single model with
// `disposeQuietly`, and `useModelWorker` terminates each worker independently.
// There is no shared teardown to `allSettled` over: a dispose that throws in one
// worker cannot reach the other, which is the property `allSettled` was wanted
// for, obtained structurally instead. The cost of the split is the progress
// table — two of them, neither of which can report the pair — which is what
// `combineProgress` is for.

import { useCallback, useEffect, useMemo, useState } from "react";

import { useTextEmbed } from "@/hooks/useTextEmbed";
import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import { combineProgress } from "@/model/progress";
import { pickBackend, type Backend } from "@/model/backend";
import { cosine } from "@/model/similarity";
import type { ModelStatus } from "@/model/types";
import {
  DEFAULT_RANKING_PAIR,
  RANKING_PAIRS,
  RERANK_TOP_K,
  type RankingPair,
} from "@/text/catalogue";

/** One cross-encoder score for a (query, document) pair. */
export interface RerankScore {
  /** Index into the corpus. */
  doc: number;
  score: number;
}

/** What a cross-encoder returns per pair: one or more `{ label, score }`. */
interface RawCrossEncoder {
  label?: string;
  score?: number;
}

export interface UseRankingResult {
  pair: RankingPair;
  /** Combined status: `ready` only when **both** models are loaded. */
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  /** Load both halves. One button, because the pair is what was chosen. */
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
  /**
   * **One** bar across both downloads, not whichever half is loudest.
   *
   * Each worker keeps its own progress table, so showing either one fills the
   * bar to 100% and then restarts it — the "100% halfway through a pair" failure
   * `/pose` documents, reached from the other direction. `combineProgress` sums
   * the bytes against the catalogue entry's **measured combined size**, which is
   * a constant and therefore monotonic by construction.
   */
  loadProgress: UseTextPipelineResult["loadProgress"];
  loadedInMs: number | null;
  backend: string | null;
  /** Either half's load error, or the most recent run error. */
  error: string | null;

  /** True while either model is running. */
  running: boolean;
  /** Non-null while the corpus is being embedded. */
  embedding: { done: number; total: number } | null;
  /** How many corpus documents currently have a vector. */
  embedded: number;

  /** Embed every document. N forward passes; cached ones are skipped. */
  embedCorpus: (corpus: readonly string[]) => Promise<void>;
  /** Embed the query and score it against the corpus. One forward pass. */
  denseSearch: (
    query: string,
    corpus: readonly string[],
  ) => Promise<RerankScore[]>;
  /** Score a shortlist with the cross-encoder. One pass per candidate. */
  rerank: (
    query: string,
    corpus: readonly string[],
    candidates: readonly number[],
  ) => Promise<RerankScore[]>;
  /** Drop every vector — the only correct response to a pair change. */
  clear: () => void;
}

export function useRanking(
  pairId: string = DEFAULT_RANKING_PAIR,
  autoLoad = false,
): UseRankingResult {
  const pair = useMemo(
    () => RANKING_PAIRS.find((p) => p.id === pairId) ?? RANKING_PAIRS[0],
    [pairId],
  );

  // The embedder rides the shared embedding hook, so the pooling, the prefix
  // handling and the text-keyed cache are the ones `/sentence-similarity`
  // already uses rather than a second implementation.
  const embed = useTextEmbed(pair.embedder.id);
  // A cross-encoder *is* a sequence classifier — the pipeline task is
  // `text-classification` and the input is a `{ text, text_pair }` object,
  // which `TextInput` already describes. No new engine arm was needed, which is
  // worth knowing before writing one.
  const cross = useTextPipeline(
    "text-classification",
    pair.reranker.id,
    autoLoad,
    pair.reranker.dtypes,
  );

  const [embedding, setEmbedding] = useState<{
    done: number;
    total: number;
  } | null>(null);

  // Which backend the bar's denominator should come from. `null` until the
  // probe answers — the combined size differs by roughly 2x between them, and
  // guessing would make the bar wrong for the first seconds of every load.
  const [probe, setProbe] = useState<Backend | null>(null);
  useEffect(() => {
    let live = true;
    void pickBackend().then((b) => {
      if (live) setProbe(b);
    });
    return () => {
      live = false;
    };
  }, []);

  const combined = useMemo(
    () =>
      combineProgress(
        [embed.loadProgress, cross.loadProgress],
        pair.bytes[probe ?? "webgpu"] ?? 0,
        Math.max(
          embed.loadProgress?.elapsedMs ?? 0,
          cross.loadProgress?.elapsedMs ?? 0,
        ),
      ),
    [embed.loadProgress, cross.loadProgress, pair.bytes, probe],
  );

  // `ready` means **both**: a page that answered a query with half its pipeline
  // loaded would show three empty columns and one full one.
  const status: ModelStatus = useMemo(() => {
    if (embed.status === "error" || cross.status === "error") return "error";
    if (embed.ready && cross.ready) return "ready";
    if (embed.loading || cross.loading) return "loading";
    return "idle";
  }, [embed.status, embed.ready, embed.loading, cross.status, cross.ready, cross.loading]);

  const load = useCallback(() => {
    embed.load();
    cross.load();
  }, [embed, cross]);

  const retry = useCallback(
    (overrides?: Record<string, unknown>) => {
      if (embed.status === "error") embed.retry(overrides);
      if (cross.status === "error") cross.retry(overrides);
    },
    [embed, cross],
  );

  const cancel = useCallback(() => {
    embed.cancel();
    cross.cancel();
  }, [embed, cross]);

  const embedCorpus = useCallback(
    async (corpus: readonly string[]): Promise<void> => {
      const docs = corpus.filter((d) => d.trim().length > 0);
      setEmbedding({ done: 0, total: docs.length });
      try {
        // One batched call per chunk rather than one per document: the
        // tokenizer pads and the model runs the batch, so a 12-line corpus is
        // a couple of requests. Chunked rather than one giant batch because
        // padding to the longest document in a 200-line corpus wastes more
        // than the batching saves.
        const CHUNK = 16;
        for (let i = 0; i < docs.length; i += CHUNK) {
          await embed.run(docs.slice(i, i + CHUNK), "document");
          setEmbedding({ done: Math.min(i + CHUNK, docs.length), total: docs.length });
        }
      } finally {
        setEmbedding(null);
      }
    },
    [embed],
  );

  const denseSearch = useCallback(
    async (
      query: string,
      corpus: readonly string[],
    ): Promise<RerankScore[]> => {
      // The query is embedded with the **query** prefix where the checkpoint
      // wants one, and the documents with the document prefix — the same
      // sentence under two prefixes is two different vectors, which is why the
      // cache is keyed on the composed string.
      const [q] = await embed.run(query, "query");
      const docs = await embed.run([...corpus], "document");
      return corpus
        .map((_, i) => ({ doc: i, score: cosine(q, docs[i]) }))
        .sort((a, b) => b.score - a.score || a.doc - b.doc);
    },
    [embed],
  );

  const rerank = useCallback(
    async (
      query: string,
      corpus: readonly string[],
      candidates: readonly number[],
    ): Promise<RerankScore[]> => {
      const out: RerankScore[] = [];
      // Sequential, and one call per candidate: a cross-encoder scores a pair,
      // so there is nothing to precompute and nothing to batch away. This is
      // the page's whole point made measurable.
      for (const doc of candidates.slice(0, RERANK_TOP_K)) {
        const raw = (await cross.run({
          text: query,
          text_pair: corpus[doc],
        })) as RawCrossEncoder[] | RawCrossEncoder | undefined;
        const first = Array.isArray(raw) ? raw[0] : raw;
        out.push({ doc, score: Number(first?.score ?? 0) });
      }
      return out.sort((a, b) => b.score - a.score || a.doc - b.doc);
    },
    [cross],
  );

  const clear = useCallback(() => embed.clearCache(), [embed]);

  return {
    pair,
    status,
    idle: status === "idle",
    loading: status === "loading",
    ready: status === "ready",
    load,
    retry,
    cancel,
    loadProgress: combined,
    loadedInMs:
      embed.loadedInMs != null && cross.loadedInMs != null
        ? Math.max(embed.loadedInMs, cross.loadedInMs)
        : null,
    backend: embed.backend ?? cross.backend,
    error:
      (embed.status === "error" ? embed.error : null) ??
      (cross.status === "error" ? cross.error : null) ??
      embed.error ??
      cross.error,
    running: embed.running || cross.running,
    embedding,
    embedded: embed.cached,
    embedCorpus,
    denseSearch,
    rerank,
    clear,
  };
}
