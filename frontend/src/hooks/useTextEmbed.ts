// Sentence embeddings — one hook behind three routes (`/text-features`,
// `/sentence-similarity`, and the dense stage of `/text-ranking`), because an
// embedding is an embedding and the pages differ only in what they do with the
// vectors afterwards.
//
// Three decisions live here rather than in the routes:
//
//  1. **The pooling comes from the catalogue, not from the caller.** It is a
//     property of the checkpoint (`EmbedModel.pooling`), and a page that passed
//     its own would silently produce a plausible wrong vector on two of the five
//     entries. This is the single call site, which is the same argument
//     `pinnedArgs` makes in the engine.
//  2. **Embed once, keyed by the exact text.** `/text-ranking` pastes a corpus
//     through this and then queries it repeatedly; re-embedding a document
//     because a neighbouring one changed is the cost this page is supposed to be
//     demonstrating away. The cache is `text/embed.ts`'s, so it can be unit
//     tested without a worker.
//  3. **A model change clears the cache.** A 384-d MiniLM vector and a 768-d BGE
//     vector are not comparable, and neither are two 768-d vectors from
//     different checkpoints — and a stale hit is a confident wrong number with
//     no error attached, which is why it is asserted directly rather than
//     assumed.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import type { PlainTensor } from "@/model/serialize";
import {
  DEFAULT_EMBED_MODEL,
  EMBED_MODELS,
  type EmbedModel,
} from "@/text/catalogue";
import { EmbeddingCache, toVector } from "@/text/embed";

/** Which task instruction to glue to the front, for a model that wants one. */
export type PrefixKind = "symmetric" | "query" | "document";

export interface UseTextEmbedResult extends Omit<UseTextPipelineResult, "run"> {
  /** The selected catalogue entry — `dim`, `pooling` and `prefixes` come from it. */
  meta: EmbedModel;
  /**
   * Embed one text, or a batch of them, returning one unit vector each.
   *
   * A batch is one pipeline call — the tokenizer pads, the model runs once — so
   * embedding a 40-line corpus is one request, not forty. Texts already in the
   * cache are **not** re-sent: only the misses go to the worker, and the result
   * is reassembled in the caller's order.
   */
  run: (texts: string | string[], kind?: PrefixKind) => Promise<Float32Array[]>;
  /** How many vectors are currently remembered. Shown, so the cache is visible. */
  cached: number;
  /** Forget every vector — exposed for the routes that hold a corpus. */
  clearCache: () => void;
  /** The exact string that would be sent for `text`, prefix included. */
  compose: (text: string, kind?: PrefixKind) => string;
}

export function useTextEmbed(
  model: string = DEFAULT_EMBED_MODEL,
  autoLoad = false,
): UseTextEmbedResult {
  const meta = useMemo(
    () => EMBED_MODELS.find((m) => m.id === model) ?? EMBED_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  // A ref, not state: a cache hit must not re-render, and the cache is not
  // something the page renders *from* — only its size is shown.
  const cache = useRef(new EmbeddingCache());
  const [cached, setCached] = useState(0);

  const clearCache = useCallback(() => {
    cache.current.clear();
    setCached(0);
  }, []);

  // Keyed on the id rather than on `meta`, so a catalogue object identity change
  // cannot drop a corpus the user is still working with.
  useEffect(() => {
    clearCache();
  }, [meta.id, clearCache]);

  const compose = useCallback(
    (text: string, kind: PrefixKind = "symmetric"): string =>
      `${meta.prefixes?.[kind] ?? ""}${text}`,
    [meta],
  );

  const run = useCallback(
    async (
      texts: string | string[],
      kind: PrefixKind = "symmetric",
    ): Promise<Float32Array[]> => {
      const list = Array.isArray(texts) ? texts : [texts];
      // The cache is keyed on the **composed** string, because the prefix is
      // part of what the model saw: the same sentence embedded as a query and
      // as a document is two different vectors, and sharing one entry between
      // them is the kind of wrong answer that looks like a working cache.
      const composed = list.map((t) => compose(t, kind));
      const missing = [...new Set(composed.filter((t) => !cache.current.has(t)))];

      if (missing.length > 0) {
        const raw = (await post(missing, [{ pooling: meta.pooling }])) as
          | PlainTensor
          | undefined;
        const vectors = splitBatch(raw, missing.length);
        missing.forEach((text, i) => cache.current.set(text, vectors[i]));
        setCached(cache.current.size);
      }

      return composed.map((text) => {
        const vector = cache.current.get(text);
        if (!vector) throw new Error("Embedding missing after a successful run");
        return vector;
      });
    },
    [post, compose, meta.pooling],
  );

  return { ...pipe, meta, run, cached, clearCache, compose };
}

/**
 * Split a batched `[n, dim]` tensor into n vectors.
 *
 * A single-element batch comes back as `[1, dim]`, so `toVector` handles it
 * directly; anything wider is sliced row by row. The row count is checked
 * against what was asked for rather than inferred, because a mismatch means the
 * vectors would be **attributed to the wrong texts** — every score after that
 * is confidently wrong, and a length check is the only place it is visible.
 */
function splitBatch(tensor: PlainTensor | undefined, expected: number): Float32Array[] {
  if (!tensor) throw new Error("The model returned no embedding");
  const dims = tensor.dims ?? [];
  if (expected === 1 && dims.length <= 2) return [toVector(tensor)];

  if (dims.length !== 2) {
    throw new Error(
      `Expected a [batch, dim] embedding for ${expected} texts, got [${dims.join(", ")}]`,
    );
  }
  const [rows, dim] = dims;
  if (rows !== expected) {
    throw new Error(`Asked for ${expected} embeddings, got ${rows}`);
  }
  const data = tensor.data;
  const out: Float32Array[] = [];
  for (let r = 0; r < rows; r++) {
    const vector = new Float32Array(dim);
    for (let d = 0; d < dim; d++) vector[d] = data[r * dim + d];
    out.push(vector);
  }
  return out;
}
