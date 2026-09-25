// BM25 — the retrieval stage with no model in it at all, and the reason
// `/text-ranking` can show four stages rather than two.
//
// It is here rather than inline in the route because the two properties that
// make BM25 *BM25* are both invisible on screen and both easy to get subtly
// wrong, so they are pinned by tests instead:
//
//   **A term in every document contributes nothing.** That is what the IDF
//   factor is for. Implemented with the standard `log(1 + (N - df + 0.5) /
//   (df + 0.5))` form rather than the textbook `log((N - df + 0.5) /
//   (df + 0.5))`, which goes **negative** for a term in more than half the
//   corpus — so a document containing a common word would score *worse* than
//   one that does not, and the ranking would be quietly inverted for exactly
//   the stopword-ish terms a user is most likely to type.
//
//   **A longer document must not win on length alone.** That is `b`: the term
//   frequency is normalised by the document's length against the corpus
//   average. With `b = 0` a long document accumulates matches for free.
//
// Everything here is pure and synchronous, so `k1`, `b` and the query re-score
// from a held index on the main thread — the same rule `/vad`'s threshold and
// the truncation control follow. Only "embed corpus" and "rerank" spend.

/** Free parameters. The usual defaults, and both are re-derivable controls. */
export interface Bm25Params {
  /** Term-frequency saturation. 0 makes it binary; ~1.2–2.0 is conventional. */
  k1: number;
  /** Length normalisation, 0–1. 0 is off, 1 is full. */
  b: number;
}

export const DEFAULT_BM25: Bm25Params = { k1: 1.2, b: 0.75 };

/**
 * Split text into lowercase word tokens.
 *
 * Deliberately crude — no stemming, no stopword list. Stemming would make the
 * lexical stage quietly *less* lexical, which is the opposite of what this page
 * compares it for: the whole point is that BM25 matches words and the dense
 * retriever matches meanings. A stemmer blurs the line the page is drawing.
 */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 0);
}

/** The corpus, prepared once: one pass over the documents, then queries are cheap. */
export interface Bm25Index {
  /** Token lists, one per document, in the caller's order. */
  docs: string[][];
  /** Document lengths in tokens. */
  lengths: number[];
  /** Mean document length — the denominator of the length normalisation. */
  avgLength: number;
  /** How many documents contain each term. */
  df: Map<string, number>;
  size: number;
}

/** Build the index. An empty corpus is a valid index that scores nothing. */
export function buildIndex(corpus: readonly string[]): Bm25Index {
  const docs = corpus.map(tokenize);
  const lengths = docs.map((d) => d.length);
  const total = lengths.reduce((a, b) => a + b, 0);
  const df = new Map<string, number>();
  for (const doc of docs) {
    // A term counts once per document, however often it appears in it.
    for (const term of new Set(doc)) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }
  return {
    docs,
    lengths,
    // Guard the divide: an empty corpus, or one of empty strings, must not
    // produce NaN scores. A NaN ranks unpredictably rather than failing.
    avgLength: docs.length > 0 && total > 0 ? total / docs.length : 1,
    df,
    size: docs.length,
  };
}

/**
 * Inverse document frequency, in the form that never goes negative.
 *
 * `log(1 + (N - df + 0.5) / (df + 0.5))`. A term present in every document
 * gives `log(1 + 0.5/(N+0.5))`, which tends to 0 from above — it contributes
 * nothing, which is correct, rather than a negative penalty.
 */
export function idf(index: Bm25Index, term: string): number {
  const df = index.df.get(term) ?? 0;
  const n = index.size;
  return Math.log(1 + (n - df + 0.5) / (df + 0.5));
}

export interface Scored {
  /** Index into the corpus the index was built from. */
  doc: number;
  score: number;
}

/**
 * Score every document against `query`, best first.
 *
 * A query term absent from the whole corpus contributes 0 rather than being an
 * error: a user types words the corpus does not have, and an empty result list
 * is the honest answer to that.
 */
export function search(
  index: Bm25Index,
  query: string,
  { k1, b }: Bm25Params = DEFAULT_BM25,
): Scored[] {
  const terms = tokenize(query);
  const scored: Scored[] = index.docs.map((doc, i) => {
    if (terms.length === 0 || doc.length === 0) return { doc: i, score: 0 };

    // Term frequencies for this document, once.
    const tf = new Map<string, number>();
    for (const t of doc) tf.set(t, (tf.get(t) ?? 0) + 1);

    const norm = k1 * (1 - b + (b * index.lengths[i]) / index.avgLength);
    let score = 0;
    for (const term of terms) {
      const f = tf.get(term);
      if (!f) continue;
      score += idf(index, term) * ((f * (k1 + 1)) / (f + norm));
    }
    return { doc: i, score };
  });

  // Ties keep corpus order, so a re-score at the same parameters is stable — a
  // list that reshuffles equal scores between renders reads as flicker.
  return scored.sort((a, b2) => b2.score - a.score || a.doc - b2.doc);
}
