// Reciprocal rank fusion — the second stage of `/text-ranking` with no model in
// it, and the cheapest useful idea in retrieval.
//
// It fuses rank lists rather than scores, which is the whole trick: BM25 scores
// and cosine similarities are on incomparable scales, so any weighted sum of
// them is a scale choice dressed up as a method. RRF only needs the *order*.
//
//   score(d) = Σ over lists of 1 / (k + rank(d))
//
// `k` damps the contribution of the top ranks. At `k = 0` the first place in one
// list is worth infinitely more than second; the conventional 60 makes the top
// dozen or so matter and the tail count for little.
//
// **One property RRF does *not* have, and it is worth knowing before reading a
// fused list.** It is tempting to assume that a document both lists rank second
// beats one that is first in one list and last in the other — "consistency
// wins". It does not: `1 / (k + r)` is **convex**, so by Jensen's inequality the
// extreme pair scores *higher*. Concretely, at `k = 60` over two lists:
//
//   first + last    1/60 + 1/62  =  0.032796
//   second + second 1/61 + 1/61  =  0.032787
//
// The margin is tiny and shrinks as `k` grows, but the sign is fixed — RRF
// rewards being loved by one retriever over being liked by both. That is a
// defensible thing to want (a specialist's top hit is evidence), it is simply
// not the behaviour the name suggests, and a page showing four rank lists side
// by side will display it. The tests pin the real ordering rather than the
// assumed one.
//
// Pure, so changing `k` re-fuses from held rank lists and spends nothing.

/** A rank list: document indices, best first. */
export type RankList = readonly number[];

/** The conventional constant, from the original TREC paper. */
export const DEFAULT_RRF_K = 60;

export interface Fused {
  doc: number;
  score: number;
  /** Rank in each input list (0-based), or null where the list omitted it. */
  ranks: (number | null)[];
}

/**
 * Fuse N rank lists.
 *
 * A document missing from a list contributes **nothing** from it rather than a
 * worst-case rank: the lists here are top-k truncations, so "absent" means "not
 * in this stage's shortlist", not "ranked last". Treating absence as last place
 * would let a long list outvote a short one purely by being longer.
 */
export function fuse(lists: readonly RankList[], k = DEFAULT_RRF_K): Fused[] {
  const ranks = new Map<number, (number | null)[]>();
  const blank = () => lists.map(() => null as number | null);

  lists.forEach((list, listIndex) => {
    list.forEach((doc, rank) => {
      const row = ranks.get(doc) ?? blank();
      // First appearance wins if a list repeats a document — a duplicate is a
      // bug upstream, and counting it twice would silently reward it.
      if (row[listIndex] == null) row[listIndex] = rank;
      ranks.set(doc, row);
    });
  });

  const fused: Fused[] = [];
  for (const [doc, row] of ranks) {
    let score = 0;
    for (const rank of row) {
      if (rank == null) continue;
      score += 1 / (k + rank);
    }
    fused.push({ doc, score, ranks: row });
  }

  // Ties keep the lowest document index, so the fusion is deterministic.
  fused.sort((a, b) => b.score - a.score || a.doc - b.doc);
  return fused;
}

/** Rank of each document in a list, for the movement arrows between stages. */
export function rankOf(list: RankList): Map<number, number> {
  const out = new Map<number, number>();
  list.forEach((doc, i) => {
    if (!out.has(doc)) out.set(doc, i);
  });
  return out;
}
