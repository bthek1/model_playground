// Masked language modelling. A thin wrapper over the generic `useTextPipeline`
// worker — it picks the catalogue entry, pins `top_k`, sends the entry's mask
// literal alongside the text, and normalises the pipeline's output. Everything
// else (worker lifecycle, the pending table, the two state machines) belongs to
// `model/useModelWorker.ts` and is not re-derived here.
//
// `top_k` is a **run** parameter, not a load option: asking for more candidates
// is a different question about the same weights, so it must not imply another
// download. The mask literal is neither — it is a fact about the tokenizer, and
// the engine reconciles what the page sends against what it actually loaded.

import { useCallback, useMemo, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import type { ClassLabel } from "@/model/types";
import {
  DEFAULT_FILL_MASK,
  FILL_MASK_MODELS,
  FILL_TOP_K,
} from "@/text/catalogue";
import { cleanFill } from "@/text/mask";
import type { FillMaskResult, RawFilling } from "@/text/types";

export interface FillMaskOutcome {
  /**
   * The mask literal the **loaded tokenizer** uses, as the worker read it.
   *
   * Null only if the checkpoint has no mask token at all. The page compares it
   * against the catalogue's declaration and says so if they differ, rather than
   * quietly showing one and running the other.
   */
  mask: string | null;
  /** One ranked list per prompt, in the order the prompts were sent. */
  fills: ClassLabel[][];
}

export interface UseFillMaskResult
  extends Omit<UseTextPipelineResult, "run"> {
  /** The latest outcome. */
  result: FillMaskOutcome | null;
  /**
   * Fill one prompt, or a batch of them.
   *
   * A batch is one call and one forward pass per prompt — which is what makes
   * the bias probe a single GENERATE rather than six.
   */
  run: (prompt: string | string[]) => Promise<FillMaskOutcome>;
  /** The mask literal the catalogue declares for the selected model. */
  maskToken: string;
}

export function useFillMask(
  model: string = DEFAULT_FILL_MASK,
  autoLoad = false,
): UseFillMaskResult {
  const meta = useMemo(
    () => FILL_MASK_MODELS.find((m) => m.id === model) ?? FILL_MASK_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<FillMaskOutcome | null>(null);
  const maskToken = meta.maskToken;

  const run = useCallback(
    async (prompt: string | string[]): Promise<FillMaskOutcome> => {
      const raw = (await post(
        prompt,
        [{ top_k: FILL_TOP_K }],
        maskToken,
      )) as FillMaskResult;

      const outcome: FillMaskOutcome = {
        mask: raw?.mask ?? null,
        fills: toLists(raw?.fills, Array.isArray(prompt) ? prompt.length : 1),
      };
      setResult(outcome);
      return outcome;
    },
    [post, maskToken],
  );

  return { ...pipe, result, run, maskToken };
}

/**
 * Normalise the pipeline's output to one ranked list per prompt.
 *
 * `FillMaskPipeline` returns a flat list for a string and a nested one for an
 * array, so a caller that batches and a caller that does not would otherwise
 * read two different shapes. `expected` disambiguates the case a shape check
 * cannot: a one-element batch.
 */
function toLists(
  fills: FillMaskResult["fills"] | undefined,
  expected: number,
): ClassLabel[][] {
  if (!Array.isArray(fills)) return [];
  const nested = fills.length > 0 && Array.isArray(fills[0]);
  const lists = nested ? (fills as RawFilling[][]) : [fills as RawFilling[]];
  // A flat list arriving where a batch was asked for means the pipeline
  // answered a different question than the caller asked; surfacing an empty
  // list is better than silently labelling prompt 2's answer as prompt 1's.
  if (expected > 1 && !nested) return [];
  return lists.map(toScores);
}

/**
 * One raw candidate list as `{ label, score }`, which is what `ScoreList` takes.
 *
 * Two things happen here, both of which are visible on screen if they do not:
 * byte-level BPE leaves the word-initial space on its decode (` Paris`), and
 * trimming can collide two distinct token ids onto one label — `ScoreList` keys
 * its rows on the label, so a duplicate is dropped rather than rendered twice
 * under a React key clash. The list is already ranked, so the survivor is the
 * higher-scoring of the pair.
 */
function toScores(list: RawFilling[]): ClassLabel[] {
  const seen = new Set<string>();
  const out: ClassLabel[] = [];
  for (const f of Array.isArray(list) ? list : []) {
    const label = cleanFill(String(f?.token_str ?? ""));
    if (label.length === 0 || seen.has(label)) continue;
    seen.add(label);
    out.push({ label, score: Number(f?.score ?? 0) });
  }
  return out;
}
