// Named-entity recognition. A thin wrapper over the generic `useTextPipeline`
// worker — it picks the catalogue entry and types the result. Everything else
// (worker lifecycle, the pending table, the two state machines) belongs to
// `model/useModelWorker.ts` and is not re-derived here.
//
// `aggregation_strategy: "simple"` is **not** passed from here: it is pinned in
// the engine, because forgetting it is a rendering bug rather than an error and
// one call site is easier to keep right than every future one. See
// `text/engine.ts` → `pinnedArgs`.

import { useCallback, useMemo, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import { DEFAULT_NER_MODEL, NER_MODELS } from "@/text/catalogue";
import type { EntitySpan } from "@/text/highlight";

/** What the pipeline returns per entity with `aggregation_strategy: "simple"`. */
interface RawEntity {
  entity_group?: string;
  entity?: string;
  score: number;
  start: number;
  end: number;
  word?: string;
}

export interface UseNerResult extends Omit<UseTextPipelineResult, "run"> {
  /** Latest spans, in the order the model returned them. */
  result: EntitySpan[] | null;
  run: (text: string) => Promise<EntitySpan[]>;
}

export function useNer(
  model: string = DEFAULT_NER_MODEL,
  autoLoad = false,
): UseNerResult {
  const meta = useMemo(
    () => NER_MODELS.find((m) => m.id === model) ?? NER_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<EntitySpan[] | null>(null);

  const run = useCallback(
    async (text: string): Promise<EntitySpan[]> => {
      const out = (await post(text)) as RawEntity[];
      const spans = (Array.isArray(out) ? out : []).map(toSpan);
      setResult(spans);
      return spans;
    },
    [post],
  );

  return { ...pipe, result, run };
}

/**
 * Normalise one pipeline entity into a `EntitySpan`.
 *
 * Aggregated results carry `entity_group` (`PER`); unaggregated ones carry
 * `entity` with the BIO prefix still on it (`B-PER`). The prefix is stripped
 * either way so the page's type names, colour slots and redact choices are the
 * same whichever shape arrives — and **the offsets are taken as given**, never
 * recomputed from `word`, which is the token-rebuild bug.
 */
function toSpan(e: RawEntity): EntitySpan {
  const raw = e.entity_group ?? e.entity ?? "MISC";
  return {
    start: e.start,
    end: e.end,
    label: raw.replace(/^[BIOES]-/, ""),
    score: e.score,
  };
}
