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
import {
  locateEntities,
  type EntitySpan,
  type LocatableEntity,
} from "@/text/highlight";

/** What the pipeline returns per entity with `aggregation_strategy: "simple"`. */
interface RawEntity {
  entity_group?: string;
  entity?: string;
  score: number;
  /** Optional in the runtime's own types, and `undefined` in 4.2.0. */
  start?: number;
  end?: number;
  word?: string;
}

export interface UseNerResult extends Omit<UseTextPipelineResult, "run"> {
  /** Latest spans, in the order the model returned them. */
  result: EntitySpan[] | null;
  /**
   * Entities the model reported but that could not be placed in the source.
   * Surfaced rather than swallowed — see `locateEntities`.
   */
  unplaced: LocatableEntity[];
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
  const [unplaced, setUnplaced] = useState<LocatableEntity[]>([]);

  const run = useCallback(
    async (text: string): Promise<EntitySpan[]> => {
      const out = (await post(text)) as RawEntity[];
      const raw = Array.isArray(out) ? out : [];

      // **The pipeline does not return character offsets.** 4.2.0 ships
      // `token-classification` with the work unwritten and the fields declared
      // optional, so `e.start` type-checks and is `undefined` at runtime —
      // which drops every span and renders a page that looks like a model that
      // found nothing. Use the offsets when a future version does provide them,
      // and recover them from the source when it does not.
      const withOffsets = raw.filter(
        (e) => Number.isInteger(e.start) && Number.isInteger(e.end),
      );
      if (withOffsets.length === raw.length && raw.length > 0) {
        const spans = raw.map(toSpan);
        setResult(spans);
        setUnplaced([]);
        return spans;
      }

      const located = locateEntities(
        text,
        raw.map((e) => ({
          word: e.word ?? "",
          label: labelOf(e),
          score: e.score,
        })),
      );
      setResult(located.spans);
      setUnplaced(located.unplaced);
      return located.spans;
    },
    [post],
  );

  return { ...pipe, result, unplaced, run };
}

/**
 * The entity type, whichever shape the runtime used.
 *
 * Aggregated results carry `entity_group` (`PER`); unaggregated ones carry
 * `entity` with the BIO prefix still on it (`B-PER`). The prefix is stripped
 * either way, so the page's type names, colour slots and redact choices are the
 * same whichever arrives — rather than `B-PER` and `I-PER` becoming two types.
 */
function labelOf(e: RawEntity): string {
  return (e.entity_group ?? e.entity ?? "MISC").replace(/^[BIOES]-/, "");
}

/** Normalise an entity that already carries offsets, taking them as given. */
function toSpan(e: RawEntity): EntitySpan {
  return {
    start: e.start as number,
    end: e.end as number,
    label: labelOf(e),
    score: e.score,
  };
}
