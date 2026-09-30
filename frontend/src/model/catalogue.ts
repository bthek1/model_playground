// What every model catalogue entry is, whatever its modality.
//
// Vision and text each declared `id`/`label`/`hint`/`params`/`bytes`/`backends`
// separately, audio's entries extended nothing, and the one shared description
// was `ModelPicker`'s `PickableModel` — a structural match that nothing
// declared. This is that shape, named once, in `model/` beside the size and
// backend code that reads it. A modality's entry type extends it and adds the
// task-specific fields; it may narrow a field (text requires `bytes`) but never
// redefine one.

import { useMemo } from "react";

import type { Backend } from "./backend";
import type { MeasuredBytes } from "./size";

export interface CatalogueEntry {
  /** The Hub repo id — or, for a pair, a composite id no cache holds. */
  id: string;
  label: string;
  /** One line on why you would pick this one. */
  hint: string;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
  /** Measured download bytes, where the params estimate would mislead. */
  bytes?: MeasuredBytes;
  /**
   * Backends this model is known to run on. Omitted means both. List one when
   * the other is a known failure rather than merely slower, so `ModelPicker`
   * can gate the row instead of the load failing after the download.
   */
  backends?: readonly Backend[];
}

/**
 * The entry for `id`, or the catalogue's first when `id` is not in it — a
 * stale id must degrade to the default rather than wedge the hook.
 */
export function findEntry<T extends { id: string }>(
  models: readonly T[],
  id: string,
): T {
  return models.find((m) => m.id === id) ?? models[0];
}

/** `findEntry`, memoised on the id — the identity the task hooks key on. */
export function useCatalogueEntry<T extends { id: string }>(
  models: readonly T[],
  id: string,
): T {
  return useMemo(() => findEntry(models, id), [models, id]);
}
