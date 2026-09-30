// The wiring between a task hook and the SELECT and LOAD slots — written once.
//
// Every task route used to spell out the same three things by hand: split the
// hook's one `error` into the slot that produced it (§4), pass ten fields from
// the hook into `ModelStatus`, and pass the catalogue and the selection into
// `ModelPicker`. Thirty-odd copies of that are thirty-odd places for one page to
// quietly send a load failure to OUTPUT.
//
// What it deliberately leaves visible: `loadError` and `runError` come back as
// named fields, because "errors render where they came from" is a rule the
// route has to be seen to follow, and a route with a third error source (a
// failed decode belongs to RUN) still passes that one itself. The props objects
// are spread, so a route that needs one field different overrides it after the
// spread — `<ModelPicker {...slots.picker} onChange={…} />` — rather than
// abandoning the helper.
//
// It also re-probes the cache on `ready`, which every route did by calling
// `useCacheRefresh` on the next line and nothing enforced.

import type { ComponentProps } from "react";

import type { PickableModel } from "@/components/model/ModelPicker";
import type { ModelStatus as ModelStatusView } from "@/components/model/ModelStatus";
import type { Backend } from "@/model/backend";
import type { ModelStatus } from "@/model/types";

import { useCacheRefresh, type UseModelSelectionResult } from "./useModelSelection";

/** The part of a task hook's result the two setup slots read. */
export interface TaskSlotSource {
  status: ModelStatus;
  loading: boolean;
  ready: boolean;
  backend: string | null;
  loadProgress: ComponentProps<typeof ModelStatusView>["loadProgress"];
  loadedInMs: number | null;
  error: string | null;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export interface TaskSlotOptions<T> {
  /** The catalogue the picker offers. */
  models: readonly T[];
  /**
   * Something else is in flight — a run, or an input being decoded. Shuts the
   * LOAD button and, with `loading`, the picker.
   */
  busy?: boolean;
  /**
   * The backend a load would resolve to, from `useBackendProbe`. When given,
   * the picker gates rows the machine cannot run.
   */
  backend?: Backend | null;
}

/** Everything `ModelPicker` takes, bound to one catalogue type. */
export interface TaskPickerProps<T extends PickableModel> {
  models: readonly T[];
  value: string;
  onChange: (model: T) => void;
  disabled: boolean;
  cached: ReadonlySet<string>;
  onEvict: (model: T) => void;
  backend?: Backend | null;
}

export interface TaskSlots<T extends PickableModel> {
  /** A failed download — renders in LOAD. */
  loadError: string | null;
  /** A failed inference — renders in OUTPUT, with the model still loaded. */
  runError: string | null;
  /** Spread onto `<ModelStatus />`. */
  status: ComponentProps<typeof ModelStatusView>;
  /** Spread onto `<ModelPicker />`. */
  picker: TaskPickerProps<T>;
}

export function useTaskSlots<T extends PickableModel>(
  session: UseModelSelectionResult<T>,
  task: TaskSlotSource,
  { models, busy = false, backend }: TaskSlotOptions<T>,
): TaskSlots<T> {
  useCacheRefresh(session, task.ready);

  const loadError = task.status === "error" ? task.error : null;
  const runError = task.status === "error" ? null : task.error;

  const picker: TaskPickerProps<T> = {
    models,
    value: session.model.id,
    onChange: session.setModel,
    disabled: task.loading || busy,
    cached: session.cached,
    onEvict: (m) => void session.evict(m.id),
  };
  // Only when asked: `backend: null` means "the probe has not answered", which
  // the picker treats differently from not probing at all.
  if (backend !== undefined) picker.backend = backend;

  return {
    loadError,
    runError,
    status: {
      status: task.status,
      backend: task.backend,
      loadProgress: task.loadProgress,
      loadedInMs: task.loadedInMs,
      cached: session.isCached,
      error: loadError,
      onLoad: task.load,
      onCancel: task.cancel,
      onRetry: task.retry,
      disabled: busy,
    },
    picker,
  };
}
