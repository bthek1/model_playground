// Mask generation — SAM's encode-once/decode-many, driven from a page.
//
// The shared contract verbatim (`model/types.ts`), plus the two additions this
// task genuinely needs and no other route has:
//
//  - **`encoding`, distinct from `loading`.** LOAD is the download; encoding is
//    the per-image vision-encoder pass, and it is the slow half of every click
//    the user makes. Collapsing them means the user clicks, waits a second, and
//    is told nothing — which is exactly the experience the split architecture
//    exists to avoid. `encoded` says when clicks will actually be fast.
//  - **`encode()` as its own call.** Not folded into `decode`: a lazy re-encode
//    inside a click handler would hide the cost this page is built to show.
//
// **Clicks queue no deeper than one.** A click during an in-flight decode
// replaces the pending point set rather than appending to a backlog — the same
// never-queue rule `useLiveFrames` follows for camera frames, and for the same
// reason: the user's most recent click is the only one whose answer they still
// want, and a queue turns a fast model into a laggy one.

import { useCallback, useMemo, useRef, useState } from "react";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { toPayload, transferablesOf } from "@/vision/image";
import type { RawImage } from "@huggingface/transformers";
import { createSamWorker } from "@/vision/sam/client";
import {
  DEFAULT_SAM_MODEL,
  SAM_MODELS,
  type SamMask,
  type SamPoint,
  type SamResult,
} from "@/vision/sam/types";

export interface MaskResult {
  /** SAM's candidates, best-scoring first. */
  masks: SamMask[];
  /** Decode time in milliseconds — the sub-100 ms claim, measured. */
  ms: number;
}

export interface UseSamResult {
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  loadProgress: LoadProgress | null;
  loadedInMs: number | null;
  backend: string | null;
  running: boolean;
  error: string | null;
  result: MaskResult | null;
  /** True while the vision encoder is running on a newly-picked image. */
  encoding: boolean;
  /** True once an image is encoded and clicks can be decoded. */
  encoded: boolean;
  /** Milliseconds the last encode took. Null if it was served from cache. */
  encodedInMs: number | null;
  /**
   * Encode an image. `token` identifies it — re-encoding is skipped when the
   * same token comes back, so a re-render never costs a second encoder pass.
   */
  encode: (token: string, image: RawImage) => Promise<void>;
  /** Decode masks for a set of clicks. */
  run: (points: readonly SamPoint[]) => Promise<MaskResult>;
  /** Drop the current masks and the "encoded" flag — used on a new picture. */
  reset: () => void;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useSam(
  model: string = DEFAULT_SAM_MODEL,
  autoLoad = false,
): UseSamResult {
  const meta = useMemo(
    () => SAM_MODELS.find((m) => m.id === model) ?? SAM_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(
    () => ({ model: meta.id, ...(meta.dtypes ? { dtypes: meta.dtypes } : {}) }),
    [meta],
  );

  const worker = useModelWorker<SamResult>({
    createWorker: createSamWorker,
    key: `sam:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Mask worker not ready",
  });

  const { run: post } = worker;
  const [result, setResult] = useState<MaskResult | null>(null);
  const [encoding, setEncoding] = useState(false);
  const [encoded, setEncoded] = useState(false);
  const [encodedInMs, setEncodedInMs] = useState<number | null>(null);

  // One decode in flight, and at most one waiting. A third click replaces the
  // waiting one rather than lengthening a queue.
  const decoding = useRef(false);
  const queued = useRef<readonly SamPoint[] | null>(null);

  const encode = useCallback(
    async (token: string, image: RawImage): Promise<void> => {
      setEncoding(true);
      // The masks belong to the previous picture. Dropping them here, before
      // the await, means the OUTPUT slot never shows one image's mask over
      // another's pixels.
      setResult(null);
      setEncoded(false);
      try {
        // `copy: false` — the payload is built for this post and the page keeps
        // its own `RawImage`; the buffer is the encoder's to consume.
        const payload = toPayload(image, { copy: false });
        const res = await post(
          { kind: "encode", token, image: payload },
          transferablesOf(payload),
        );
        if (res.kind !== "encode") throw new Error("Unexpected encode reply");
        setEncoded(true);
        setEncodedInMs(res.cached ? null : Math.round(res.ms));
      } finally {
        setEncoding(false);
      }
    },
    [post],
  );

  const run = useCallback(
    async (points: readonly SamPoint[]): Promise<MaskResult> => {
      if (decoding.current) {
        // Supersede whatever was waiting: only the latest click matters.
        queued.current = points;
        return result ?? { masks: [], ms: 0 };
      }

      decoding.current = true;
      try {
        let next: readonly SamPoint[] | null = points;
        let last: MaskResult = { masks: [], ms: 0 };
        while (next) {
          const asked = next;
          queued.current = null;
          const res = await post({ kind: "decode", points: [...asked] });
          if (res.kind !== "decode") throw new Error("Unexpected decode reply");
          last = { masks: res.masks, ms: res.ms };
          setResult(last);
          next = queued.current;
        }
        return last;
      } finally {
        decoding.current = false;
        queued.current = null;
      }
    },
    [post, result],
  );

  const reset = useCallback(() => {
    setResult(null);
    setEncoded(false);
    setEncodedInMs(null);
    queued.current = null;
  }, []);

  return {
    status: worker.status,
    idle: worker.idle,
    loading: worker.loading,
    ready: worker.ready,
    progress: worker.progress,
    loadProgress: worker.loadProgress,
    loadedInMs: worker.loadedInMs,
    backend: worker.backend,
    running: worker.running,
    error: worker.error,
    result,
    encoding,
    encoded,
    encodedInMs,
    encode,
    run,
    reset,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
