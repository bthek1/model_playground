// Picking an image, for every vision route.
//
// Four pages now need the same three sources — a file, a drop, a bundled sample
// — plus the one piece of bookkeeping that is easy to get wrong and invisible
// when you do: **exactly one object URL alive at a time, and none after
// unmount.** A preview URL that outlives its `<img>` pins the decoded bitmap in
// memory for the tab's lifetime, and five copies of that rule is five chances to
// drop it.
//
// It owns the *input* half only. The model, the run and the result stay with the
// route's task hook — this hook never sees either.

import { useCallback, useEffect, useRef, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import { fromFile, fromUrl } from "@/vision/image";
import type { ImageSample } from "@/vision/samples";

export interface PickedImage {
  /** The decoded pixels, kept so the page can re-run without re-decoding. */
  image: RawImage;
  /** Object URL or sample URL, for the preview. */
  previewUrl: string;
  /** True when `previewUrl` is ours to revoke. */
  owned: boolean;
  name: string;
}

export interface UseImagePickResult {
  picked: PickedImage | null;
  /** Which source is decoding, for the button that started it. Null when idle. */
  preparing: null | "file" | "sample";
  /** A failed decode or fetch. Belongs to the RUN slot, never to OUTPUT. */
  error: string | null;
  clearError: () => void;
  pickFile: (file: File | undefined) => void;
  pickSample: (sample: ImageSample) => void;
  /** Drop whatever is picked — used when a page switches to the live camera. */
  clear: () => void;
}

export function useImagePick({
  /**
   * Called with the freshly decoded image. A page uses it to run the model
   * immediately when one is loaded. Awaited so `preparing` covers the whole
   * round trip, but its **failures are swallowed**: an inference error belongs
   * to the OUTPUT slot, where the task hook already puts it, not to this one.
   */
  onPicked,
}: { onPicked?: (picked: PickedImage) => void | Promise<void> } = {}): UseImagePickResult {
  const [picked, setPicked] = useState<PickedImage | null>(null);
  const [preparing, setPreparing] = useState<null | "file" | "sample">(null);
  const [error, setError] = useState<string | null>(null);

  // A ref, not a dep: routes pass an inline closure over `ready`, and re-creating
  // the pick callbacks on every render would churn every button below them.
  const onPickedRef = useRef(onPicked);
  onPickedRef.current = onPicked;

  const ownedUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    },
    [],
  );

  const adopt = useCallback((next: PickedImage | null) => {
    if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    ownedUrl.current = next?.owned ? next.previewUrl : null;
    setPicked(next);
  }, []);

  const pick = useCallback(
    async (
      open: () => Promise<PickedImage>,
      kind: "file" | "sample",
    ): Promise<void> => {
      setError(null);
      setPreparing(kind);
      try {
        const next = await open();
        adopt(next);
        try {
          await onPickedRef.current?.(next);
        } catch {
          /* the task hook surfaces an inference failure in OUTPUT */
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPreparing(null);
      }
    },
    [adopt],
  );

  const pickFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      void pick(
        async () => ({
          image: await fromFile(file),
          previewUrl: URL.createObjectURL(file),
          owned: true,
          name: file.name,
        }),
        "file",
      );
    },
    [pick],
  );

  const pickSample = useCallback(
    (sample: ImageSample) =>
      void pick(
        async () => ({
          image: await fromUrl(sample.url),
          previewUrl: sample.url,
          owned: false,
          name: sample.label,
        }),
        "sample",
      ),
    [pick],
  );

  const clear = useCallback(() => adopt(null), [adopt]);
  const clearError = useCallback(() => setError(null), []);

  return { picked, preparing, error, clearError, pickFile, pickSample, clear };
}
