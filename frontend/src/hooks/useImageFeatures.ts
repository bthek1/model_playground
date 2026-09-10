// In-browser image feature extraction, plus the in-memory index the page is
// built around. A thin wrapper over `useVisionPipeline` for the forward pass,
// and one genuinely task-specific piece on top of it: embedding a gallery is N
// runs, not one.
//
// Three decisions live here:
//
//  1. **One forward pass yields every vector the page can offer.** The pipeline
//     is called with no `pool` option, so it returns the last hidden state and
//     `poolEmbedding` derives CLS *and* the mean of the patches from it. The
//     pooling toggle is then a pure re-rank over vectors already in hand rather
//     than a second pass over the whole gallery — which on twelve images is the
//     difference between instant and a quarter of a minute. (The plan in #18
//     assumed the toggle would re-embed; it does not need to, and the shipped
//     behaviour is the better one.)
//  2. **One request in flight at a time.** The gallery is embedded in a
//     sequential loop with a progress count, never as a `Promise.all` — two
//     overlapping calls into one ONNX session is not a guarantee worth relying
//     on, and a twelve-way fan-out on WASM would simply queue anyway while
//     making the progress meaningless.
//  3. **The index is invalidated by the model, not by the pooling.** Embeddings
//     from a different checkpoint are not comparable with these, so switching
//     models drops the index; switching the pooling does not.

import { useCallback, useMemo, useRef, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import {
  DEFAULT_FEATURE_MODEL,
  FEATURE_MODELS,
  MAX_INFERENCE_SIDE,
  poolEmbedding,
  type Embedding,
} from "@/vision/features";
import type { GalleryImage } from "@/vision/gallery";
import { downscale, fromUrl } from "@/vision/image";
import type { PlainTensor } from "@/vision/serialize";

/** One embedded gallery picture. */
export interface IndexEntry {
  image: GalleryImage;
  embedding: Embedding;
}

export interface IndexProgress {
  done: number;
  total: number;
  /** What is being embedded right now, for the progress line. */
  label: string;
}

export interface UseImageFeaturesResult
  extends Omit<UseVisionPipelineResult, "run"> {
  /** The query image's embedding, or null before anything has been embedded. */
  result: Embedding | null;
  /** Embed one image and keep it as `result`. */
  run: (image: RawImage, opts?: { consume?: boolean }) => Promise<Embedding>;
  /** The embedded gallery. Empty until `buildIndex` has run. */
  index: IndexEntry[];
  /** Non-null while the gallery is being embedded. */
  indexing: IndexProgress | null;
  /** Embed every picture in `images`, one at a time. Re-running replaces it. */
  buildIndex: (images: readonly GalleryImage[]) => Promise<void>;
  /** Add one already-decoded picture to the index. */
  addToIndex: (image: GalleryImage, decoded: RawImage) => Promise<void>;
  /** Drop the index — used when the checkpoint changes under it. */
  clearIndex: () => void;
}

export function useImageFeatures(
  model: string = DEFAULT_FEATURE_MODEL,
  autoLoad = false,
): UseImageFeaturesResult {
  const meta = useMemo(
    () => FEATURE_MODELS.find((m) => m.id === model) ?? FEATURE_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<Embedding | null>(null);
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [indexing, setIndexing] = useState<IndexProgress | null>(null);
  // Bumped whenever the index is invalidated, so a build still in flight when
  // the user switches models cannot write its results into the new index.
  const generation = useRef(0);

  const embed = useCallback(
    async (image: RawImage, opts?: { consume?: boolean }): Promise<Embedding> => {
      // No `pool` option: the pipeline returns the last hidden state, and both
      // poolings are derived from it. Asking for `pool: true` would throw the
      // patch rows away and with them the control this page exists for.
      const tensor = (await post(image, undefined, opts)) as PlainTensor;
      return poolEmbedding(tensor);
    },
    [post],
  );

  const run = useCallback(
    async (image: RawImage, opts?: { consume?: boolean }): Promise<Embedding> => {
      const embedding = await embed(image, opts);
      setResult(embedding);
      return embedding;
    },
    [embed],
  );

  const clearIndex = useCallback(() => {
    generation.current += 1;
    setIndex([]);
    setIndexing(null);
  }, []);

  const buildIndex = useCallback(
    async (images: readonly GalleryImage[]): Promise<void> => {
      generation.current += 1;
      const mine = generation.current;
      setIndex([]);
      setIndexing({ done: 0, total: images.length, label: "" });

      try {
        for (const image of images) {
          if (generation.current !== mine) return; // superseded
          setIndexing({
            done: images.indexOf(image),
            total: images.length,
            label: image.label,
          });
          // A failed fetch or decode skips that picture rather than abandoning
          // the gallery: eleven neighbours are still a working page, and the
          // one that failed is visibly absent.
          try {
            const decoded = await downscale(
              await fromUrl(image.url),
              MAX_INFERENCE_SIDE,
            );
            const embedding = await embed(decoded, { consume: true });
            if (generation.current !== mine) return;
            setIndex((prev) => [...prev, { image, embedding }]);
          } catch {
            /* skipped — the gallery is a convenience, not the input */
          }
        }
      } finally {
        if (generation.current === mine) setIndexing(null);
      }
    },
    [embed],
  );

  const addToIndex = useCallback(
    async (image: GalleryImage, decoded: RawImage): Promise<void> => {
      const embedding = await embed(decoded);
      setIndex((prev) => [
        ...prev.filter((e) => e.image.id !== image.id),
        { image, embedding },
      ]);
    },
    [embed],
  );

  return {
    ...pipe,
    result,
    run,
    index,
    indexing,
    buildIndex,
    addToIndex,
    clearIndex,
  };
}
