// Image Feature Extraction — the task with no visible output of its own, so the
// page *is* the similarity search: embed a small gallery in the tab, embed the
// user's picture, show the nearest neighbours. The whole index lives in memory
// and never leaves the machine, which is what makes an in-browser version
// genuinely private rather than merely convenient.
//
// Three decisions, each guarding a specific failure:
//
//  1. **"Which vector do you take" is the control this page exists for.** One
//     forward pass returns `[1, 1 + patches, dim]`; row 0 is CLS and the rest
//     are patches, and the two give visibly different neighbours. Both are
//     derived from the same pass (`poolEmbedding`), so switching between them
//     re-ranks and never re-embeds — on a twelve-image gallery that is the
//     difference between instant and a quarter of a minute. A CLIP-shaped model
//     returns one projected vector and the page says so instead of showing a
//     toggle that does nothing.
//  2. **`k` re-ranks, pooling re-ranks, only the checkpoint invalidates.**
//     Embeddings from a different model are not comparable with these, so
//     switching models drops the index; nothing else does.
//  3. **The norm is displayed, not asserted.** "The vectors are normalised" is a
//     claim, and this is the one page that can simply show it — the pre-
//     normalisation L2 norm beside a unit-length result.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Fingerprint, Loader2, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useImageFeatures, type IndexEntry } from "@/hooks/useImageFeatures";
import { useImagePick } from "@/hooks/useImagePick";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_FEATURE_MODEL,
  DEFAULT_K,
  FEATURE_MODELS,
  MAX_INFERENCE_SIDE,
  VECTOR_HINTS,
  VECTOR_LABELS,
  vectorKinds,
  type Embedding,
  type VectorKind,
} from "@/vision/features";
import { GALLERY_IMAGES, type GalleryImage } from "@/vision/gallery";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";
import { topK, type Neighbour } from "@/vision/similarity";

export const Route = createFileRoute("/image-features")({
  component: ImageFeaturesPage,
});

function ImageFeaturesPage() {
  const session = useModelSelection({
    routeKey: "image-features",
    models: FEATURE_MODELS,
    fallback:
      FEATURE_MODELS.find((m) => m.id === DEFAULT_FEATURE_MODEL) ??
      FEATURE_MODELS[0],
  });
  const model = session.model.id;
  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    result,
    error,
    load,
    retry,
    cancel,
    run,
    index,
    indexing,
    buildIndex,
    addToIndex,
    clearIndex,
  } = useImageFeatures(model, session.autoLoad);
  useCacheRefresh(session, ready);

  const [kind, setKind] = useState<VectorKind>("cls");
  const [k, setK] = useState(DEFAULT_K);
  const [live, setLive] = useState(false);

  // Vectors from another checkpoint are not comparable with these. Switching
  // models is the only thing that invalidates the index — not the pooling, and
  // not `k`.
  useEffect(() => {
    clearIndex();
  }, [model, clearIndex]);

  const embed = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      await run(small, { consume: small !== image });
    },
    [run],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
    useImagePick({
      onPicked: async (next) => {
        if (ready) await embed(next.image);
      },
    });

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (!ready) return;
      await embed(frame);
    },
    [ready, embed],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null || indexing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // Which vectors this checkpoint actually offers. Derived from a real result
  // rather than from the catalogue, so a model whose output shape surprises us
  // shows the truth instead of a promise.
  const kinds = useMemo(
    () => (result ? vectorKinds(result) : (["cls", "mean"] as VectorKind[])),
    [result],
  );
  const activeKind = kinds.includes(kind) ? kind : kinds[0];

  // Pure, on the main thread, and the whole point: changing the vector or `k`
  // re-ranks numbers already in hand. Nothing is re-embedded.
  const neighbours = useMemo((): Neighbour[] => {
    const query = result?.vectors[activeKind];
    if (!query) return [];
    const vectors = index
      .map((entry) => ({
        id: entry.image.id,
        vector: entry.embedding.vectors[activeKind],
      }))
      .filter((e): e is { id: string; vector: Float32Array } => e.vector != null);
    return topK(query, vectors, k);
  }, [result, index, activeKind, k]);

  const byId = useMemo(
    () => new Map(index.map((entry) => [entry.image.id, entry])),
    [index],
  );

  const embedCurrent = () => {
    if (!picked) return;
    clearError();
    void embed(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  const addCurrent = () => {
    if (!picked) return;
    const entry: GalleryImage = {
      id: `user:${picked.name}`,
      label: picked.name,
      url: picked.previewUrl,
      group: "object",
    };
    void addToIndex(entry, picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Fingerprint}
      title="Image Feature Extraction"
      description={
        <>
          Turn pictures into vectors and search them by similarity — the index is
          built in this tab and never leaves it.
        </>
      }
      labels={{ output: "Nearest neighbours" }}
      select={
        <ModelPicker
          models={FEATURE_MODELS}
          value={model}
          onChange={session.setModel}
          disabled={loading || busy}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
        />
      }
      load={
        <ModelStatus
          status={status}
          backend={backend}
          loadProgress={loadProgress}
          loadedInMs={loadedInMs}
          cached={session.isCached}
          restoring={session.restoring}
          error={loadError}
          onLoad={session.onLoad(load)}
          onCancel={session.onCancel(cancel)}
          onRetry={retry}
          disabled={busy}
        />
      }
      run={
        <InputPanel
          ready={ready}
          error={ioError}
          disabledHint="Load a model to embed anything. You can pick a picture first."
          controls={
            <>
              <Button
                disabled={!ready || busy || !picked || live}
                onClick={embedCurrent}
              >
                {running && !indexing ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Embedding…
                  </>
                ) : (
                  <>
                    <Fingerprint className="size-4" /> Embed
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                disabled={!ready || busy || !picked || live}
                onClick={addCurrent}
              >
                <Plus className="size-4" /> Add to gallery
              </Button>
            </>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — the query. The tiger is not in the gallery, which is what makes its neighbours worth reading."
            onFile={pickFile}
            onSample={pickSample}
            busy={busy}
            camera={{
              live,
              onToggle: setLive,
              videoRef: camera.videoRef,
              error: camera.error,
              fps: camera.fps,
            }}
          >
            <div className="space-y-3">
              <GalleryPanel
                index={index}
                indexing={indexing}
                disabled={!ready || busy}
                onBuild={() => void buildIndex(GALLERY_IMAGES)}
                onClear={clearIndex}
              />

              <div className="space-y-1.5">
                <span className="text-sm font-medium">Vector</span>
                <div className="flex flex-wrap gap-2">
                  {kinds.map((option) => (
                    <Button
                      key={option}
                      variant="outline"
                      size="sm"
                      aria-pressed={option === activeKind}
                      onClick={() => setKind(option)}
                      title={VECTOR_HINTS[option]}
                    >
                      {VECTOR_LABELS[option]}
                    </Button>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  {VECTOR_HINTS[activeKind]} Switching re-ranks what is already
                  embedded — nothing runs again.
                </p>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="topk">
                  Neighbours: <span className="tabular-nums">{k}</span>
                </Label>
                <input
                  id="topk"
                  type="range"
                  min={1}
                  max={12}
                  step={1}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="block w-full max-w-xs"
                />
              </div>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Nearest neighbours"
          description="Cosine similarity against every picture in the in-memory index, best first."
          meta={
            result ? (
              <span className="tabular-nums">
                {result.dim}-d · {index.length} indexed
              </span>
            ) : undefined
          }
          running={running && !indexing}
          runningLabel="Embedding…"
          error={runError}
          empty="Embed the gallery, then pick a picture — its nearest neighbours and their cosine scores appear here."
        >
          {result && (
            <NeighbourList
              embedding={result}
              kind={activeKind}
              neighbours={neighbours}
              entries={byId}
              indexed={index.length}
            />
          )}
        </OutputPanel>
      }
    />
  );
}

function GalleryPanel({
  index,
  indexing,
  disabled,
  onBuild,
  onClear,
}: {
  index: readonly IndexEntry[];
  indexing: { done: number; total: number; label: string } | null;
  disabled: boolean;
  onBuild: () => void;
  onClear: () => void;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-sm font-medium">Gallery</span>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" disabled={disabled} onClick={onBuild}>
          {indexing ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Embedding{" "}
              {indexing.done + 1}/{indexing.total}
            </>
          ) : (
            <>Embed the {GALLERY_IMAGES.length} bundled pictures</>
          )}
        </Button>
        {index.length > 0 && !indexing && (
          <Button variant="ghost" size="sm" onClick={onClear}>
            <Trash2 className="size-4" /> Clear
          </Button>
        )}
        <span
          data-testid="index-size"
          className="font-mono text-xs text-muted-foreground tabular-nums"
        >
          {index.length} indexed
        </span>
      </div>
      {indexing && (
        <p className="text-xs text-muted-foreground">
          {/* One at a time, on purpose: two overlapping calls into one ONNX
              session is not a guarantee worth relying on, and a fan-out would
              queue anyway while making this counter meaningless. */}
          {indexing.label} — one at a time, so the count means something.
        </p>
      )}
      {index.length === 0 && !indexing && (
        <p className="text-xs text-muted-foreground">
          Nothing is indexed yet. The gallery is embedded in this tab and stays
          here.
        </p>
      )}
    </div>
  );
}

function NeighbourList({
  embedding,
  kind,
  neighbours,
  entries,
  indexed,
}: {
  embedding: Embedding;
  kind: VectorKind;
  neighbours: readonly Neighbour[];
  entries: ReadonlyMap<string, IndexEntry>;
  indexed: number;
}) {
  const norm = embedding.norms[kind];

  return (
    <div className="space-y-4">
      {indexed === 0 ? (
        <p className="text-sm text-muted-foreground">
          The picture is embedded, but there is nothing to compare it with yet —
          embed the gallery in the RUN panel.
        </p>
      ) : (
        <ul className="space-y-2" data-testid="neighbours">
          {neighbours.map((n) => {
            const entry = entries.get(n.id);
            if (!entry) return null;
            return (
              <li key={n.id} className="flex items-center gap-3">
                <img
                  src={entry.image.url}
                  alt=""
                  className="size-12 shrink-0 rounded object-cover"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="truncate text-sm">{entry.image.label}</span>
                    <span className="text-xs text-muted-foreground">
                      {entry.image.group}
                    </span>
                    <span className="ml-auto font-mono text-xs tabular-nums">
                      {n.score.toFixed(3)}
                    </span>
                  </div>
                  <div
                    aria-hidden
                    className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted"
                  >
                    <div
                      className="h-full bg-primary"
                      // Cosine over a self-supervised encoder rarely drops below
                      // ~0.3 even for unrelated pictures, so the bar is scaled to
                      // the [0.3, 1] band the scores actually occupy. A raw 0–1
                      // bar makes every neighbour look equally close.
                      style={{
                        width: `${Math.max(0, Math.min(1, (n.score - 0.3) / 0.7)) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <p data-testid="embedding-facts" className="text-xs text-muted-foreground">
        {/* The normalisation made visible. "The vectors are normalised" is a
            claim, and this is the one page that can simply show it. */}
        <code>{VECTOR_LABELS[kind]}</code> · {embedding.dim} dimensions ·{" "}
        {embedding.tokens} token{embedding.tokens === 1 ? "" : "s"} returned ·
        L2 norm before normalising{" "}
        <span className="tabular-nums">
          {norm == null ? "—" : norm.toFixed(3)}
        </span>
        , after{" "}
        <span className="tabular-nums">1.000</span>
      </p>

      <p className="text-xs text-muted-foreground">
        Cosine similarity is scale-free, which is the reason the vectors are
        normalised first: without it a picture the encoder simply responds
        strongly to would outrank a picture that actually looks like the query.
      </p>
    </div>
  );
}
