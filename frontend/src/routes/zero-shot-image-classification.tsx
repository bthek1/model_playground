// Zero-Shot Image Classification — the user types the labels, CLIP or SigLIP
// scores the picture against them. No fine-tuning, no fixed class list, and the
// most satisfying page in the Computer Vision category.
//
// **The prompt-template comparison is why this page exists**, rather than being
// a fourth classifier. `"a photo of a {}"` beats a bare `"{}"` on the same model
// and the same picture by several points — the single most instructive result in
// the vision roadmap, and something very few demos anywhere show. So both
// templates are scored and shown side by side: a before/after the user has to
// hold in their head is not a demonstration.
//
// The other thing this page owes the user is a sentence it would be easy to
// leave out: **a zero-shot score is relative to the labels given.** CLIP
// softmaxes over exactly the list it was handed, so "cat 0.98" against
// ["cat", "dog"] means "more cat than dog" — not "a cat is present". SigLIP was
// trained with a sigmoid loss and scores each label independently, which is why
// its numbers look different and mean more; the catalogue says which is which
// and the OUTPUT slot repeats it for the model in use.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { createFileRoute } from "@tanstack/react-router";
import type { RawImage } from "@huggingface/transformers";
import { Loader2, Tags } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ImageSourcePanel } from "@/components/vision/ImageSourcePanel";
import { PhraseList } from "@/components/vision/PhraseList";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useCameraFrames } from "@/hooks/useCameraFrames";
import { useImagePick } from "@/hooks/useImagePick";
import {
  useZeroShotImage,
  type TemplateScores,
} from "@/hooks/useZeroShotImage";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import { downscale } from "@/vision/image";
import { IMAGE_SAMPLES } from "@/vision/samples";
import {
  applyTemplate,
  DEFAULT_LABELS,
  DEFAULT_ZERO_SHOT_MODEL,
  MAX_INFERENCE_SIDE,
  TEMPLATES,
  ZERO_SHOT_MODELS,
} from "@/vision/zeroShot";

export const Route = createFileRoute("/zero-shot-image-classification")({
  component: ZeroShotImageClassificationPage,
});

function ZeroShotImageClassificationPage() {
  const session = useModelSelection({
    routeKey: "zero-shot-image-classification",
    models: ZERO_SHOT_MODELS,
    fallback:
      ZERO_SHOT_MODELS.find((m) => m.id === DEFAULT_ZERO_SHOT_MODEL) ??
      ZERO_SHOT_MODELS[0],
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
  } = useZeroShotImage(model);
  useCacheRefresh(session, ready);

  const [labels, setLabels] = useState<string[]>(DEFAULT_LABELS);
  const [template, setTemplate] = useState<string>(TEMPLATES.photo);
  const [compare, setCompare] = useState(true);
  const [live, setLive] = useState(false);

  // Both templates on a still image; one on a live feed, where the second
  // doubles the per-frame cost for a comparison nobody can read at 8 fps.
  const templates = useMemo(
    () =>
      compare && !live
        ? Array.from(new Set([TEMPLATES.bare, template]))
        : [template],
    [compare, live, template],
  );

  const score = useCallback(
    async (image: RawImage) => {
      const small = await downscale(image, MAX_INFERENCE_SIDE);
      await run(small, labels, templates);
    },
    [run, labels, templates],
  );

  const { picked, preparing, error: ioError, clearError, pickFile, pickSample } =
useImagePick();

  const onFrame = useCallback(
    async (frame: RawImage) => {
      if (!ready || labels.length === 0) return;
      await score(frame);
    },
    [ready, labels, score],
  );
  const camera = useCameraFrames({
    active: live,
    maxSide: MAX_INFERENCE_SIDE,
    onFrame,
  });

  const busy = running || preparing !== null;
  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const scoreCurrent = () => {
    if (!picked) return;
    clearError();
    void score(picked.image).catch(() => {
      /* the hook surfaces it in OUTPUT */
    });
  };

  return (
    <ModelPage
      icon={Tags}
      title="Zero-Shot Image Classification"
      description={
        <>
          Type your own labels and let CLIP score the picture against them — no
          training, no fixed class list. It runs on your GPU in a Web Worker.
        </>
      }
      labels={{ output: "Scores" }}
      select={
        <ModelPicker
          models={ZERO_SHOT_MODELS}
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
          disabledHint="Load a model to score your labels. You can write them and pick a picture first."
          controls={
            <Button
              disabled={!ready || busy || !picked || live || labels.length === 0}
              onClick={scoreCurrent}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Scoring…
                </>
              ) : (
                <>
                  <Tags className="size-4" /> Score labels
                </>
              )}
            </Button>
          }
        >
          <ImageSourcePanel
            picked={picked}
            preparing={preparing}
            samples={IMAGE_SAMPLES}
            sampleHint="Samples — try labels the model has to choose between, not one obvious answer."
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
              <PhraseList
                id="new-label"
                title="Labels"
                items={labels}
                onChange={setLabels}
                placeholder="bicycle"
                emptyHint="Add at least one label to score against."
              />

              <div className="space-y-1.5">
                <Label htmlFor="template">Prompt template</Label>
                <Input
                  id="template"
                  value={template}
                  onChange={(e) => setTemplate(e.target.value)}
                  className="max-w-xs font-mono text-xs"
                />
                <div className="flex flex-wrap gap-2">
                  {Object.entries(TEMPLATES).map(([key, value]) => (
                    <Button
                      key={key}
                      variant="outline"
                      size="sm"
                      aria-pressed={template === value}
                      onClick={() => setTemplate(value)}
                    >
                      {value}
                    </Button>
                  ))}
                </div>
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={compare}
                    disabled={live}
                    onChange={(e) => setCompare(e.target.checked)}
                  />
                  Also score the bare <code>{TEMPLATES.bare}</code> template, to
                  compare
                </label>
                <p className="text-xs text-muted-foreground">
                  {labels[0]
                    ? `Sent to the model as “${applyTemplate(template, labels[0])}”.`
                    : "`{}` is replaced with each label."}
                </p>
              </div>
            </div>
          </ImageSourcePanel>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Scores"
          description={
            session.model.scoring === "softmax"
              ? "A softmax over your labels — the scores always sum to 1."
              : "Sigmoid scores, one per label, independent of each other."
          }
          running={running}
          runningLabel="Scoring…"
          error={runError}
          empty="Write some labels, pick an image, and each label's score appears here — with and without the prompt template."
        >
          {result && result.length > 0 && (
            <ScoreTable results={result} labels={labels} />
          )}
        </OutputPanel>
      }
    />
  );
}

function ScoreTable({
  results,
  labels,
}: {
  results: readonly TemplateScores[];
  labels: readonly string[];
}) {
  // Rows in the order the user wrote their labels, not by score: the comparison
  // across templates is the point, and a table that re-sorts each column
  // separately makes it unreadable.
  const byTemplate = results.map(
    (r) => new Map(r.scores.map((s) => [s.label, s.score])),
  );
  const best = results.map((r) =>
    r.scores.reduce(
      (top, s) => (s.score > top.score ? s : top),
      r.scores[0] ?? { label: "", score: 0 },
    ),
  );

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left">
              <th className="py-1 pr-3 font-medium">Label</th>
              {results.map((r) => (
                <th
                  key={r.template}
                  className="py-1 pr-3 font-mono text-xs font-normal text-muted-foreground"
                >
                  {r.template}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((label) => (
              <tr key={label} className="border-b last:border-0">
                <td className="py-1 pr-3">{label}</td>
                {byTemplate.map((scores, i) => {
                  const score = scores.get(label);
                  return (
                    <td
                      key={results[i].template}
                      className="py-1 pr-3 font-mono text-xs tabular-nums"
                    >
                      {score == null ? "—" : score.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p data-testid="encode-cost" className="text-xs text-muted-foreground">
        {/* The caching made visible. The text tower is ~40% of CLIP's work and
            its output does not depend on the image, so it is computed once per
            label set and reused — on a live feed, for every subsequent frame.
            Showing the split is the only way a reader can tell the difference
            between an optimisation that works and a comment claiming one. */}
        {results.map((r, i) => (
          <span key={r.template}>
            {i > 0 && " · "}
            <code>{r.template}</code>: image {Math.round(r.imageMs)} ms, labels{" "}
            {r.textCached ? (
              <span className="text-emerald-600 dark:text-emerald-500">
                reused
              </span>
            ) : (
              `${Math.round(r.textMs)} ms`
            )}
          </span>
        ))}
      </p>

      {results.length > 1 && (
        <p data-testid="template-verdict" className="text-xs text-muted-foreground">
          Top label per template:{" "}
          {results.map((r, i) => (
            <span key={r.template}>
              {i > 0 && " · "}
              <code>{r.template}</code> → {best[i].label}{" "}
              <span className="tabular-nums">{best[i].score.toFixed(3)}</span>
            </span>
          ))}
          . The wording of the prompt moves these numbers as much as the
          checkpoint does — that is the experiment.
        </p>
      )}

      <p className="text-xs text-amber-600 dark:text-amber-500">
        These scores are <span className="font-medium">relative to the labels
        you gave</span>. The model picks the best match from your list; it has no
        way to say “none of these”.
      </p>
    </div>
  );
}
