// Text feature extraction — a sentence in, a vector out. The mirror of
// `/image-features`, and named for it rather than for the Hub's
// `feature-extraction` slug for exactly that reason.
//
// Three things this page is built to make visible, none of which are visible in
// the number itself:
//
//   **The pooling is not a free choice.** A sentence embedding is a pooling of
//   the token rows, and *which* pooling is part of how the checkpoint was
//   trained — mean for all-MiniLM and all-mpnet, CLS for BGE and
//   gte-modernbert. Pooling a CLS-trained model by the mean returns a vector of
//   the right width that ranks plausibly and is wrong, with nothing failing. So
//   it is catalogue data (`EmbedModel.pooling`), it is on screen in SELECT
//   before anything downloads, and `just fe-e2e-models` checks it against the
//   upstream repo's own `1_Pooling/config.json`.
//
//   **Truncation costs something, and the page measures it rather than
//   asserting it doesn't.** Matryoshka representation learning is a claim about
//   *MRL-trained* checkpoints; four of the five entries here are not, so the
//   slider reports the fraction of the vector's length the prefix actually held
//   (`Truncated.kept`) instead of implying the tail was free. Renormalising
//   after the cut is the part that is easy to forget and looks like a working
//   slider without it — `‖v‖` printed under the strip is what shows it happened.
//
//   **Truncation re-derives.** It is arithmetic over a vector in hand, so the
//   slider spends nothing — the same rule `/vad`'s threshold follows.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Ruler, Waypoints } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { VectorStrip } from "@/components/text/VectorStrip";
import { Button } from "@/components/ui/button";
import { useTextEmbed } from "@/hooks/useTextEmbed";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_EMBED_MODEL,
  EMBED_MODELS,
  FEATURE_TEXT_SAMPLES,
} from "@/text/catalogue";
import { truncate, truncationSteps } from "@/text/embed";

export const Route = createFileRoute("/text-features")({
  component: TextFeaturesPage,
});

/** One finished embedding, captured at run time. */
interface RunRecord {
  /** The text as typed… */
  text: string;
  /** …and the exact string that was sent, prefix included. */
  sent: string;
  vector: Float32Array;
  modelLabel: string;
  pooling: string;
}

function TextFeaturesPage() {
  const session = useModelSelection({
    routeKey: "text-features",
    models: EMBED_MODELS,
    fallback:
      EMBED_MODELS.find((m) => m.id === DEFAULT_EMBED_MODEL) ?? EMBED_MODELS[0],
  });
  const model = session.model;

  const {
    status,
    ready,
    loading,
    loadProgress,
    loadedInMs,
    backend,
    running,
    error,
    load,
    retry,
    cancel,
    run,
    compose,
    cached,
  } = useTextEmbed(model.id);
  useCacheRefresh(session, ready);

  const [text, setText] = useState(FEATURE_TEXT_SAMPLES[0].text);
  const [ran, setRan] = useState<RunRecord | null>(null);
  /** Dimensions kept. `null` means "all of them" — the honest default. */
  const [keep, setKeep] = useState<number | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const steps = useMemo(
    () => truncationSteps(ran?.vector.length ?? model.dim),
    [ran, model.dim],
  );

  // Pure derivation over the vector already in hand: no model, no worker, no
  // cost. `keep === null` still goes through `truncate` so the norm shown for
  // the full vector comes from the same code path as the truncated one.
  const shown = useMemo(() => {
    if (!ran) return null;
    return truncate(ran.vector, keep ?? ran.vector.length);
  }, [ran, keep]);

  async function embed() {
    if (!ready) return;
    const captured = text.trim();
    if (!captured) return;
    const sent = compose(captured);
    const [vector] = await run(captured);
    setRan({
      text: captured,
      sent,
      vector,
      modelLabel: model.label,
      pooling: model.pooling,
    });
  }

  return (
    <ModelPage
      icon={Waypoints}
      title="Text Features"
      description={
        <>
          Turn a sentence into a vector, in your browser, on your GPU (WebGPU) or
          CPU (WASM). Nothing about the vector is meaningful on its own — it is
          meaningful <em>relative to other vectors</em>, which is what{" "}
          <a className="underline" href="/sentence-similarity">
            Sentence Similarity
          </a>{" "}
          next door does with it.
        </>
      }
      labels={{ run: "Text", output: "Vector" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={EMBED_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            <span data-testid="embed-dim">{model.dim}</span> dimensions, pooled
            by{" "}
            <code data-testid="embed-pooling" className="rounded bg-muted px-1 font-mono">
              {model.pooling}
            </code>
            {" — "}
            read from this checkpoint's own training config, not chosen here.
            {model.matryoshka
              ? " Matryoshka-trained, so its leading dimensions were optimised to stand alone."
              : " Not Matryoshka-trained, so truncating it below costs something measurable."}
          </p>
          {model.prefixes && (
            <p
              data-testid="embed-prefix"
              className="text-xs leading-snug text-muted-foreground"
            >
              This checkpoint expects a task instruction, so{" "}
              <code className="rounded bg-muted px-1 font-mono">
                {model.prefixes.symmetric}
              </code>{" "}
              is prepended to whatever you type. Omitting it costs accuracy with
              nothing failing.
            </p>
          )}
        </div>
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
          disabled={running}
        />
      }
      run={
        <InputPanel
          ready={ready}
          disabledHint="Load a model to embed text. You can write it first."
          controls={
            <>
              <Button
                disabled={!ready || running || text.trim().length === 0}
                onClick={() => void embed().catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Embedding…
                  </>
                ) : (
                  <>
                    <Ruler className="size-4" /> Embed
                  </>
                )}
              </Button>
              {cached > 0 && (
                <span
                  data-testid="embed-cache"
                  className="text-xs text-muted-foreground"
                >
                  {cached} embedding{cached === 1 ? "" : "s"} remembered — the
                  same text is never embedded twice.
                </span>
              )}
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="tf-text" className="sr-only">
              Text to embed
            </label>
            <textarea
              id="tf-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
              className="min-h-28 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Write something to embed…"
            />

            {model.prefixes && (
              <p className="font-mono text-xs break-words text-muted-foreground">
                Sending: {compose(text.trim())}
              </p>
            )}

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — picking one fills the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {FEATURE_TEXT_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => setText(s.text)}
                    title={s.hint}
                  >
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Vector"
          description="One row of a diverging heatmap: red is positive, blue negative, and how solid a column looks is how large that dimension is."
          meta={ran ? <span>{ran.modelLabel}</span> : undefined}
          running={running}
          runningLabel="Embedding…"
          error={runError}
          empty="Write a sentence and press Embed — its vector is drawn here, with its width and its length."
        >
          {ran && shown && (
            <div className="space-y-5" data-testid="embedding">
              <VectorStrip
                vector={shown.vector}
                label={
                  <span className="text-muted-foreground italic">
                    “{ran.text}”
                  </span>
                }
                note={`pooled by ${ran.pooling}`}
              />

              <div className="space-y-2 border-t pt-4">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <p className="text-sm font-medium">Truncation</p>
                  <span className="text-xs text-muted-foreground">
                    Re-derives from the vector above. Runs nothing.
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {steps.map((n) => {
                    const active = (keep ?? ran.vector.length) === n;
                    return (
                      <Button
                        key={n}
                        size="sm"
                        variant={active ? "default" : "outline"}
                        aria-pressed={active}
                        data-testid={`truncate-${n}`}
                        onClick={() =>
                          setKeep(n === ran.vector.length ? null : n)
                        }
                      >
                        {n}-d
                      </Button>
                    );
                  })}
                </div>
                <p
                  data-testid="truncate-kept"
                  className="text-xs leading-snug text-muted-foreground"
                >
                  The first {shown.dim} of {ran.vector.length} dimensions hold{" "}
                  <strong className="tabular-nums">
                    {(shown.kept * 100).toFixed(1)}%
                  </strong>{" "}
                  of this vector's length. That fraction is the factor a{" "}
                  <em>missing</em> renormalisation would scale every similarity
                  by — which is why the strip above is renormalised and reads
                  ‖v‖ = 1.000 at every width.
                </p>
              </div>

              {ran.sent !== ran.text && (
                <p
                  data-testid="embed-sent"
                  className="text-xs text-muted-foreground"
                >
                  Sent to the model as{" "}
                  <code className="font-mono">{ran.sent}</code>.
                </p>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
