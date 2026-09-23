// Fill-Mask — a masked language model guessing the word you took out, and the
// page where that stops being a parlour trick: **the ranked list is the
// training corpus, read back out**. Nothing here is a claim about the world.
//
// Two rules the page is built around, and both were measured against the real
// pipeline before they were written down:
//
//   **The mask token is never a literal.** BERT, DistilBERT and ModernBERT use
//   `[MASK]`; RoBERTa uses `<mask>`. So the token is inserted by a button, it is
//   rewritten in place when the model changes, and the engine reconciles
//   whatever arrives against the loaded tokenizer's own `mask_token`. A wrong
//   literal does not lie — `FillMaskPipeline` raises "Mask token (…) not found
//   in text." — but it is still a failure the user did nothing to cause.
//
//   **Exactly one mask, or the page refuses.** The pipeline takes `findIndex`
//   over the token ids: it fills the first mask and drops the rest with no
//   error at all. "The [MASK] of France is [MASK]." comes back as "the border
//   of france is." — one filling, a sentence quietly missing a word. That is
//   the silent failure on this page, and the reason the trigger states its
//   refusal instead of running.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useEffect, useMemo, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Eraser, Loader2, SquareAsterisk, Wand2 } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ScoreList } from "@/components/text/ScoreList";
import { SpanOverlay } from "@/components/text/SpanOverlay";
import { Button } from "@/components/ui/button";
import { useFillMask } from "@/hooks/useFillMask";
import type { ClassLabel } from "@/model/types";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_FILL_MASK,
  FILL_MASK_MODELS,
  FILL_TOP_K,
  MASK_PROBES,
  MASK_SAMPLES,
  MASK_TOKENS,
  type MaskProbe,
} from "@/text/catalogue";
import { highlight } from "@/text/highlight";
import {
  countMasks,
  expandTemplate,
  findMasks,
  insertMask,
  retargetMasks,
  spliceFill,
} from "@/text/mask";

export const Route = createFileRoute("/fill-mask")({
  component: FillMaskPage,
});

/** One finished fill, captured at run time so editing the box cannot restyle it. */
interface RunRecord {
  /** The exact string that was sent. */
  text: string;
  /** Where the mask sat in that string. */
  start: number;
  end: number;
  /** The literal the page put there… */
  sent: string;
  /** …and the one the tokenizer actually used. Usually the same. */
  resolved: string | null;
  fills: ClassLabel[];
  modelLabel: string;
}

/** The bias probe's result, kept beside the prompts that produced it. */
interface ProbeRecord {
  rows: { probe: MaskProbe; prompts: [string, string]; fills: ClassLabel[][] }[];
}

function FillMaskPage() {
  const session = useModelSelection({
    routeKey: "fill-mask",
    models: FILL_MASK_MODELS,
    fallback:
      FILL_MASK_MODELS.find((m) => m.id === DEFAULT_FILL_MASK) ??
      FILL_MASK_MODELS[0],
  });
  const model = session.model;
  const mask = model.maskToken;

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
  } = useFillMask(model.id);
  useCacheRefresh(session, ready);

  const boxRef = useRef<HTMLTextAreaElement | null>(null);
  const [text, setText] = useState(() =>
    expandTemplate(MASK_SAMPLES[0].template, FILL_MASK_MODELS[0].maskToken),
  );
  const [ran, setRan] = useState<RunRecord | null>(null);
  const [probed, setProbed] = useState<ProbeRecord | null>(null);
  /** Set when a model change rewrote the mask in the box — never silent. */
  const [rewrote, setRewrote] = useState<string | null>(null);

  // A model change makes the token already in the box the *other* family's.
  // Rewriting is the agreed answer rather than refusing: the user's sentence is
  // the part worth keeping, and the token is punctuation they did not type.
  // It re-derives on the main thread and spends nothing.
  const previousMask = useRef(mask);
  useEffect(() => {
    if (previousMask.current === mask) return;
    const from = previousMask.current;
    previousMask.current = mask;
    setText((current) => {
      const next = retargetMasks(current, MASK_TOKENS, mask);
      setRewrote(next === current ? null : from);
      return next;
    });
  }, [mask]);

  const masks = countMasks(text, mask);
  const blocked = maskProblem(masks, mask);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  // OUTPUT's sentence: the user's own string with the top filling spliced in,
  // and the span drawn over the characters it now occupies. Never the
  // pipeline's own `sequence`, which is a decode — on an uncased model that is
  // "the capital of france is paris." and would rewrite what the user typed.
  const marked = useMemo(() => {
    const top = ran?.fills[0];
    if (!ran || !top) return null;
    const filled = spliceFill(ran.text, { start: ran.start, end: ran.end }, top.label);
    return highlight(filled.text, [
      { ...filled.span, label: "FILL", score: top.score },
    ]);
  }, [ran]);

  function putMask() {
    const box = boxRef.current;
    const at = box?.selectionStart ?? text.length;
    const to = box?.selectionEnd ?? at;
    const next = insertMask(text, mask, at, to);
    setText(next.text);
    setRewrote(null);
    // Put the caret back where the user was, after the token they inserted.
    requestAnimationFrame(() => {
      box?.focus();
      box?.setSelectionRange(next.caret, next.caret);
    });
  }

  async function fill() {
    if (!ready || blocked) return;
    const captured = text;
    const span = findMasks(captured, mask)[0];
    if (!span) return;
    const out = await run(captured);
    setRan({
      text: captured,
      start: span.start,
      end: span.end,
      sent: mask,
      resolved: out.mask,
      fills: out.fills[0] ?? [],
      modelLabel: model.label,
    });
  }

  async function probe() {
    if (!ready) return;
    const prompts = MASK_PROBES.flatMap(
      (p) => p.templates.map((t) => expandTemplate(t, mask)) as [string, string],
    );
    const out = await run(prompts);
    setProbed({
      rows: MASK_PROBES.map((p, i) => ({
        probe: p,
        prompts: [prompts[i * 2], prompts[i * 2 + 1]] as [string, string],
        fills: [out.fills[i * 2] ?? [], out.fills[i * 2 + 1] ?? []],
      })),
    });
  }

  return (
    <ModelPage
      icon={SquareAsterisk}
      title="Fill Mask"
      description={
        <>
          Take a word out of a sentence and ask the model to put it back —
          entirely in your browser, on your GPU (WebGPU) or CPU (WASM). This is
          the objective these encoders were actually trained on, so what comes
          back is <strong>a measurement of the training text</strong>, not of
          the world.
        </>
      }
      labels={{ run: "Sentence", output: "Fillings" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={FILL_MASK_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            Pretrained on {model.domain}. Its mask token is{" "}
            <code data-testid="mask-token" className="rounded bg-muted px-1 font-mono">
              {mask}
            </code>{" "}
            — read from this checkpoint's tokenizer, and inserted for you.
          </p>
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
          disabledHint="Load a model to fill the mask. You can write the sentence first."
          controls={
            <>
              <Button
                disabled={!ready || running || blocked != null}
                onClick={() => void fill().catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Filling…
                  </>
                ) : (
                  <>
                    <Wand2 className="size-4" /> Fill the mask
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                disabled={!ready || running}
                onClick={() => void probe().catch(() => {})}
                title="Six prompts in one batch — three pairs differing by a single word."
              >
                Run the bias probes
              </Button>
              {blocked && (
                // The reason, on the trigger. A disabled button with nothing
                // beside it is a dead end, and a silent no-op is worse.
                <p
                  data-testid="mask-note"
                  className="basis-full text-xs text-amber-600 dark:text-amber-500"
                >
                  {blocked}
                </p>
              )}
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="fm-text" className="sr-only">
              Sentence with a mask
            </label>
            <textarea
              id="fm-text"
              ref={boxRef}
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setRewrote(null);
              }}
              rows={5}
              className="min-h-28 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Write a sentence, then insert the mask…"
            />

            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                data-testid="insert-mask"
                disabled={running}
                onClick={putMask}
              >
                <Eraser className="size-4" /> Insert {mask}
              </Button>
              <span className="text-xs text-muted-foreground">
                Typing the token by hand is how the wrong one ends up in the box.
              </span>
            </div>

            {rewrote && (
              <p
                data-testid="mask-rewritten"
                className="text-xs text-muted-foreground"
              >
                Switching model rewrote <code className="font-mono">{rewrote}</code>{" "}
                in your sentence to <code className="font-mono">{mask}</code>.
                Nothing ran.
              </p>
            )}

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — picking one fills the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {MASK_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => {
                      setText(expandTemplate(s.template, mask));
                      setRewrote(null);
                    }}
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
          title="Fillings"
          description={`The ${FILL_TOP_K} words this model thinks most likely, with their probabilities. The ranking is the evidence — a top answer alone hides it.`}
          meta={ran ? <span>{ran.modelLabel}</span> : undefined}
          running={running}
          runningLabel="Filling…"
          error={runError}
          empty="Write a sentence, insert the mask and press Fill the mask — the words the model would put there appear here, ranked."
        >
          {(ran || probed) && (
            <div className="space-y-5">
              {ran && marked && (
                <div className="space-y-3" data-testid="filled">
                  <SpanOverlay result={marked} showLabels={false} />
                  <ScoreList
                    scores={ran.fills}
                    data-testid="fill-scores"
                    caption={
                      <>
                        Read this as a description of {model.domain} — the text
                        this model was fitted to — rather than as a fact.
                      </>
                    }
                  />
                  {ran.resolved && ran.resolved !== ran.sent && (
                    // The catalogue and the checkpoint have drifted. The run was
                    // still correct, because the engine deferred to the
                    // tokenizer; saying nothing would leave the SELECT slot
                    // showing a token that was not used.
                    <p
                      data-testid="mask-drift"
                      className="text-xs text-amber-600 dark:text-amber-500"
                    >
                      This checkpoint's tokenizer uses{" "}
                      <code className="font-mono">{ran.resolved}</code>, not the{" "}
                      <code className="font-mono">{ran.sent}</code> this page
                      inserted. The run used the tokenizer's.
                    </p>
                  )}
                </div>
              )}

              {ran && ran.fills.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  The model returned no candidates. That is a failure, not an
                  answer — try loading it again.
                </p>
              )}

              {probed && (
                <div className="space-y-4 border-t pt-4" data-testid="probes">
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Paired prompts</p>
                    <p className="text-xs leading-snug text-muted-foreground">
                      Each pair is the same sentence with one word changed.
                      Whatever differs between the two columns was not in the
                      sentence — it came from the corpus. This is{" "}
                      <strong>evidence about the training data</strong>, not
                      about doctors, nurses, men or women.
                    </p>
                  </div>

                  {probed.rows.map(({ probe: p, prompts, fills }) => (
                    <div key={p.id} className="space-y-2">
                      <p className="text-xs font-medium">
                        {p.label}
                        <span className="ml-1.5 font-normal text-muted-foreground">
                          {p.hint}
                        </span>
                      </p>
                      <div className="grid gap-4 sm:grid-cols-2">
                        {[0, 1].map((i) => (
                          <div key={i} className="space-y-1.5">
                            <p className="font-mono text-xs text-muted-foreground">
                              {prompts[i]}
                            </p>
                            <ScoreList
                              scores={fills[i].slice(0, 5)}
                              quiet
                              data-testid={`probe-${p.id}-${p.varies[i]}`}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/**
 * Why the page will not run, or null if it will.
 *
 * Both refusals are measurements rather than fussiness: zero masks makes the
 * pipeline throw, and a second mask is silently dropped — the model answers the
 * first and the sentence loses a word with nothing failing.
 */
function maskProblem(count: number, mask: string): string | null {
  if (count === 0) {
    return `There is no ${mask} in the sentence. Press "Insert ${mask}" where the missing word belongs.`;
  }
  if (count > 1) {
    return `There are ${count} masks in the sentence, and this pipeline only fills the first — the rest are dropped without an error. Leave exactly one.`;
  }
  return null;
}
