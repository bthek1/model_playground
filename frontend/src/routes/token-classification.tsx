// Token Classification — named-entity recognition with the entities marked in
// the user's **own text**, and a redact button that replaces the chosen types
// with a placeholder.
//
// This is the page where "it runs in your browser" stops being a performance
// claim and becomes the point: redacting a document you are not allowed to
// upload is a real reason to want the model on this side of the wire. Nothing
// leaves the tab.
//
// Two rules the page is built around:
//
//   The overlay slices the **original string** by character offset. Rebuilding
//   the text from the model's tokens loses the whitespace between them and the
//   highlight lands a character or two off — a wrong answer that reads as a
//   styling problem. See `text/highlight.ts`.
//
//   Redaction is a **pure derivation** over spans already in hand, so toggling
//   it, or changing which types it removes, re-derives on the main thread and
//   costs nothing. Only GENERATE spends.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Check, Copy, Loader2, ScanText } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { SpanOverlay } from "@/components/text/SpanOverlay";
import { Button } from "@/components/ui/button";
import { useNer } from "@/hooks/useNer";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_NER_MODEL,
  DEFAULT_REDACTED,
  NER_MODELS,
  NER_SAMPLES,
} from "@/text/catalogue";
import { highlight, redact, type EntitySpan } from "@/text/highlight";

export const Route = createFileRoute("/token-classification")({
  component: TokenClassificationPage,
});

/** The result, captured at run time so editing the box cannot restyle it. */
interface RunRecord {
  text: string;
  spans: EntitySpan[];
  modelId: string;
}

function TokenClassificationPage() {
  const session = useModelSelection({
    routeKey: "token-classification",
    models: NER_MODELS,
    fallback:
      NER_MODELS.find((m) => m.id === DEFAULT_NER_MODEL) ?? NER_MODELS[0],
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
    unplaced,
    load,
    retry,
    cancel,
    run,
  } = useNer(model.id);
  useCacheRefresh(session, ready);

  const [text, setText] = useState(NER_SAMPLES[0].text);
  const [ran, setRan] = useState<RunRecord | null>(null);

  // Redaction state. Both of these re-derive from `ran` — neither re-runs.
  const [redacting, setRedacting] = useState(false);
  const [types, setTypes] = useState<Set<string>>(
    () => new Set(DEFAULT_REDACTED),
  );
  const [copied, setCopied] = useState(false);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const marked = useMemo(
    () => (ran ? highlight(ran.text, ran.spans) : null),
    [ran],
  );
  const redactedText = useMemo(
    () => (ran ? redact(ran.text, ran.spans, types) : ""),
    [ran, types],
  );

  const found = useMemo(() => {
    const counts = new Map<string, number>();
    for (const s of ran?.spans ?? []) {
      counts.set(s.label, (counts.get(s.label) ?? 0) + 1);
    }
    return counts;
  }, [ran]);

  async function tag() {
    if (!ready || text.trim().length === 0) return;
    const captured = text;
    const spans = await run(captured);
    setRan({ text: captured, spans, modelId: model.id });
    setCopied(false);
  }

  function toggleType(type: string) {
    setTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  async function copyRedacted() {
    try {
      await navigator.clipboard.writeText(redactedText);
      setCopied(true);
    } catch {
      /* a denied clipboard is not worth an error band; the text is on screen */
    }
  }

  return (
    <ModelPage
      icon={ScanText}
      title="Token Classification"
      description={
        <>
          Find the people, places and organisations in a piece of text, and
          redact them — entirely in your browser, on your GPU (WebGPU) or CPU
          (WASM). The document is never uploaded, which is the whole reason to
          want this one client-side.
        </>
      }
      labels={{ run: "Text", output: "Entities" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={NER_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            This head tags{" "}
            <span className="font-medium">{model.entities.join(", ")}</span> —
            trained on {model.domain}.
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
          disabledHint="Load a model to tag entities. You can paste the text first."
          controls={
            <Button
              disabled={!ready || running || text.trim().length === 0}
              onClick={() => void tag().catch(() => {})}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Tagging…
                </>
              ) : (
                <>
                  <ScanText className="size-4" /> Find entities
                </>
              )}
            </Button>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="ner-text" className="sr-only">
              Text to tag
            </label>
            <textarea
              id="ner-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={7}
              className="min-h-36 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Paste a paragraph…"
            />
            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples — picking one fills the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {NER_SAMPLES.map((s) => (
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
          title="Entities"
          description="The model's spans, marked in your own text. Each one carries its type — colour alone would not be readable for everyone."
          meta={
            ran ? (
              <span className="tabular-nums">
                {ran.spans.length} found
              </span>
            ) : undefined
          }
          running={running}
          runningLabel="Tagging…"
          error={runError}
          empty="Paste some text and press Find entities — the people, places and organisations appear marked in place, with a redact button beside them."
          actions={
            ran && redacting ? (
              <Button size="sm" variant="outline" onClick={() => void copyRedacted()}>
                {copied ? (
                  <>
                    <Check className="size-4" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="size-4" /> Copy redacted text
                  </>
                )}
              </Button>
            ) : undefined
          }
        >
          {ran && marked && (
            <div className="space-y-4">
              <SpanOverlay
                result={marked}
                redacted={redacting ? types : undefined}
              />

              {unplaced.length > 0 && (
                // Never swallowed: the model found something the page could
                // not place in the source, and a silently shorter list of
                // highlights is indistinguishable from a model that found less.
                <p
                  data-testid="unplaced-note"
                  className="text-xs text-amber-600 dark:text-amber-500"
                >
                  {unplaced.length} entit
                  {unplaced.length === 1 ? "y" : "ies"} could not be located in
                  the text and {unplaced.length === 1 ? "is" : "are"} not shown.
                </p>
              )}

              {ran.spans.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  The model found nothing to tag in this text. That is an
                  answer, not a failure — try a sentence with a name in it.
                </p>
              )}

              {ran.spans.length > 0 && (
                <div className="space-y-2 border-t pt-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      variant={redacting ? "default" : "outline"}
                      data-testid="redact-toggle"
                      onClick={() => setRedacting((r) => !r)}
                    >
                      {redacting ? "Show entities" : "Redact"}
                    </Button>
                    <span className="text-xs text-muted-foreground">
                      Re-derived from the spans already in hand — it runs nothing.
                    </span>
                  </div>

                  {redacting && (
                    <div
                      className="flex flex-wrap gap-1.5"
                      data-testid="redact-types"
                    >
                      {[...found.keys()].map((type) => (
                        <Button
                          key={type}
                          size="sm"
                          variant={types.has(type) ? "default" : "outline"}
                          aria-pressed={types.has(type)}
                          onClick={() => toggleType(type)}
                        >
                          {type}
                          <span className="ml-1 opacity-70 tabular-nums">
                            {found.get(type)}
                          </span>
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
