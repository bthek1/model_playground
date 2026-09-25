// Translation — **one language pair at a time**, which is the honest version of
// this page rather than a reduced one.
//
// The rule the whole page is arranged around, and the one it is most likely to
// be misread as breaking:
//
//   **A pair is a model, so changing the pair is a LOAD.** There is no language
//   argument in this task anywhere — a Marian checkpoint carries its own
//   direction and `tr(text)` takes nothing else. So the direction control is a
//   *model selector*, it sits in SELECT beside the download it costs, and the
//   page says so before the click. A direction control that looked like a
//   toggle would imply en→de and de→en were the same 200 MB.
//
// And the trade-off that justifies the design, in numbers rather than prose: one
// Marian pair is ~209 MB against 894.6 MB for NLLB-200-distilled-600M, the
// smallest multilingual model with a browser export — which is itself over this
// repo's feasibility bar, so it is quoted here and not offered.
//
// The CPU path costs 2.7x what a uniform q8 would, and that is a measurement:
// a Marian decoder cannot be quantized on the WASM provider bundled with
// Transformers.js 4.2.0 (the session does not open at all). See
// `SEQ2SEQ_WASM_DTYPES` in `model/backend.ts`.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Languages, Loader2 } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useTranslate } from "@/hooks/useTranslate";
import { formatBytes, sizeEstimate } from "@/model/size";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DEFAULT_TRANSLATION_MODEL,
  NLLB_BYTES,
  TRANSLATION_MODELS,
  TRANSLATION_SAMPLES,
} from "@/text/catalogue";

export const Route = createFileRoute("/translation")({
  component: TranslationPage,
});

/** One finished translation, captured at run time. */
interface RunRecord {
  /** The source text as it was when the button was pressed. */
  source: string;
  translation: string;
  /** The direction that actually produced it. */
  direction: string;
}

function TranslationPage() {
  const session = useModelSelection({
    routeKey: "translation",
    models: TRANSLATION_MODELS,
    fallback:
      TRANSLATION_MODELS.find((m) => m.id === DEFAULT_TRANSLATION_MODEL) ??
      TRANSLATION_MODELS[0],
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
  } = useTranslate(model.id);
  useCacheRefresh(session, ready);

  const samples = TRANSLATION_SAMPLES[model.source] ?? [];
  const [text, setText] = useState(() => samples[0]?.text ?? "");
  const [ran, setRan] = useState<RunRecord | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  const size = useMemo(
    () => sizeEstimate(model.params, model.bytes),
    [model.params, model.bytes],
  );

  /**
   * How many of this pair fit inside one multilingual download.
   *
   * Deliberately **per pair, not the whole catalogue**: nobody downloads all
   * six directions, so summing them is the wrong comparison — and it happens to
   * come out at 1.6 GB, which reads as an argument *against* the design. The
   * real claim is that someone who wants one pair pays a fraction of NLLB.
   */
  const fraction = useMemo(
    () => Math.round(NLLB_BYTES / Math.max(1, size.fp16)),
    [size.fp16],
  );

  async function translate() {
    if (!ready) return;
    const captured = text.trim();
    if (!captured) return;
    const out = await run(captured);
    setRan({ source: captured, translation: out, direction: model.direction });
  }

  return (
    <ModelPage
      icon={Languages}
      title="Translation"
      description={
        <>
          One language pair at a time, running in your browser. A Marian
          checkpoint <em>is</em> its direction — there is no language setting to
          change — so picking a direction below picks a model, and{" "}
          <strong>switching it is another download</strong>.
        </>
      }
      labels={{ select: "Direction", run: "Source text", output: "Translation" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={TRANSLATION_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            onEvict={(m) => void session.evict(m.id)}
          />
          {/* The most likely misreading of this page, stated where the control
              that invites it lives. */}
          <p
            data-testid="pair-is-a-download"
            className="text-xs leading-snug text-muted-foreground"
          >
            Each direction is a separate checkpoint, so changing it above is a
            new {formatBytes(size.fp16)}–{formatBytes(size.q8)} download rather
            than a setting. Nothing is fetched until you press Load.
          </p>
          <p
            data-testid="specialists-note"
            className="text-xs leading-snug text-muted-foreground"
          >
            The smallest multilingual model with a browser export,
            NLLB-200-distilled-600M, is {formatBytes(NLLB_BYTES)} — about{" "}
            <strong>{fraction}x</strong> this pair — and it is past this app's
            size bar, which is why it is quoted here and not offered. One
            specialist per direction is the cheaper answer as long as you want
            one direction.
          </p>
          <p
            data-testid="wasm-cost-note"
            className="text-xs leading-snug text-muted-foreground"
          >
            The CPU (WASM) download is the larger of the two, which is unusual
            and is measured: this decoder cannot be quantized on the ONNX
            Runtime build shipped in the browser — the session does not open at
            all — so it loads at full precision there.
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
          disabledHint="Load a direction to translate. You can write the text first."
          controls={
            <Button
              disabled={!ready || running || text.trim().length === 0}
              onClick={() => void translate().catch(() => {})}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Translating…
                </>
              ) : (
                <>
                  <Languages className="size-4" /> Translate
                </>
              )}
            </Button>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="tr-text" className="sr-only">
              Source text
            </label>
            <textarea
              id="tr-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              className="min-h-32 w-full flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder={`Write ${model.source === "en" ? "English" : model.direction.split(" → ")[0]} text to translate…`}
            />

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Samples in {model.direction.split(" → ")[0]} — picking one fills
                the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {samples.map((s) => (
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
          title="Translation"
          description={`Produced by this checkpoint alone — one direction, no language argument anywhere in the call.`}
          meta={ran ? <span>{ran.direction}</span> : undefined}
          running={running}
          runningLabel="Translating…"
          error={runError}
          empty="Write something and press Translate — the output appears here, labelled with the direction that produced it."
        >
          {ran && (
            <div className="space-y-4" data-testid="translated">
              <p
                data-testid="translation-text"
                className="text-sm leading-relaxed whitespace-pre-wrap"
              >
                {ran.translation}
              </p>
              {/* The source captured *inside* the run, so editing the box
                  afterwards cannot relabel a result already on screen. */}
              <div className="space-y-1 border-t pt-3">
                <p className="text-xs font-medium text-muted-foreground">
                  {ran.direction}
                </p>
                <p
                  data-testid="ran-source"
                  className="text-xs leading-snug text-muted-foreground italic"
                >
                  “{ran.source}”
                </p>
              </div>
              {ran.translation.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  The model returned nothing. That is a failure rather than an
                  answer — try loading the direction again.
                </p>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}
