// Text Generation — and per §3.8 **the best page here is not a chatbot, it is
// the decoding strategies made interactive**. Greedy against sampling,
// temperature and `top_p` as live controls, and the repetition loop visible on
// the same prompt.
//
// Four things it is built around:
//
//   **It streams, always.** The first token arrives in about a second and the
//   rest follows steadily; waiting for a complete 96-token answer feels like a
//   hang at identical throughput. `partial` carries the text so far against the
//   request id — Machine A stays `ready`, `running` stays an inflight count, and
//   a partial whose request already settled is dropped by `useModelWorker`. All
//   three came free from #30's decision to put `partial` in the *shared*
//   envelope; this page is the claim that decision made, and it needed nothing
//   the envelope did not have.
//
//   **The decoding controls are INPUT and they spend.** Temperature, `top_p`,
//   `top_k`, greedy-vs-sampling and the repetition penalty cannot re-derive from
//   a finished generation — the model must run again. So changing one runs
//   nothing, and the page says the next GENERATE is a real inference. Same shape
//   as `/video-text-to-text`'s reverse toggle.
//
//   **The comparison is the page.** Two generations of the same prompt under two
//   settings, side by side, is what makes a repetition loop attributable to
//   greedy decoding rather than to the model. Two generations is two inferences
//   and the page says so before the click.
//
//   **`shader-f16` is the gate, not "has a GPU".** Two of the three entries load
//   f16 weights, and an adapter without the feature loads them, reports `ready`,
//   and fails on the first operator of every run — after the download is paid
//   for. The picker is gated on `useBackendProbe({ requireShaderF16: true })`,
//   and the worker asks the same question again at load time through
//   `pickBackendForF16`, because the default entry has a working CPU fallback
//   and so its row is legitimately enabled either way.
//
// Four-slot page pattern — docs/standards/model-page-pattern.md.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Sparkles, Wand2 } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelPicker } from "@/components/model/ModelPicker";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useTextGen } from "@/hooks/useTextGen";
import { useBackendProbe } from "@/model/useBackendProbe";
import { useCacheRefresh, useModelSelection } from "@/model/useModelSelection";
import {
  DECODING_PRESETS,
  DEFAULT_TEXTGEN_MODEL,
  TEXTGEN_MODELS,
  TEXTGEN_SAMPLES,
} from "@/text/catalogue";
import type { Decoding } from "@/text/textgenTypes";

export const Route = createFileRoute("/text-generation")({
  component: TextGenerationPage,
});

/** One finished generation, labelled with what actually produced it. */
interface RunRecord {
  prompt: string;
  text: string;
  tokens: number;
  ms: number;
  decoding: Decoding;
  modelLabel: string;
}

/** How a run's settings read on screen. Short, because it sits in a caption. */
function describe(d: Decoding): string {
  if (!d.doSample) {
    return `greedy · rep ${d.repetitionPenalty.toFixed(2)} · ≤${d.maxNewTokens} tok`;
  }
  return `T=${d.temperature.toFixed(2)} · top_p=${d.topP.toFixed(2)}${
    d.topK > 0 ? ` · top_k=${d.topK}` : ""
  } · rep ${d.repetitionPenalty.toFixed(2)} · ≤${d.maxNewTokens} tok`;
}

function TextGenerationPage() {
  const session = useModelSelection({
    routeKey: "text-generation",
    models: TEXTGEN_MODELS,
    fallback:
      TEXTGEN_MODELS.find((m) => m.id === DEFAULT_TEXTGEN_MODEL) ??
      TEXTGEN_MODELS[0],
  });
  const model = session.model;
  // Not "has a GPU": two entries load f16 weights, and an adapter without
  // `shader-f16` fails on the first operator of every run after paying for the
  // download. #30's finding.
  const backendProbe = useBackendProbe({ requireShaderF16: true });

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
    partial,
  } = useTextGen(model.id);
  useCacheRefresh(session, ready);

  const [prompt, setPrompt] = useState(TEXTGEN_SAMPLES[0].text);
  const [decoding, setDecoding] = useState<Decoding>(
    DECODING_PRESETS[0].decoding,
  );
  const [ran, setRan] = useState<RunRecord | null>(null);
  /** The other half of the head-to-head, when the user asks for one. */
  const [compare, setCompare] = useState<RunRecord | null>(null);

  const loadError = status === "error" ? error : null;
  const runError = status === "error" ? null : error;

  /**
   * Entries the machine cannot run because the adapter has no `shader-f16`.
   *
   * Derived rather than hard-coded, so an entry that stops needing f16 weights
   * drops out of the note by editing its catalogue row.
   */
  const f16Gated = useMemo(
    () =>
      backendProbe === "wasm"
        ? TEXTGEN_MODELS.filter(
            (m) => m.requireShaderF16 && !(m.backends ?? []).includes("wasm"),
          )
        : [],
    [backendProbe],
  );

  const activePreset = useMemo(
    () =>
      DECODING_PRESETS.find(
        (p) => JSON.stringify(p.decoding) === JSON.stringify(decoding),
      )?.id ?? null,
    [decoding],
  );

  async function generate(): Promise<RunRecord | null> {
    if (!ready) return null;
    const captured = prompt.trim();
    if (!captured) return null;
    // Captured inside the run, so editing the box or a slider afterwards cannot
    // relabel an answer already on screen.
    const settings = { ...decoding };
    const out = await run(captured, settings);
    return {
      prompt: captured,
      text: out.text,
      tokens: out.tokens,
      ms: out.ms,
      decoding: settings,
      modelLabel: model.label,
    };
  }

  async function once() {
    const record = await generate();
    if (record) {
      setRan(record);
      setCompare(null);
    }
  }

  /**
   * The same prompt, twice, under the current settings and greedy.
   *
   * Sequential rather than concurrent: two overlapping requests into one ONNX
   * session is not a guarantee worth relying on, and the point is the
   * comparison rather than the wall-clock.
   */
  async function compareRuns() {
    const a = await generate();
    if (!a) return;
    setRan(a);
    const greedy = { ...decoding, doSample: false };
    const out = await run(a.prompt, greedy);
    setCompare({
      prompt: a.prompt,
      text: out.text,
      tokens: out.tokens,
      ms: out.ms,
      decoding: greedy,
      modelLabel: model.label,
    });
  }

  const set = <K extends keyof Decoding>(key: K, value: Decoding[K]) =>
    setDecoding((d) => ({ ...d, [key]: value }));

  return (
    <ModelPage
      icon={Sparkles}
      title="Text Generation"
      description={
        <>
          A language model writing, in your browser — and the{" "}
          <strong>decoding strategy</strong> as the thing you actually change.
          Greedy decoding is reproducible and loops; sampling reads better and
          cannot be repeated. The interesting question is not which model, it is
          which strategy.
        </>
      }
      labels={{ run: "Prompt", output: "Continuation" }}
      select={
        <div className="space-y-2">
          <ModelPicker
            models={TEXTGEN_MODELS}
            value={model.id}
            onChange={session.setModel}
            disabled={loading || running}
            cached={session.cached}
            backend={backendProbe}
            onEvict={(m) => void session.evict(m.id)}
          />
          <p className="text-xs leading-snug text-muted-foreground">
            {model.instruct
              ? "Instruction-tuned, so your prompt is wrapped in its chat template and it answers."
              : "A bare language model — no instruction tuning. It continues your text rather than answering it."}{" "}
            Trained on {model.domain}.
          </p>
          {backendProbe === "wasm" && f16Gated.length > 0 && (
            // #30's finding, and the note is about the **machine** rather than
            // the selected row — which is what makes it useful, because the rows
            // it explains are the disabled ones the user cannot select to read a
            // message on. "Resolved to wasm" would be misleading on a machine
            // that plainly has a GPU, so the missing feature is named.
            <p
              data-testid="shader-f16-note"
              className="text-xs leading-snug text-amber-600 dark:text-amber-500"
            >
              {f16Gated.map((m) => m.label).join(" and ")}{" "}
              {f16Gated.length === 1 ? "is" : "are"} disabled above: they load
              f16 weights and need the GPU feature{" "}
              <code className="font-mono">shader-f16</code>, which this
              browser's adapter does not report. Without it the download
              succeeds, the model reports ready, and then every run fails on its
              first operator — so the row is withheld rather than offered.
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
          disabledHint="Load a model to generate. You can write the prompt and set the strategy first."
          controls={
            <>
              <Button
                disabled={!ready || running || prompt.trim().length === 0}
                onClick={() => void once().catch(() => {})}
              >
                {running ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Generating…
                  </>
                ) : (
                  <>
                    <Wand2 className="size-4" /> Generate
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                disabled={!ready || running || prompt.trim().length === 0}
                data-testid="compare-run"
                onClick={() => void compareRuns().catch(() => {})}
                title="The same prompt under these settings and under greedy decoding."
              >
                Compare against greedy
              </Button>
              <span
                data-testid="spend-note"
                className="basis-full text-xs text-muted-foreground"
              >
                Changing the strategy below runs nothing — it cannot be derived
                from a finished answer, so the next Generate is a real inference.
                Comparing is <strong>two</strong>.
              </span>
            </>
          }
        >
          <div className="flex min-h-0 flex-1 flex-col gap-3">
            <label htmlFor="tg-prompt" className="sr-only">
              Prompt
            </label>
            <textarea
              id="tg-prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              className="min-h-24 w-full resize-none rounded-md border bg-background px-3 py-2 text-sm"
              placeholder="Write a prompt…"
            />

            <div className="space-y-1.5">
              <p className="text-xs text-muted-foreground">
                Prompts — picking one fills the box and runs nothing.
              </p>
              <div className="flex flex-wrap gap-1.5">
                {TEXTGEN_SAMPLES.map((s) => (
                  <Button
                    key={s.id}
                    size="sm"
                    variant="outline"
                    disabled={running}
                    onClick={() => setPrompt(s.text)}
                    title={s.hint}
                  >
                    {s.label}
                  </Button>
                ))}
              </div>
            </div>

            <div className="space-y-2 border-t pt-3">
              <p className="text-xs font-medium">Strategy</p>
              <div className="flex flex-wrap gap-1.5">
                {DECODING_PRESETS.map((p) => (
                  <Button
                    key={p.id}
                    size="sm"
                    variant={activePreset === p.id ? "default" : "outline"}
                    aria-pressed={activePreset === p.id}
                    data-testid={`preset-${p.id}`}
                    disabled={running}
                    onClick={() => setDecoding(p.decoding)}
                    title={p.hint}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>

              <div className="flex flex-wrap items-center gap-2 pt-1">
                <Button
                  size="sm"
                  variant={decoding.doSample ? "outline" : "default"}
                  aria-pressed={!decoding.doSample}
                  data-testid="mode-greedy"
                  disabled={running}
                  onClick={() => set("doSample", false)}
                >
                  Greedy
                </Button>
                <Button
                  size="sm"
                  variant={decoding.doSample ? "default" : "outline"}
                  aria-pressed={decoding.doSample}
                  data-testid="mode-sample"
                  disabled={running}
                  onClick={() => set("doSample", true)}
                >
                  Sampling
                </Button>
                <span className="text-xs text-muted-foreground">
                  {decoding.doSample
                    ? "Different every time."
                    : "Same text every time — which is what makes a loop the strategy's fault."}
                </span>
              </div>

              <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
                <Slider
                  id="tg-temp"
                  label="Temperature"
                  value={decoding.temperature}
                  min={0.1}
                  max={2}
                  step={0.05}
                  disabled={!decoding.doSample}
                  onChange={(v) => set("temperature", v)}
                />
                <Slider
                  id="tg-topp"
                  label="top_p"
                  value={decoding.topP}
                  min={0.1}
                  max={1}
                  step={0.01}
                  disabled={!decoding.doSample}
                  onChange={(v) => set("topP", v)}
                />
                <Slider
                  id="tg-topk"
                  label="top_k (0 = off)"
                  value={decoding.topK}
                  min={0}
                  max={200}
                  step={5}
                  disabled={!decoding.doSample}
                  onChange={(v) => set("topK", v)}
                />
                <Slider
                  id="tg-rep"
                  label="Repetition penalty"
                  value={decoding.repetitionPenalty}
                  min={1}
                  max={2}
                  step={0.05}
                  onChange={(v) => set("repetitionPenalty", v)}
                />
                <Slider
                  id="tg-max"
                  label="Max new tokens"
                  value={decoding.maxNewTokens}
                  min={16}
                  max={400}
                  step={8}
                  onChange={(v) => set("maxNewTokens", v)}
                />
              </div>
              {!decoding.doSample && (
                <p className="text-xs leading-snug text-muted-foreground">
                  Temperature, <code>top_p</code> and <code>top_k</code> are
                  greyed out because greedy decoding does not sample — they are
                  not sent at all, rather than sent and ignored.
                </p>
              )}
            </div>
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Continuation"
          description="Streamed as it is produced — the first token arrives in about a second, and waiting for the whole answer at the same throughput feels like a hang."
          meta={
            ran ? (
              <span>
                {ran.tokens} tok ·{" "}
                {ran.ms > 0 ? ((ran.tokens / ran.ms) * 1000).toFixed(1) : "0"}{" "}
                tok/s
              </span>
            ) : undefined
          }
          running={running && !partial}
          runningLabel="Starting…"
          error={runError}
          empty="Write a prompt, pick a strategy and press Generate — the continuation streams in here as it is produced."
        >
          {(partial || ran) && (
            <div className="space-y-5" data-testid="generated">
              {/* While a run is in flight the stream is the result; `partial`
                  is cleared the moment the result lands, so the two are never
                  both on screen. */}
              {partial ? (
                <div className="space-y-1" data-testid="stream">
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {partial.text}
                    <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-foreground/60 align-text-bottom" />
                  </p>
                  <p className="font-mono text-xs text-muted-foreground tabular-nums">
                    {partial.tokens} tokens
                  </p>
                </div>
              ) : (
                ran && (
                  <div className="space-y-2">
                    <p
                      data-testid="generated-text"
                      className="text-sm leading-relaxed whitespace-pre-wrap"
                    >
                      {ran.text}
                    </p>
                    {/* The settings captured at run time — editing a slider
                        afterwards must not relabel this. */}
                    <p
                      data-testid="ran-settings"
                      className="font-mono text-xs text-muted-foreground"
                    >
                      {describe(ran.decoding)}
                    </p>
                    <p
                      data-testid="ran-prompt"
                      className="text-xs leading-snug text-muted-foreground italic"
                    >
                      “{ran.prompt}”
                    </p>
                  </div>
                )
              )}

              {compare && (
                <div className="space-y-2 border-t pt-4" data-testid="comparison">
                  <p className="text-sm font-medium">Same prompt, greedy</p>
                  <p
                    data-testid="comparison-text"
                    className="text-sm leading-relaxed whitespace-pre-wrap"
                  >
                    {compare.text}
                  </p>
                  <p className="font-mono text-xs text-muted-foreground">
                    {describe(compare.decoding)}
                  </p>
                  <p className="text-xs leading-snug text-muted-foreground">
                    Two generations of one prompt. Whatever differs between them
                    is the <strong>strategy</strong>, not the model — which is
                    the only way a repetition loop can be attributed to greedy
                    decoding rather than blamed on the weights.
                  </p>
                </div>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/** A labelled range with its value shown. Local — nothing else needs it yet. */
function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  disabled = false,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <div className={disabled ? "space-y-1 opacity-50" : "space-y-1"}>
      <label htmlFor={id} className="text-xs font-medium">
        {label}: <span className="tabular-nums">{value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
