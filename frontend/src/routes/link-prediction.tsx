// Link prediction on Cora: train on a graph with 15 % of its citations removed,
// then score the pairs the model was never shown.
//
// The sibling of /graph, and the same four bands, but the question is different
// and so is the thing that can go wrong. On /graph the failure mode is a wrong
// aggregation producing a plausible accuracy curve; here it is **leakage**, and
// it fails upward: an encoder that can still aggregate over a held-out citation
// scores it from memory and the test number goes up. lib/edgeSplit.ts is what
// prevents that, and the page states the split it is reporting against.
//
// Two controls, and the difference between them is the page-pattern rule:
// the **held-out fraction** changes the question (a different split is a
// different graph and a different layout), so it re-loads and re-trains; the
// **top-k slider** only re-reads candidates already in hand, so it re-derives on
// the main thread without a press. Same rule /vad's threshold follows.

import type { EChartsOption } from "echarts";
import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Play, Share2, Square } from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import { GraphCanvas } from "@/components/graph/GraphCanvas";
import { DeviceStatus } from "@/components/model/DeviceStatus";
import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { PanZoom } from "@/components/viz/PanZoom";
import { useLinkPrediction } from "@/hooks/useLinkPrediction";
import { useTheme } from "@/hooks/useTheme";
import { useWebGPU } from "@/hooks/useWebGPU";
import { CORA_CLASSES } from "@/lib/cora";
import { getCSSVar } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { GNN_ARCHITECTURES, type GnnArch } from "@/webgpu/gnn";
import { sigmoid } from "@/webgpu/linkPredictor";
import { scorePair } from "@/webgpu/linkSession";

const EChart = lazy(() => import("@/components/charts/EChart"));

export const Route = createFileRoute("/link-prediction")({
  component: LinkPredictionPage,
});

const ARCHES: GnnArch[] = ["gcn", "sage", "gin", "gat"];

/** Side of the square the graph is painted into before PanZoom scales it. */
const GRAPH_RENDER_PX = 720;

/** How many candidate non-edges the worker brings back to draw from. */
const MAX_CANDIDATES = 60;

const HELD_OUT = [0.05, 0.1, 0.15, 0.2, 0.3] as const;

/**
 * Close to Kipf's graph auto-encoder recipe, and deliberately **not** `/graph`'s.
 *
 * The node-classification page regularises hard — dropout 0.5, weight decay
 * 5e-4 — because it fits 1433 features from 140 labelled papers and overfits
 * immediately. This page has ~4500 supervised citations, so the same settings
 * are simply too much model-starving, and they cost it 0.19 of AUC. Measured on
 * the same split and seed:
 *
 *     /graph's recipe (0.5 dropout, 5e-4 decay, 32-wide, 150 ep)   0.737
 *     GAE's           (no dropout,  no decay,   32→16,  150 ep)    0.879
 *     …at 250 epochs                                               0.906
 *     …plus light dropout of 0.2                                   0.925
 *
 * The published GAE number on Cora is 0.910, so the last row is the one to
 * ship. These are measurements, not precautions — see docs/roadmaps/graph.md.
 */
const DEFAULTS = {
  layers: 2,
  hidden: 32,
  embedding: 16,
  learningRate: 0.01,
  weightDecay: 0,
  dropout: 0.2,
  epochs: 250,
  seed: 42,
};

function LinkPredictionPage() {
  const { capabilities, loading: probing } = useWebGPU();
  const session = useLinkPrediction();
  const [arch, setArch] = useState<GnnArch>("gcn");
  const [testFrac, setTestFrac] = useState(0.1);
  const [topK, setTopK] = useState(20);
  const [picked, setPicked] = useState<number[]>([]);
  const [zoom, setZoom] = useState(1);

  const { summary, status, training, metrics, result } = session;
  const ready = status === "ready" && summary != null;
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  // Re-derivation, not a re-run: the worker already returned MAX_CANDIDATES
  // ranked pairs, and the slider is choosing how many of them to draw.
  const drawn = useMemo(
    () => result?.candidates.slice(0, topK * 2) ?? undefined,
    [result, topK],
  );

  const pair = picked.length === 2 ? ([picked[0], picked[1]] as const) : null;
  const pairScore =
    pair && result
      ? scorePair(result.embedding, result.embeddingDim, pair[0], pair[1])
      : null;

  const start = () =>
    session.start({ ...DEFAULTS, arch, topK: MAX_CANDIDATES });

  const pick = (node: number | null) => {
    if (node == null) {
      setPicked([]);
      return;
    }
    // Two clicks make a pair; a third starts a new one, which is less surprising
    // than silently replacing one end of the pair already on screen.
    setPicked((current) =>
      current.length >= 2 ? [node] : [...current, node],
    );
  };

  return (
    <ModelPage
      icon={Share2}
      title="Link Prediction"
      description={
        <>
          The same <strong>Cora</strong> citation graph as node classification,
          with <strong>{Math.round(testFrac * 100)}% of its citations removed</strong>{" "}
          before training. The model never sees them, and is then asked to score
          every pair of papers that is not linked — the dashed lines are
          predictions, not data.
        </>
      }
      labels={{
        select: "Architecture",
        load: "Split & device",
        run: "Train",
        output: "Predicted links",
      }}
      select={<ArchPicker value={arch} onChange={setArch} disabled={training} />}
      load={
        <div className="space-y-3">
          <DeviceStatus capabilities={capabilities} loading={probing} />
          <HeldOutControl
            value={testFrac}
            onChange={(next) => {
              setTestFrac(next);
              // A different split is a different graph: the layout moves with
              // it, so this is a load, not a hyperparameter.
              if (status !== "idle") session.load({ testFrac: next });
            }}
            disabled={training || status === "loading"}
          />
          <SplitStatus session={session} testFrac={testFrac} />
        </div>
      }
      run={
        <div className="flex min-h-0 flex-1 flex-col gap-4">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <Fact label="Layers" value={`${DEFAULTS.layers}`} />
            <Fact label="Embedding" value={`${DEFAULTS.embedding}`} />
            <Fact label="Epochs" value={`${DEFAULTS.epochs}`} />
            <Fact label="Dropout" value={`${DEFAULTS.dropout}`} />
          </dl>

          <p className="text-xs text-muted-foreground">
            Each epoch scores every training citation against a freshly sampled
            non-citation, so the model cannot learn the sampler. The score for a
            pair is the dot product of the two papers&rsquo; embeddings; the
            number reported is <strong>AUC</strong> — the probability a real
            citation outranks a pair that is not one.
          </p>

          <ErrorNote message={session.trainError} />

          <div className="sticky bottom-0 mt-2 flex flex-wrap items-center gap-2 border-t bg-background/80 pt-3 backdrop-blur">
            <Button onClick={start} disabled={!ready || training}>
              {training ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Training…
                </>
              ) : (
                <>
                  <Play className="size-4" /> Train
                </>
              )}
            </Button>
            <Button variant="outline" onClick={session.stop} disabled={!training}>
              <Square className="size-4" /> Stop
            </Button>
            {latest && (
              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                epoch {latest.epoch + 1}/{latest.totalEpochs}
              </span>
            )}
          </div>
        </div>
      }
      output={
        <OutputPanel
          title="Cora"
          meta={
            summary
              ? `${summary.nTrainEdges} kept · ${summary.nTestEdges} held out`
              : undefined
          }
          description={
            result
              ? "Solid lines are the citations the model trained on. Dashed lines are the pairs it scores highest among those that are not citations at all. Click two papers to score them yourself."
              : undefined
          }
          running={training && !result}
          runningLabel="Training…"
          error={null}
          empty={
            ready
              ? "Press Train to fit an encoder on the reduced graph. The dashed lines appear when it finishes — the candidates are scored once, over every pair Cora does not contain."
              : "Load the graph to draw it."
          }
        >
          {ready && summary && result && (
            <div className="flex min-h-0 flex-1 flex-col gap-3">
              <PanZoom
                className="relative min-h-56 w-full flex-1 rounded-md border bg-background"
                onScaleChange={setZoom}
              >
                <div
                  className="flex"
                  style={{ width: GRAPH_RENDER_PX, height: GRAPH_RENDER_PX }}
                >
                  <GraphCanvas
                    nNodes={summary.nNodes}
                    rowPtr={summary.rowPtr}
                    colIdx={summary.colIdx}
                    x={summary.x}
                    y={summary.y}
                    labels={summary.labels}
                    predictions={null}
                    trainMask={EMPTY_MASK}
                    colorBy="true"
                    predicted={drawn}
                    selected={pair}
                    onPickNode={pick}
                    zoom={zoom}
                  />
                </div>
              </PanZoom>

              <TopKControl value={topK} onChange={setTopK} max={MAX_CANDIDATES} />
              <PairReadout
                picked={picked}
                score={pairScore}
                labels={summary.labels}
              />
              <CandidateList
                candidates={result.candidates}
                scores={result.candidateScores}
                labels={summary.labels}
                selected={pair}
                onPick={(u, v) => setPicked([u, v])}
              />

              {latest && (
                <>
                  <Scoreboard
                    testAuc={latest.testAuc}
                    valAuc={latest.valAuc}
                    testAp={latest.testAp}
                    elapsedMs={session.elapsedMs}
                    backend={summary.backend}
                  />
                  <Curves metrics={metrics} />
                </>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/** No node is "labelled" here — the supervision is on edges. */
const EMPTY_MASK = new Uint8Array(0);

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt>{label}</dt>
      <dd className="font-mono text-foreground tabular-nums">{value}</dd>
    </div>
  );
}

function ArchPicker({
  value,
  onChange,
  disabled,
}: {
  value: GnnArch;
  onChange: (arch: GnnArch) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {ARCHES.map((arch) => {
        const meta = GNN_ARCHITECTURES[arch];
        const selected = arch === value;
        return (
          <Button
            key={arch}
            variant={selected ? "default" : "outline"}
            disabled={disabled}
            aria-pressed={selected}
            onClick={() => onChange(arch)}
            className="h-auto w-full flex-col items-start gap-0.5 px-3 py-2 text-left whitespace-normal"
          >
            <span className="flex w-full items-baseline gap-2 text-sm font-medium">
              {meta.label}
              <span
                className={cn(
                  "font-mono text-[0.65rem] font-normal",
                  selected
                    ? "text-primary-foreground/70"
                    : "text-muted-foreground",
                )}
              >
                {meta.aggregation}
              </span>
            </span>
          </Button>
        );
      })}
    </div>
  );
}

function HeldOutControl({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (value: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-sm font-medium">Held out for test</span>
      <div className="flex flex-wrap gap-1">
        {HELD_OUT.map((frac) => (
          <Button
            key={frac}
            size="sm"
            variant={frac === value ? "default" : "outline"}
            aria-pressed={frac === value}
            disabled={disabled}
            onClick={() => onChange(frac)}
          >
            {Math.round(frac * 100)}%
          </Button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        These citations are removed from the graph before training, not just
        hidden from the loss — otherwise the model averages the two papers
        together and scores the pair from memory.
      </p>
    </div>
  );
}

function SplitStatus({
  session,
  testFrac,
}: {
  session: ReturnType<typeof useLinkPrediction>;
  testFrac: number;
}) {
  const { status, summary, loadError } = session;

  if (status === "error") {
    return (
      <ErrorNote
        message={loadError}
        action={
          <Button size="sm" variant="outline" onClick={session.retry}>
            Retry
          </Button>
        }
      />
    );
  }

  if (status === "loading") {
    return (
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        Splitting the citations and laying the graph out…
      </p>
    );
  }

  if (status === "ready" && summary) {
    return (
      <p data-testid="model-ready" className="text-xs text-muted-foreground">
        {summary.nTrainEdges} citations kept · {summary.nValEdges} validation ·{" "}
        {summary.nTestEdges} test · laid out in {Math.round(summary.layoutMs)} ms
        · training runs on{" "}
        <span className="font-medium">
          {summary.backend === "webgpu" ? "the GPU" : "the CPU"}
        </span>
        {summary.rescued > 0 && (
          <>
            {" "}
            · {summary.rescued} kept to avoid isolating a paper
          </>
        )}
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      <Button size="sm" onClick={() => session.load({ testFrac })}>
        Load graph
      </Button>
      <p className="text-xs text-muted-foreground">
        161 KB, bundled with the app — no download. The wait is the split and the
        force-directed layout, about a second.
      </p>
    </div>
  );
}

function TopKControl({
  value,
  onChange,
  max,
}: {
  value: number;
  onChange: (value: number) => void;
  max: number;
}) {
  return (
    <div className="space-y-1">
      <label
        htmlFor="link-topk"
        className="flex items-baseline justify-between text-sm font-medium"
      >
        Predicted links drawn
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          top {value}
        </span>
      </label>
      <input
        id="link-topk"
        type="range"
        min={0}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
      <p className="text-xs text-muted-foreground">
        Re-draws the ranking already computed — it does not re-run the model.
      </p>
    </div>
  );
}

function PairReadout({
  picked,
  score,
  labels,
}: {
  picked: number[];
  score: number | null;
  labels: Uint8Array;
}) {
  if (picked.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Click a paper, then another, to see what the model scores the pair.
      </p>
    );
  }

  const topic = (node: number) =>
    CORA_CLASSES[labels[node]] ?? `class ${labels[node]}`;

  if (picked.length === 1 || score == null) {
    return (
      <p className="text-xs text-muted-foreground">
        Paper <span className="font-mono">#{picked[0]}</span> ({topic(picked[0])})
        — click another to score the pair.
      </p>
    );
  }

  return (
    <p
      data-testid="pair-score"
      className="text-xs text-muted-foreground"
    >
      <span className="font-mono">#{picked[0]}</span> ({topic(picked[0])}) ↔{" "}
      <span className="font-mono">#{picked[1]}</span> ({topic(picked[1])}) ·
      score{" "}
      <span className="font-mono text-foreground tabular-nums">
        {score.toFixed(2)}
      </span>{" "}
      · p ={" "}
      <span className="font-mono text-foreground tabular-nums">
        {sigmoid(score).toFixed(3)}
      </span>
    </p>
  );
}

/**
 * The highest-scoring pairs, as text.
 *
 * The drawing alone is not enough, and the reason is a property of the task
 * rather than of the rendering: the pairs a link predictor ranks highest are the
 * ones that already share neighbours, so a force-directed layout has put them on
 * top of each other and the dashed line between them is a few pixels long. The
 * list is what makes the answer readable; clicking a row rings the pair on the
 * canvas, which is what makes it locatable.
 */
function CandidateList({
  candidates,
  scores,
  labels,
  selected,
  onPick,
}: {
  candidates: Uint32Array;
  scores: Float32Array;
  labels: Uint8Array;
  selected: readonly [number, number] | null;
  onPick: (u: number, v: number) => void;
}) {
  const rows = Math.min(6, scores.length);
  if (rows === 0) return null;

  return (
    <div className="space-y-1">
      <p className="text-xs font-medium">
        Highest-scoring pairs that are not citations
      </p>
      <ul className="divide-y rounded-md border text-xs">
        {Array.from({ length: rows }, (_, i) => {
          const u = candidates[2 * i];
          const v = candidates[2 * i + 1];
          const isSelected = selected?.[0] === u && selected?.[1] === v;
          return (
            <li key={`${u}-${v}`}>
              <button
                type="button"
                onClick={() => onPick(u, v)}
                className={cn(
                  "flex w-full items-baseline justify-between gap-2 px-2 py-1 text-left hover:bg-muted",
                  isSelected && "bg-muted",
                )}
              >
                <span className="truncate">
                  <span className="font-mono">#{u}</span>{" "}
                  <span className="text-muted-foreground">
                    ({CORA_CLASSES[labels[u]]})
                  </span>{" "}
                  ↔ <span className="font-mono">#{v}</span>{" "}
                  <span className="text-muted-foreground">
                    ({CORA_CLASSES[labels[v]]})
                  </span>
                </span>
                {/* The raw score, not σ(score): the top pairs all saturate to
                    1.000 and a column of identical numbers ranks nothing. */}
                <span className="font-mono tabular-nums">
                  {scores[i].toFixed(1)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Scoreboard({
  testAuc,
  valAuc,
  testAp,
  elapsedMs,
  backend,
}: {
  testAuc: number;
  valAuc: number;
  testAp: number;
  elapsedMs: number | null;
  backend: string;
}) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
      <Score label="Test AUC" value={testAuc.toFixed(3)} />
      <Score label="Validation AUC" value={valAuc.toFixed(3)} />
      <Score label="Test AP" value={testAp.toFixed(3)} />
      <Score
        label="Trained in"
        value={elapsedMs == null ? "—" : `${(elapsedMs / 1000).toFixed(1)}s (${backend})`}
      />
    </dl>
  );
}

function Score({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono text-lg tabular-nums">{value}</dd>
    </div>
  );
}

// ECharts cannot read CSS variables, so the design tokens are resolved here and
// re-resolved when the theme flips. `theme` is a dependency only to force that.
function useChartTheme() {
  const { theme } = useTheme();
  return useMemo(
    () => ({
      axis: getCSSVar("mutedForeground"),
      series: [getCSSVar("chart2"), getCSSVar("chart4")],
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [theme],
  );
}

function Curves({
  metrics,
}: {
  metrics: { epoch: number; loss: number; testAuc: number }[];
}) {
  const colors = useChartTheme();
  const option = useMemo<EChartsOption>(
    () => ({
      color: colors.series,
      grid: { left: 44, right: 44, top: 24, bottom: 32 },
      legend: { top: 0, textStyle: { color: colors.axis } },
      xAxis: {
        type: "value",
        name: "epoch",
        min: 1,
        axisLabel: { color: colors.axis },
        nameTextStyle: { color: colors.axis },
      },
      yAxis: [
        {
          type: "value",
          name: "loss",
          min: 0,
          axisLabel: { color: colors.axis },
          nameTextStyle: { color: colors.axis },
        },
        {
          type: "value",
          name: "AUC",
          // Not from zero: a link predictor starts at chance, and the whole
          // movement worth seeing lives between 0.5 and 1.
          min: 0.4,
          max: 1,
          axisLabel: { color: colors.axis },
          nameTextStyle: { color: colors.axis },
        },
      ],
      tooltip: { trigger: "axis" },
      animation: false,
      series: [
        {
          name: "loss",
          type: "line",
          showSymbol: false,
          data: metrics.map((m) => [m.epoch + 1, m.loss]),
        },
        {
          name: "test AUC",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          data: metrics.map((m) => [m.epoch + 1, m.testAuc]),
        },
      ],
    }),
    [metrics, colors],
  );

  return (
    <div>
      <p className="text-xs font-medium">Loss and test AUC per epoch</p>
      {/* EChart fills 100% of its parent, so the parent owes it a height —
          without one the chart collapses to a strip of stray axis labels. */}
      <div className="h-48 w-full">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              Loading chart…
            </div>
          }
        >
          <EChart option={option} />
        </Suspense>
      </div>
    </div>
  );
}
