import { createFileRoute } from "@tanstack/react-router";
import type { EChartsOption } from "echarts";
import { Loader2, Play, Share2, Square } from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import {
  CLASS_COLORS,
  type ColorBy,
  GraphCanvas,
} from "@/components/graph/GraphCanvas";
import { DeviceStatus } from "@/components/model/DeviceStatus";
import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { CORA_CLASSES } from "@/lib/cora";
import { cn } from "@/lib/utils";
import { useGraphTraining } from "@/hooks/useGraphTraining";
import { useTheme } from "@/hooks/useTheme";
import { useWebGPU } from "@/hooks/useWebGPU";
import { getCSSVar } from "@/lib/theme";
import { GNN_ARCHITECTURES, type GnnArch } from "@/webgpu/gnn";

const EChart = lazy(() => import("@/components/charts/EChart"));

export const Route = createFileRoute("/graph")({
  component: GraphPage,
});

const ARCHES: GnnArch[] = ["gcn", "sage", "gin", "gat"];
const MIN_LAYERS = 2;
const MAX_LAYERS = 8;

const DEFAULTS = {
  hidden: 16,
  learningRate: 0.01,
  weightDecay: 5e-4,
  dropout: 0.5,
  epochs: 200,
  seed: 42,
};

function GraphPage() {
  const { capabilities, loading: probing } = useWebGPU();
  const session = useGraphTraining();
  const [arch, setArch] = useState<GnnArch>("gcn");
  const [layers, setLayers] = useState(2);
  const [colorBy, setColorBy] = useState<ColorBy>("predicted");

  const { summary, status, training, metrics } = session;
  const ready = status === "ready" && summary != null;
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  const start = () =>
    session.start({ ...DEFAULTS, arch, layers });

  return (
    <ModelPage
      icon={Share2}
      title="Graph Machine Learning"
      description={
        <>
          Semi-supervised node classification on <strong>Cora</strong> — 2708
          machine-learning papers linked by 5278 citations. There is no
          pretrained checkpoint here and nothing to download: the network is
          written as WGSL and trained in front of you, from{" "}
          <strong>20 labelled papers per topic</strong> out of 2708.
        </>
      }
      labels={{ select: "Architecture", load: "Data & device", run: "Train", output: "The graph" }}
      select={
        <ArchPicker value={arch} onChange={setArch} disabled={training} />
      }
      load={
        <div className="space-y-2">
          {/* The GPU answer comes before the data, because it is the one that
              decides which of the two compute paths a run will take. */}
          <DeviceStatus capabilities={capabilities} loading={probing} />
          <DataStatus session={session} />
        </div>
      }
      run={
        <div className="flex min-h-0 flex-1 flex-col gap-4">
          <DepthControl
            layers={layers}
            onChange={setLayers}
            disabled={training}
          />

          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <Fact label="Hidden width" value={`${DEFAULTS.hidden}`} />
            <Fact label="Epochs" value={`${DEFAULTS.epochs}`} />
            <Fact label="Learning rate" value={`${DEFAULTS.learningRate} (Adam)`} />
            <Fact label="Dropout" value={`${DEFAULTS.dropout}`} />
          </dl>

          <p className="text-xs text-muted-foreground">
            Depth changes the model, so moving the slider re-trains — unlike the
            colour switch beside the graph, which only re-reads the result in
            hand. Nothing starts until you press Train.
          </p>

          <ErrorNote message={session.trainError} />

          {/* Sticky, not pushed down by mt-auto: this input is short, and a
              pinned transport row would open a chasm above it. */}
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
              ? `${summary.nNodes} nodes · ${summary.nEdges / 2} edges`
              : undefined
          }
          description={
            latest
              ? "Each dot is a paper, positioned by a force-directed layout and coloured by topic. Ringed dots are the 140 the model was told about."
              : undefined
          }
          running={training && !latest}
          runningLabel="Training…"
          error={null}
          empty={
            ready
              ? "Press Train to fit a network on the graph. The colours will settle as the labels propagate outward from the 140 labelled papers."
              : "Load the graph to draw it."
          }
        >
          {/* The canvas appears with the first epoch, not with the load. An
              OUTPUT band that fills in before anything has run reads as a
              result, and the pattern's contract is that this slot says what you
              will get until there is something to read
              (model-page-pattern.md §4). */}
          {ready && summary && latest && (
            <div className="flex min-h-0 flex-1 flex-col gap-3">
              <GraphCanvas
                nNodes={summary.nNodes}
                rowPtr={summary.rowPtr}
                colIdx={summary.colIdx}
                x={summary.x}
                y={summary.y}
                labels={summary.labels}
                predictions={session.predictions}
                trainMask={summary.trainMask}
                colorBy={colorBy}
              />

              <ColorLegend colorBy={colorBy} onChange={setColorBy} />

              {latest && (
                <>
                  <Scoreboard
                    testAcc={latest.testAcc}
                    valAcc={latest.valAcc}
                    smoothness={latest.smoothness}
                    elapsedMs={session.elapsedMs}
                    backend={session.backend}
                  />
                  {latest.deadFraction > 0.5 && <CollapseNote />}
                  <Charts
                    metrics={metrics}
                    depthHistory={session.depthHistory}
                    arch={arch}
                  />
                </>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

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
                  selected ? "text-primary-foreground/70" : "text-muted-foreground",
                )}
              >
                {meta.aggregation}
              </span>
            </span>
            <span
              className={cn(
                "w-full text-xs leading-snug font-normal",
                selected ? "text-primary-foreground/75" : "text-muted-foreground",
              )}
            >
              {meta.note}
            </span>
          </Button>
        );
      })}
    </div>
  );
}

function DataStatus({
  session,
}: {
  session: ReturnType<typeof useGraphTraining>;
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
        Decoding the graph and laying it out…
      </p>
    );
  }

  if (status === "ready" && summary) {
    return (
      <p
        data-testid="model-ready"
        className="text-xs text-muted-foreground"
      >
        {summary.nNodes} nodes · {summary.nEdges / 2} edges ·{" "}
        {summary.nFeat} features · laid out in {Math.round(summary.layoutMs)} ms ·
        training runs on{" "}
        <span className="font-medium">
          {summary.backend === "webgpu" ? "the GPU" : "the CPU"}
        </span>
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      <Button size="sm" onClick={session.load}>
        Load graph
      </Button>
      <p className="text-xs text-muted-foreground">
        161 KB, bundled with the app — no download. The wait is the force-directed
        layout, about a second.
      </p>
    </div>
  );
}

function DepthControl({
  layers,
  onChange,
  disabled,
}: {
  layers: number;
  onChange: (layers: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor="graph-depth"
        className="flex items-baseline justify-between text-sm font-medium"
      >
        Depth
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {layers} layers
        </span>
      </label>
      <input
        id="graph-depth"
        type="range"
        min={MIN_LAYERS}
        max={MAX_LAYERS}
        step={1}
        value={layers}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
      <p className="text-xs text-muted-foreground">
        Every deployed GNN is shallow, and this is why. Each layer averages a node
        with its neighbours, so stacking them drives every representation toward
        the same vector — the colours wash out and accuracy falls. Train at 2, then
        at 8.
      </p>
    </div>
  );
}

function ColorLegend({
  colorBy,
  onChange,
}: {
  colorBy: ColorBy;
  onChange: (value: ColorBy) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {CORA_CLASSES.map((label, i) => (
          <span
            key={label}
            className="flex items-center gap-1 text-[0.7rem] text-muted-foreground"
          >
            <span
              aria-hidden
              className="size-2 rounded-full"
              style={{ background: CLASS_COLORS[i] }}
            />
            {label}
          </span>
        ))}
      </div>
      <div className="ml-auto flex gap-1">
        {(["predicted", "true"] as const).map((mode) => (
          <Button
            key={mode}
            size="sm"
            variant={colorBy === mode ? "default" : "outline"}
            aria-pressed={colorBy === mode}
            onClick={() => onChange(mode)}
            className="h-7 px-2 text-xs"
          >
            {mode === "predicted" ? "Predicted" : "True"}
          </Button>
        ))}
      </div>
    </div>
  );
}

function Scoreboard({
  testAcc,
  valAcc,
  smoothness,
  elapsedMs,
  backend,
}: {
  testAcc: number;
  valAcc: number;
  smoothness: number;
  elapsedMs: number | null;
  backend: string | null;
}) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
      <Metric label="Test accuracy" value={`${(testAcc * 100).toFixed(1)}%`} />
      <Metric label="Validation" value={`${(valAcc * 100).toFixed(1)}%`} />
      <Metric
        label="Neighbour similarity"
        value={smoothness.toFixed(3)}
        hint="Mean cosine between adjacent nodes' representations. It climbs toward 1 as depth grows — that is oversmoothing."
      />
      <Metric
        label="Trained in"
        value={
          elapsedMs == null
            ? "—"
            : `${(elapsedMs / 1000).toFixed(1)}s${backend ? ` · ${backend}` : ""}`
        }
      />
    </dl>
  );
}

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div title={hint}>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono text-sm tabular-nums">{value}</dd>
    </div>
  );
}

function CollapseNote() {
  return (
    <p
      data-testid="collapse-note"
      className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400"
    >
      More than half the representations are exactly zero. This is{" "}
      <strong>not</strong> oversmoothing — it is GIN&apos;s unnormalised sum
      overflowing: it multiplies activations by roughly the mean degree at every
      layer, and past four or five layers ReLU zeroes what is left. The
      similarity number is undefined here, because there are no directions left
      to compare.
    </p>
  );
}

function Charts({
  metrics,
  depthHistory,
  arch,
}: {
  metrics: { epoch: number; loss: number; testAcc: number }[];
  depthHistory: { arch: GnnArch; layers: number; testAcc: number; smoothness: number }[];
  arch: GnnArch;
}) {
  const forArch = depthHistory
    .filter((p) => p.arch === arch)
    .sort((a, b) => a.layers - b.layers);

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <ChartPanel title="Training" description="Loss and test accuracy per epoch">
        <EpochChart metrics={metrics} />
      </ChartPanel>
      <ChartPanel
        title="Depth"
        description={
          forArch.length < 2
            ? "Train at another depth to fill this in"
            : `Accuracy and neighbour similarity against depth, for ${GNN_ARCHITECTURES[arch].label}`
        }
      >
        {forArch.length >= 2 ? (
          <DepthChart points={forArch} />
        ) : (
          <p className="flex h-full items-center justify-center text-center text-xs text-muted-foreground">
            One point so far. Move the depth slider and train again — the
            oversmoothing curve is the two numbers diverging.
          </p>
        )}
      </ChartPanel>
    </div>
  );
}

function ChartPanel({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="text-sm font-medium">{title}</p>
      <p className="mb-1 text-xs text-muted-foreground">{description}</p>
      <div className="h-48 w-full">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              Loading chart…
            </div>
          }
        >
          {children}
        </Suspense>
      </div>
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

function EpochChart({
  metrics,
}: {
  metrics: { epoch: number; loss: number; testAcc: number }[];
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
          name: "acc",
          min: 0,
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
          name: "test acc",
          type: "line",
          yAxisIndex: 1,
          showSymbol: false,
          data: metrics.map((m) => [m.epoch + 1, m.testAcc]),
        },
      ],
    }),
    [metrics, colors],
  );
  return <EChart option={option} />;
}

function DepthChart({
  points,
}: {
  points: { layers: number; testAcc: number; smoothness: number }[];
}) {
  const colors = useChartTheme();
  const option = useMemo<EChartsOption>(
    () => ({
      color: colors.series,
      grid: { left: 44, right: 16, top: 24, bottom: 32 },
      legend: { top: 0, textStyle: { color: colors.axis } },
      xAxis: {
        type: "category",
        name: "layers",
        data: points.map((p) => p.layers),
        axisLabel: { color: colors.axis },
        nameTextStyle: { color: colors.axis },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 1,
        axisLabel: { color: colors.axis },
      },
      tooltip: { trigger: "axis" },
      animation: false,
      series: [
        {
          name: "test acc",
          type: "line",
          data: points.map((p) => p.testAcc),
        },
        {
          name: "neighbour similarity",
          type: "line",
          data: points.map((p) => p.smoothness),
        },
      ],
    }),
    [points, colors],
  );
  return <EChart option={option} />;
}
