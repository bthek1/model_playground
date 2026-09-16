// Graph classification on PROTEINS: one label per molecule rather than per node.
//
// The third and last page of the Graph ML roadmap, and the only one that
// downloads anything — 2.06 MB of JSONL from the Hub, cached in IndexedDB after
// the first visit. Everything else is the machinery /graph already has: the same
// aggregation kernel, the same four architectures, one CSR trained full-batch.
//
// **The majority baseline is rendered beside the accuracy, and that is a
// correctness requirement rather than a courtesy.** PROTEINS is 663 enzymes to
// 450 non-enzymes, so a classifier that ignores its input entirely scores 0.598.
// Published GNNs score 0.73-0.76. An accuracy printed on its own cannot be told
// apart from a model that learned the prior — the reader has no way to know which
// they are looking at. So the null model travels inside the metrics and is shown
// next to every number it qualifies.

import type { EChartsOption } from "echarts";
import { createFileRoute } from "@tanstack/react-router";
import { Boxes, Loader2, Play, Square } from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import { ProteinGallery } from "@/components/graph/ProteinGallery";
import { DeviceStatus } from "@/components/model/DeviceStatus";
import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useGraphClassifier } from "@/hooks/useGraphClassifier";
import { useTheme } from "@/hooks/useTheme";
import { useWebGPU } from "@/hooks/useWebGPU";
import { PROTEINS_BYTES, PROTEINS_CLASSES } from "@/lib/proteins";
import { getCSSVar } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { GNN_ARCHITECTURES, type GnnArch } from "@/webgpu/gnn";
import { READOUTS, type ReadoutMode } from "@/webgpu/graphPool";

const EChart = lazy(() => import("@/components/charts/EChart"));

export const Route = createFileRoute("/graph-classification")({
  component: GraphClassificationPage,
});

const ARCHES: GnnArch[] = ["gcn", "sage", "gin", "gat"];
const READOUT_MODES: ReadoutMode[] = ["mean", "sum"];

/** How many test proteins the gallery draws. */
const GALLERY_SIZE = 48;

const DEFAULTS = {
  layers: 3,
  hidden: 32,
  learningRate: 0.01,
  weightDecay: 0,
  dropout: 0.2,
  epochs: 150,
  seed: 42,
};

function GraphClassificationPage() {
  const { capabilities, loading: probing } = useWebGPU();
  const session = useGraphClassifier();
  const [arch, setArch] = useState<GnnArch>("gin");
  const [readout, setReadout] = useState<ReadoutMode>("mean");
  const [selected, setSelected] = useState<number | null>(null);

  const { summary, status, training, metrics, predicted } = session;
  const ready = status === "ready" && summary != null;
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  const gallery = useMemo(
    () => (summary ? Array.from(summary.testIdx.slice(0, GALLERY_SIZE)) : []),
    [summary],
  );

  const start = () => session.start({ ...DEFAULTS, arch, readout });

  return (
    <ModelPage
      icon={Boxes}
      title="Graph Classification"
      description={
        <>
          One label for a whole graph rather than one per node.{" "}
          <strong>PROTEINS</strong> is 1113 protein structures, each either an
          enzyme or not; the model reads the shape of the molecule and answers
          once. The network is written as WGSL and trained in front of you — there
          is no checkpoint, only a 2 MB dataset.
        </>
      }
      labels={{
        select: "Architecture",
        load: "Dataset & device",
        run: "Train",
        output: "Proteins",
      }}
      select={
        <div className="space-y-3">
          <ArchPicker value={arch} onChange={setArch} disabled={training} />
          <ReadoutPicker
            value={readout}
            onChange={setReadout}
            disabled={training}
          />
        </div>
      }
      load={
        <div className="space-y-2">
          <DeviceStatus capabilities={capabilities} loading={probing} />
          <DatasetStatus session={session} />
        </div>
      }
      run={
        <div className="flex min-h-0 flex-1 flex-col gap-4">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <Fact label="Layers" value={`${DEFAULTS.layers}`} />
            <Fact label="Hidden width" value={`${DEFAULTS.hidden}`} />
            <Fact label="Epochs" value={`${DEFAULTS.epochs}`} />
            <Fact label="Dropout" value={`${DEFAULTS.dropout}`} />
          </dl>

          <p className="text-xs text-muted-foreground">
            All 1113 graphs are trained at once, as a single block-diagonal
            graph — that is what batching means here. The readout pools the
            per-node logits into one row per protein, which is a{" "}
            <strong>single linear readout</strong>: GIN&rsquo;s paper uses an MLP
            over summed per-layer representations, and this does not.
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
          title="PROTEINS"
          meta={summary ? `${summary.nTest} held-out structures` : undefined}
          description={
            latest
              ? "Each tile is one held-out protein, outlined green where the model got it right and red where it did not. Click one to see its structure."
              : undefined
          }
          running={training && !latest}
          runningLabel="Training…"
          error={null}
          empty={
            ready
              ? "Press Train to fit a network on 1113 molecules. The test structures appear as tiles, outlined by whether the answer was right."
              : "Load the dataset to begin."
          }
        >
          {ready && summary && latest && predicted && (
            <div className="flex min-h-0 flex-1 flex-col gap-3">
              <Scoreboard
                testAcc={latest.testAcc}
                valAcc={latest.valAcc}
                baselineAcc={latest.baselineAcc}
                elapsedMs={session.elapsedMs}
                backend={summary.backend}
              />

              <BaselineNote
                testAcc={latest.testAcc}
                baselineAcc={latest.baselineAcc}
                majority={PROTEINS_CLASSES[majorityOf(summary.labels)]}
              />

              <ProteinGallery
                graphs={gallery}
                labels={summary.labels}
                predicted={predicted}
                layouts={session.layouts}
                onNeedLayout={session.requestLayout}
                selected={selected}
                onSelect={setSelected}
                classNames={PROTEINS_CLASSES}
              />

              {selected != null && (
                <SelectedNote
                  graph={selected}
                  summary={summary}
                  predicted={predicted}
                />
              )}

              <EpochChart metrics={metrics} />
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

/** The commonest label, for the sentence that names the null model. */
function majorityOf(labels: Uint8Array): number {
  const counts = new Map<number, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  let best = 0;
  let bestCount = -1;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      best = label;
      bestCount = count;
    }
  }
  return best;
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

function ReadoutPicker({
  value,
  onChange,
  disabled,
}: {
  value: ReadoutMode;
  onChange: (mode: ReadoutMode) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <span className="text-sm font-medium">Readout</span>
      <div className="flex gap-1">
        {READOUT_MODES.map((mode) => (
          <Button
            key={mode}
            size="sm"
            variant={mode === value ? "default" : "outline"}
            aria-pressed={mode === value}
            disabled={disabled}
            onClick={() => onChange(mode)}
          >
            {READOUTS[mode].label}
          </Button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">{READOUTS[value].note}</p>
    </div>
  );
}

function DatasetStatus({
  session,
}: {
  session: ReturnType<typeof useGraphClassifier>;
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
        Downloading and joining 1113 graphs…
      </p>
    );
  }

  if (status === "ready" && summary) {
    return (
      <p data-testid="model-ready" className="text-xs text-muted-foreground">
        {summary.nGraphs} graphs · {summary.nNodes} nodes ·{" "}
        {summary.nEdges / 2} bonds · {summary.nTrain}/{summary.nVal}/
        {summary.nTest} train/val/test ·{" "}
        {summary.fromCache ? "read from cache" : "downloaded"} in{" "}
        {Math.round(summary.loadMs)} ms · training runs on{" "}
        <span className="font-medium">
          {summary.backend === "webgpu" ? "the GPU" : "the CPU"}
        </span>
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      <Button size="sm" onClick={session.load}>
        Load dataset
      </Button>
      <p className="text-xs text-muted-foreground">
        {(PROTEINS_BYTES / 1024 / 1024).toFixed(1)} MB from the Hugging Face Hub —
        the only page here that downloads anything. Cached in your browser
        afterwards, so this is a one-time wait.
      </p>
    </div>
  );
}

/**
 * The sentence that makes the accuracy readable.
 *
 * Without it "72 %" is unreadable rather than merely unadorned: the reader cannot
 * separate real signal from a model that learned the class prior. With it the
 * page states the margin, and says so plainly when there is not one.
 */
function BaselineNote({
  testAcc,
  baselineAcc,
  majority,
}: {
  testAcc: number;
  baselineAcc: number;
  majority: string;
}) {
  const margin = testAcc - baselineAcc;
  const beat = margin > 0.01;
  return (
    <p
      data-testid="baseline-note"
      className={cn(
        "rounded-md border px-3 py-2 text-xs",
        beat ? "text-muted-foreground" : "border-amber-500/50 text-foreground",
      )}
    >
      {beat ? (
        <>
          <strong>{(margin * 100).toFixed(1)} points above the baseline.</strong>{" "}
          Always answering &ldquo;{majority}&rdquo; would score{" "}
          {(baselineAcc * 100).toFixed(1)}% on this split — that is what a model
          which ignores the molecule entirely gets, and the number worth reading
          the accuracy against.
        </>
      ) : (
        <>
          <strong>This has not beaten the baseline.</strong> Always answering
          &ldquo;{majority}&rdquo; scores {(baselineAcc * 100).toFixed(1)}% on this
          split, so the model has learned the class prior and little else. Try
          more epochs, a different readout, or GIN.
        </>
      )}
    </p>
  );
}

function SelectedNote({
  graph,
  summary,
  predicted,
}: {
  graph: number;
  summary: { labels: Uint8Array; graphPtr: Uint32Array };
  predicted: Uint8Array;
}) {
  const nNodes = summary.graphPtr[graph + 1] - summary.graphPtr[graph];
  const right = predicted[graph] === summary.labels[graph];
  return (
    <p data-testid="selected-note" className="text-xs text-muted-foreground">
      <span className="font-mono">#{graph}</span> · {nNodes} residues · really{" "}
      <strong>{PROTEINS_CLASSES[summary.labels[graph]]}</strong>, predicted{" "}
      <strong>{PROTEINS_CLASSES[predicted[graph]]}</strong>{" "}
      {right ? "✓" : "✗"}
    </p>
  );
}

function Scoreboard({
  testAcc,
  valAcc,
  baselineAcc,
  elapsedMs,
  backend,
}: {
  testAcc: number;
  valAcc: number;
  baselineAcc: number;
  elapsedMs: number | null;
  backend: string;
}) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
      <Score label="Test accuracy" value={pct(testAcc)} />
      <Score label="Validation" value={pct(valAcc)} />
      <Score label="Majority baseline" value={pct(baselineAcc)} />
      <Score
        label="Trained in"
        value={
          elapsedMs == null
            ? "—"
            : `${(elapsedMs / 1000).toFixed(1)}s (${backend})`
        }
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

// ECharts cannot read CSS variables, so the tokens are resolved here and
// re-resolved when the theme flips. `theme` is a dependency only to force that.
function useChartTheme() {
  const { theme } = useTheme();
  return useMemo(
    () => ({
      axis: getCSSVar("mutedForeground"),
      series: [getCSSVar("chart2"), getCSSVar("chart4"), getCSSVar("chart5")],
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [theme],
  );
}

function EpochChart({
  metrics,
}: {
  metrics: {
    epoch: number;
    loss: number;
    testAcc: number;
    baselineAcc: number;
  }[];
}) {
  const colors = useChartTheme();
  const baseline = metrics.length > 0 ? metrics[0].baselineAcc : 0;
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
          // The null model as a line across the chart: a curve that never rises
          // above it has not learned anything about the molecules.
          markLine: {
            silent: true,
            symbol: "none",
            label: { formatter: "baseline", color: colors.axis },
            data: [{ yAxis: baseline }],
          },
        },
      ],
    }),
    [metrics, colors, baseline],
  );

  return (
    <div>
      <p className="text-xs font-medium">Loss and test accuracy per epoch</p>
      {/* EChart fills 100% of its parent, so the parent owes it a height. */}
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
