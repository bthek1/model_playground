import { createFileRoute } from "@tanstack/react-router";
import type { EChartsOption } from "echarts";
import {
  BarChart3,
  Grid3x3,
  Loader2,
  Play,
  Settings2,
  SlidersHorizontal,
  Square,
} from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import { DatasetDialog } from "@/components/training/DatasetDialog";
import { HyperparamsDialog } from "@/components/training/HyperparamsDialog";
import { ModelWeights } from "@/components/training/ModelWeights";
import { NetworkBackground } from "@/components/training/NetworkBackground";
import { pct, Stat } from "@/components/training/controls";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  type TrainingSettings,
  useLinearTraining,
} from "@/hooks/useLinearTraining";
import { useWebGPU } from "@/hooks/useWebGPU";

const EChart = lazy(() => import("@/components/charts/EChart"));

export const Route = createFileRoute("/training")({
  component: TrainingPage,
});

const DEFAULTS: TrainingSettings = {
  learningRate: 0.5,
  batchSize: 64,
  epochs: 10,
  trainSize: 8000,
  testSize: 2000,
};

const EMPTY_WEIGHTS = new Float32Array(784 * 10);

function TrainingPage() {
  const { supported, loading: gpuLoading } = useWebGPU();
  const {
    datasetStatus,
    datasetProgress,
    datasetError,
    poolCount,
    loadData,
    training,
    metrics,
    result,
    snapshot,
    trainError,
    start,
    stop,
  } = useLinearTraining();

  const [settings, setSettings] = useState<TrainingSettings>(DEFAULTS);
  const set = <K extends keyof TrainingSettings>(
    key: K,
    value: TrainingSettings[K],
  ) => setSettings((s) => ({ ...s, [key]: value }));

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : undefined;
  const lastEval = useMemo(
    () => [...metrics].reverse().find((m) => m.testAcc !== null),
    [metrics],
  );

  const canStart = supported && datasetStatus === "ready" && !training;
  const status = training
    ? "training"
    : result
      ? "done"
      : datasetStatus === "ready"
        ? "ready"
        : "idle";

  return (
    <div className="relative -m-6 h-[calc(100%+3rem)] w-[calc(100%+3rem)] overflow-hidden">
      {/* Background: live model-structure visualization */}
      <NetworkBackground
        weights={snapshot?.weights ?? EMPTY_WEIGHTS}
        bias={snapshot?.bias}
        active={training}
        className="absolute inset-0 h-full w-full text-foreground"
      />

      {/* Foreground HUD ------------------------------------------------- */}
      {/* Top-left: title, status, live stats */}
      <div className="pointer-events-none absolute top-4 left-4 max-w-[min(90vw,26rem)] space-y-3">
        <div className="pointer-events-auto rounded-xl border bg-card/70 p-4 backdrop-blur-md">
          <div className="mb-1 flex items-center gap-2">
            <h1 className="text-lg font-semibold">Linear Model Training</h1>
            <StatusPill status={status} />
          </div>
          <p className="text-xs text-muted-foreground">
            A softmax classifier (784&nbsp;→&nbsp;10) trained on MNIST in your
            browser. Edges are pooled weights — red pushes for a digit, blue
            against — sharpening live as training runs.
          </p>
        </div>

        {(latest || result) && (
          <div className="pointer-events-auto grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat
              label="Epoch"
              value={latest ? `${latest.epoch + 1}` : "—"}
              sub={
                latest ? `step ${latest.step}/${latest.totalSteps}` : undefined
              }
            />
            <Stat
              label="Batch loss"
              value={latest ? latest.loss.toFixed(4) : "—"}
            />
            <Stat
              label="Train acc"
              value={lastEval?.trainAcc != null ? pct(lastEval.trainAcc) : "—"}
            />
            <Stat
              label="Test acc"
              value={
                result
                  ? pct(result.testAcc)
                  : lastEval?.testAcc != null
                    ? pct(lastEval.testAcc)
                    : "—"
              }
            />
          </div>
        )}
      </div>

      {/* Top-right: settings popups */}
      <div className="pointer-events-auto absolute top-4 right-4 flex gap-2">
        <DatasetDialog
          trigger={
            <Button variant="outline" size="sm" className="bg-card/70 backdrop-blur-md">
              <Settings2 className="size-4" /> Dataset
            </Button>
          }
          settings={settings}
          set={set}
          disabled={training}
          datasetStatus={datasetStatus}
          datasetProgress={datasetProgress}
          datasetError={datasetError}
          poolCount={poolCount}
          loadData={loadData}
        />
        <HyperparamsDialog
          trigger={
            <Button variant="outline" size="sm" className="bg-card/70 backdrop-blur-md">
              <SlidersHorizontal className="size-4" /> Tune
            </Button>
          }
          settings={settings}
          set={set}
          disabled={training}
        />
      </div>

      {/* WebGPU unavailable notice */}
      {!gpuLoading && !supported && (
        <div className="pointer-events-auto absolute top-20 left-1/2 -translate-x-1/2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-center text-sm backdrop-blur-md">
          WebGPU isn't available here, so training can't run. See the Playground
          page for details on enabling it.
        </div>
      )}

      {/* Bottom-center: transport controls */}
      <div className="pointer-events-auto absolute bottom-4 left-1/2 flex -translate-x-1/2 flex-wrap items-center justify-center gap-2 rounded-full border bg-card/70 p-2 pl-4 backdrop-blur-md">
        {datasetStatus !== "ready" && (
          <span className="text-xs text-muted-foreground">
            Open <span className="font-medium">Dataset</span> to load MNIST →
          </span>
        )}
        <Button onClick={() => start(settings)} disabled={!canStart} size="sm">
          {training ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Training…
            </>
          ) : (
            <>
              <Play className="size-4" /> Start
            </>
          )}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={stop}
          disabled={!training}
        >
          <Square className="size-4" /> Stop
        </Button>
        {trainError && (
          <span className="max-w-48 truncate text-xs text-destructive">
            {trainError}
          </span>
        )}
      </div>

      {/* Bottom-right: collapsible panels for templates + charts */}
      <div className="pointer-events-auto absolute right-4 bottom-4 flex gap-2">
        {snapshot && (
          <Popover>
            <PopoverTrigger
              render={
                <Button
                  variant="outline"
                  size="sm"
                  className="bg-card/70 backdrop-blur-md"
                />
              }
            >
              <Grid3x3 className="size-4" /> Weights
            </PopoverTrigger>
            <PopoverContent
              side="top"
              align="end"
              className="w-[min(92vw,34rem)]"
            >
              <p className="mb-2 text-xs text-muted-foreground">
                Each tile is one digit detector: its 784 weights reshaped to
                28×28. Red pixels vote for the digit, blue against.
              </p>
              <ModelWeights
                weights={snapshot.weights}
                bias={snapshot.bias}
                epoch={snapshot.epoch}
              />
            </PopoverContent>
          </Popover>
        )}
        {metrics.length > 0 && (
          <Popover>
            <PopoverTrigger
              render={
                <Button
                  variant="outline"
                  size="sm"
                  className="bg-card/70 backdrop-blur-md"
                />
              }
            >
              <BarChart3 className="size-4" /> Charts
            </PopoverTrigger>
            <PopoverContent
              side="top"
              align="end"
              className="w-[min(92vw,44rem)]"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <ChartPanel title="Loss" description="Cross-entropy per batch">
                  <LossChart metrics={metrics} />
                </ChartPanel>
                <ChartPanel
                  title="Accuracy"
                  description="Per-epoch train vs test"
                >
                  <AccuracyChart metrics={metrics} />
                </ChartPanel>
              </div>
            </PopoverContent>
          </Popover>
        )}
      </div>
    </div>
  );
}

function StatusPill({
  status,
}: {
  status: "idle" | "ready" | "training" | "done";
}) {
  const map = {
    idle: { label: "idle", cls: "bg-muted text-muted-foreground" },
    ready: { label: "ready", cls: "bg-primary/15 text-primary" },
    training: { label: "training", cls: "bg-amber-500/15 text-amber-600 dark:text-amber-400" },
    done: { label: "done", cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400" },
  }[status];
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[10px] font-medium tracking-wide uppercase ${map.cls}`}
    >
      {map.label}
    </span>
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
      <div className="h-56 w-full">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
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

function LossChart({ metrics }: { metrics: { step: number; loss: number }[] }) {
  const option = useMemo<EChartsOption>(
    () => ({
      grid: { left: 48, right: 16, top: 16, bottom: 32 },
      xAxis: { type: "value", name: "step", min: 1 },
      yAxis: { type: "value", name: "loss", min: 0 },
      tooltip: { trigger: "axis" },
      animation: false,
      series: [
        {
          type: "line",
          showSymbol: false,
          data: metrics.map((m) => [m.step, m.loss]),
        },
      ],
    }),
    [metrics],
  );
  return <EChart option={option} />;
}

function AccuracyChart({
  metrics,
}: {
  metrics: { step: number; trainAcc: number | null; testAcc: number | null }[];
}) {
  const option = useMemo<EChartsOption>(() => {
    const evals = metrics.filter((m) => m.testAcc !== null);
    return {
      grid: { left: 48, right: 16, top: 24, bottom: 32 },
      legend: { top: 0 },
      xAxis: { type: "value", name: "step", min: 1 },
      yAxis: { type: "value", name: "acc", min: 0, max: 1 },
      tooltip: { trigger: "axis" },
      animation: false,
      series: [
        {
          name: "train",
          type: "line",
          data: evals.map((m) => [m.step, m.trainAcc]),
        },
        {
          name: "test",
          type: "line",
          data: evals.map((m) => [m.step, m.testAcc]),
        },
      ],
    };
  }, [metrics]);
  return <EChart option={option} />;
}
