import { createFileRoute } from "@tanstack/react-router";
import type { EChartsOption } from "echarts";
import { Download, Loader2, Play, Square } from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import { ModelWeights } from "@/components/training/ModelWeights";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
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

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Linear Model Training</h1>
        <p className="text-sm text-muted-foreground">
          Train a linear softmax classifier (784 → 10) on MNIST, entirely in your
          browser. The two matmuls per step run on your GPU via raw WebGPU in a
          Web Worker; loss and accuracy stream in live below.
        </p>
      </div>

      {!gpuLoading && !supported && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm">
          WebGPU isn't available here, so training can't run. See the Playground
          page for details on enabling it.
        </div>
      )}

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>1 · Dataset</CardTitle>
            <CardDescription>
              MNIST is fetched from a CDN and decoded in your browser.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <Button
              variant="outline"
              disabled={datasetStatus === "loading"}
              onClick={() => loadData(settings.trainSize + settings.testSize)}
            >
              {datasetStatus === "loading" ? (
                <>
                  <Loader2 className="size-4 animate-spin" /> Loading…
                </>
              ) : (
                <>
                  <Download className="size-4" /> Load MNIST
                </>
              )}
            </Button>

            {datasetStatus === "loading" && (
              <div className="h-2 w-full overflow-hidden rounded bg-muted">
                <div
                  className="h-full bg-primary transition-[width]"
                  style={{ width: `${Math.round(datasetProgress * 100)}%` }}
                />
              </div>
            )}
            {datasetStatus === "ready" && (
              <p className="text-muted-foreground">
                Loaded {poolCount.toLocaleString()} images (28×28, greyscale).
              </p>
            )}
            {datasetError && <p className="text-destructive">{datasetError}</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2 · Hyperparameters</CardTitle>
            <CardDescription>Plain mini-batch SGD.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3">
              <NumberField
                label="Learning rate"
                value={settings.learningRate}
                step={0.05}
                min={0.001}
                disabled={training}
                onChange={(v) => set("learningRate", v)}
              />
              <NumberField
                label="Batch size"
                value={settings.batchSize}
                step={16}
                min={1}
                disabled={training}
                onChange={(v) => set("batchSize", Math.round(v))}
              />
              <NumberField
                label="Epochs"
                value={settings.epochs}
                step={1}
                min={1}
                disabled={training}
                onChange={(v) => set("epochs", Math.round(v))}
              />
              <NumberField
                label="Train images"
                value={settings.trainSize}
                step={1000}
                min={100}
                disabled={training}
                onChange={(v) => set("trainSize", Math.round(v))}
              />
              <NumberField
                label="Test images"
                value={settings.testSize}
                step={500}
                min={100}
                disabled={training}
                onChange={(v) => set("testSize", Math.round(v))}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => start(settings)} disabled={!canStart}>
          {training ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Training…
            </>
          ) : (
            <>
              <Play className="size-4" /> Start training
            </>
          )}
        </Button>
        <Button variant="outline" onClick={stop} disabled={!training}>
          <Square className="size-4" /> Stop
        </Button>
        {trainError && <p className="text-sm text-destructive">{trainError}</p>}
      </div>

      {(latest || result) && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="Epoch"
            value={latest ? `${latest.epoch + 1}` : "—"}
            sub={latest ? `step ${latest.step}/${latest.totalSteps}` : undefined}
          />
          <Stat label="Batch loss" value={latest ? latest.loss.toFixed(4) : "—"} />
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

      {snapshot && (
        <Card>
          <CardHeader>
            <CardTitle>The model</CardTitle>
            <CardDescription>
              Each 28×28 tile is one digit detector: its 784 weights reshaped to
              the input image. Red pixels push toward that digit, blue pixels
              against it. Watch the templates sharpen into digit shapes as
              training runs.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ModelWeights
              weights={snapshot.weights}
              bias={snapshot.bias}
              epoch={snapshot.epoch}
            />
          </CardContent>
        </Card>
      )}

      {metrics.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <ChartCard title="Loss" description="Cross-entropy per batch">
            <LossChart metrics={metrics} />
          </ChartCard>
          <ChartCard title="Accuracy" description="Per-epoch train vs test">
            <AccuracyChart metrics={metrics} />
          </ChartCard>
        </div>
      )}
    </div>
  );
}

function pct(x: number): string {
  return `${(x * 100).toFixed(2)}%`;
}

function NumberField({
  label,
  value,
  onChange,
  step,
  min,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
  min: number;
  disabled?: boolean;
}) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        className="h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
      />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-lg tabular-nums">{value}</p>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

function ChartCard({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
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
      </CardContent>
    </Card>
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
