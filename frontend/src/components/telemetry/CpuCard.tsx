// The CPU card — responsiveness and activity, not utilisation.
//
// The two figures answer different questions and both matter during a run:
// "busy" is whether the app has an inference in flight (reported by the worker
// hooks through `telemetry/activity.ts`), while "lag" is whether the main thread
// is still keeping up. A worker saturating every core shows a high busy
// fraction and near-zero lag, which is the architecture working as intended.

import { Activity } from "lucide-react";

import { formatMs } from "@/lib/format";
import { maxOf } from "@/telemetry/series";
import type { CpuSample, Metric } from "@/telemetry/types";

import { MetricCard, Meter, Readout, Row } from "./MetricCard";
import { Sparkline } from "./Sparkline";

/**
 * Full scale for the lag chart: the threshold at which the browser itself calls
 * a block a long task. An auto-ranged lag chart is always dramatic — 0.13 ms of
 * frame overrun drawn against 0.13 ms fills the box — while against 50 ms an
 * idle machine correctly reads as flat and a real stall stands out. The scale
 * expands if something exceeds it, so a spike is never clipped.
 */
const LAG_FULL_SCALE_MS = 50;

export interface CpuCardProps {
  cpu: CpuSample;
  cores: Metric<number>;
  /** Why long-task counts are missing, or null when they aren't. */
  longTaskReason: string | null;
  lagHistory: number[];
  busyHistory: number[];
}

export function CpuCard({
  cpu,
  cores,
  longTaskReason,
  lagHistory,
  busyHistory,
}: CpuCardProps) {
  return (
    <MetricCard
      title="CPU"
      icon={Activity}
      testId="telemetry-cpu"
      note="Inference activity and main-thread responsiveness. No browser reports host CPU utilisation, so this is what the app is doing rather than what the machine is doing."
    >
      <Readout value={`${Math.round(cpu.busyFraction * 100)}%`} of="inference busy" />
      <Meter fraction={cpu.busyFraction} label="Share of the interval with an inference in flight" />
      {/* Absolute: an idle second belongs on the floor of a 0–1 axis. */}
      <Sparkline
        values={busyHistory}
        min={0}
        max={1}
        label="Inference activity over the last two minutes"
      />
      <Row label="Main-thread lag" value={formatMs(cpu.lagMs)} />
      <Sparkline
        values={lagHistory}
        min={0}
        max={Math.max(LAG_FULL_SCALE_MS, maxOf(lagHistory))}
        label="Worst frame overrun per second, over the last two minutes"
        token="--chart-4"
      />
      <Row
        label="Long tasks"
        value={
          cpu.longTasks == null
            ? "unavailable"
            : `${cpu.longTasks}${cpu.longTaskMs ? ` (${formatMs(cpu.longTaskMs)})` : ""}`
        }
        title={longTaskReason ?? undefined}
      />
      {cores.status === "ok" && (
        <Row
          label="Logical cores"
          value={cores.value}
          title="Capacity. Nothing reports how many are free."
        />
      )}
    </MetricCard>
  );
}
