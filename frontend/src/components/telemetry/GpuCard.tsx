// The GPU card: identity, and the bytes we are holding on the device.
//
// Everything here is load or capability. There is no utilisation figure to show
// — WebGPU deliberately exposes neither VRAM nor busy time — so the card's job
// is to be useful without one and to say plainly that the missing number is
// missing rather than zero.

import { Cpu } from "lucide-react";

import { formatBytes, formatMs, titleCase } from "@/lib/format";
import type { GpuMemorySample, Metric } from "@/telemetry/types";
import type { WebGPUCapabilities } from "@/webgpu/types";

import { MetricCard, Readout, Row } from "./MetricCard";
import { Sparkline } from "./Sparkline";

export interface GpuCardProps {
  identity: Metric<WebGPUCapabilities> | null;
  memory: Metric<GpuMemorySample> | null;
  /** Live bytes, oldest → newest. */
  history: number[];
}

/** "NVIDIA · ampere" from whichever fields the adapter actually filled in. */
function adapterLabel(caps: WebGPUCapabilities): string {
  const info = caps.adapter;
  if (!info) return "Unknown adapter";
  const parts = [info.description, info.vendor, info.architecture]
    .map((part) => part?.trim())
    .filter((part): part is string => Boolean(part) && part !== "unknown");
  return parts.length ? titleCase(parts[0]) : "Unknown adapter";
}

export function GpuCard({ identity, memory, history }: GpuCardProps) {
  const unavailable =
    identity?.status === "unavailable"
      ? identity.reason
      : memory?.status === "unavailable"
        ? memory.reason
        : null;

  const sample = memory?.status === "ok" ? memory.value : null;
  const caps = identity?.status === "ok" ? identity.value : null;

  return (
    <MetricCard
      title="GPU"
      icon={Cpu}
      testId="telemetry-gpu"
      unavailable={unavailable}
      note="Buffers this app allocated on the device, across the page and every WebGPU worker. WebGPU exposes no VRAM total and no utilisation; ONNX Runtime's own GPU memory is invisible to us."
    >
      <Readout value={sample ? formatBytes(sample.liveBytes) : "—"} />
      <Sparkline
        values={history}
        label="Tracked GPU buffer bytes over the last two minutes"
        token="--chart-2"
      />
      {caps && (
        <Row
          label="Adapter"
          value={adapterLabel(caps)}
          title={caps.adapter?.description || undefined}
        />
      )}
      {caps?.isFallbackAdapter && (
        <Row label="Adapter type" value="Software fallback" />
      )}
      {sample && (
        <>
          <Row label="Peak" value={formatBytes(sample.peakBytes)} />
          <Row label="Buffers held" value={sample.buffers} />
          <Row
            label="Last WGSL pass"
            value={
              sample.lastPassMs == null ? "not timed" : formatMs(sample.lastPassMs)
            }
            title={
              sample.lastPassMs == null
                ? "Needs the timestamp-query feature, and only covers this app's own WGSL kernels."
                : undefined
            }
          />
        </>
      )}
    </MetricCard>
  );
}
