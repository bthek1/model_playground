// The memory card, and the one most likely to be misread.
//
// `performance.memory` is the JS heap of the realm that asks — this page. The
// weights live in workers, which have their own heaps and are not visible from
// here, so a 300 MB model load barely moves this number. The note says so,
// because the alternative is a user concluding the load was free.

import { MemoryStick } from "lucide-react";

import { formatBytes } from "@/lib/format";
import type { MemorySample, Metric } from "@/telemetry/types";

import { MetricCard, Meter, Readout, Row } from "./MetricCard";
import { Sparkline } from "./Sparkline";

export interface MemoryCardProps {
  memory: Metric<MemorySample>;
  deviceMemoryGiB: Metric<number>;
  /** Used heap bytes, oldest → newest. */
  history: number[];
}

export function MemoryCard({
  memory,
  deviceMemoryGiB,
  history,
}: MemoryCardProps) {
  const sample = memory.status === "ok" ? memory.value : null;

  return (
    <MetricCard
      title="Memory"
      icon={MemoryStick}
      testId="telemetry-memory"
      unavailable={memory.status === "unavailable" ? memory.reason : null}
      note="JS heap of this page only. Model weights are held in Web Workers, whose heaps no API exposes — so a large load moves this very little."
    >
      {sample && (
        <>
          <Readout
            value={formatBytes(sample.usedBytes)}
            of={formatBytes(sample.limitBytes)}
          />
          <Meter
            fraction={sample.limitBytes > 0 ? sample.usedBytes / sample.limitBytes : 0}
            label="Page heap against its ceiling"
            token="--chart-3"
          />
          {/* Trend, not ratio: the meter above already says where the heap
              sits against its ceiling, and against a 4 GiB ceiling a real
              50 MB climb would be a flat line. */}
          <Sparkline
            values={history}
            label="Page heap over the last two minutes"
            token="--chart-3"
          />
          <Row label="Heap allocated" value={formatBytes(sample.totalBytes)} />
          {deviceMemoryGiB.status === "ok" && (
            <Row
              label="Device class"
              value={`~${deviceMemoryGiB.value} GiB`}
              title="Coarse and capped by the browser to resist fingerprinting — capacity, not free RAM."
            />
          )}
        </>
      )}
    </MetricCard>
  );
}
