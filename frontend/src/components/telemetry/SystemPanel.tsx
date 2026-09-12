// The panel's body: one sampling loop, four cards.
//
// `active` is threaded down rather than read from the store here so the panel
// body can be rendered (and tested) without one, and so the loop is provably
// off when the shell says it should be. `data-tick` exposes the sample count to
// the E2E spec, which needs to prove that closing the panel really does stop
// the sampling rather than merely hiding the numbers.

import { useTelemetry } from "@/telemetry/useTelemetry";

import { CpuCard } from "./CpuCard";
import { GpuCard } from "./GpuCard";
import { MemoryCard } from "./MemoryCard";
import { StorageCard } from "./StorageCard";

export function SystemPanel({ active }: { active: boolean }) {
  const view = useTelemetry(active);

  return (
    <div
      data-testid="system-panel-body"
      data-tick={view.tick}
      className="flex flex-col gap-3 p-3"
    >
      <p className="text-[11px] leading-snug text-muted-foreground">
        What this machine is doing, as far as a web page can see it — load and
        capacity, never utilisation. A browser reports no host CPU, GPU or RAM
        usage; each card says what its number actually is. Sampled every{" "}
        {Math.round(view.intervalMs / 1000)} s while this panel is open.
      </p>

      <GpuCard
        identity={view.identity}
        memory={view.gpu}
        history={view.gpuHistory}
      />
      <MemoryCard
        memory={view.memory}
        deviceMemoryGiB={view.capacity.deviceMemoryGiB}
        history={view.memoryHistory}
      />
      <CpuCard
        cpu={view.cpu}
        cores={view.capacity.cores}
        longTaskReason={view.longTaskReason}
        lagHistory={view.lagHistory}
        busyHistory={view.busyHistory}
      />
      <StorageCard
        storage={view.storage}
        download={view.download}
        history={view.downloadHistory}
      />
    </div>
  );
}
