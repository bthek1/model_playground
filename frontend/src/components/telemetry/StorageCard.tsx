// Storage, and the download in flight.
//
// This is the panel's most solid number: `navigator.storage.estimate()` really
// does measure the origin's footprint, and for this app that footprint is almost
// entirely cached model weights. It is the after-the-fact counterpart to the
// size guardrail in `model/size.ts` — what the downloads actually cost.
//
// The download block appears only while something is loading. Its rate comes
// from differencing the byte aggregate in `model/progress.ts`, because the Hub's
// cross-origin responses report a transfer size of zero to resource timing.

import { HardDrive } from "lucide-react";

import { formatBytes } from "@/lib/format";
import type { DownloadSample, Metric, StorageSample } from "@/telemetry/types";

import { MetricCard, Meter, Readout, Row } from "./MetricCard";
import { Sparkline } from "./Sparkline";

export interface StorageCardProps {
  storage: Metric<StorageSample> | null;
  download: DownloadSample | null;
  /** Bytes per second, oldest → newest. */
  history: number[];
}

export function StorageCard({ storage, download, history }: StorageCardProps) {
  const sample = storage?.status === "ok" ? storage.value : null;

  return (
    <MetricCard
      title="Storage"
      icon={HardDrive}
      testId="telemetry-storage"
      unavailable={storage?.status === "unavailable" ? storage.reason : null}
      note="Everything this origin has stored, which here is almost entirely cached model weights. The quota is what the browser will allow, not free disk."
    >
      <Readout
        value={sample ? formatBytes(sample.usageBytes) : "—"}
        of={sample ? formatBytes(sample.quotaBytes) : undefined}
      />
      {sample && (
        <>
          <Meter
            fraction={sample.quotaBytes > 0 ? sample.usageBytes / sample.quotaBytes : 0}
            label="Storage used against the quota"
            token="--chart-5"
          />
          {sample.cachedModels != null && (
            <Row label="Models cached" value={sample.cachedModels} />
          )}
        </>
      )}
      {download && (
        <div
          data-testid="telemetry-download"
          className="mt-3 space-y-2 border-t pt-2"
        >
          <Row
            label="Downloading"
            value={`${formatBytes(download.loadedBytes)} / ${formatBytes(download.totalBytes)}`}
          />
          <Meter
            fraction={
              download.totalBytes > 0 ? download.loadedBytes / download.totalBytes : 0
            }
            label="Download progress"
            token="--chart-2"
          />
          <Row
            label="Rate"
            value={
              download.bytesPerSecond == null
                ? "measuring…"
                : `${formatBytes(download.bytesPerSecond)}/s`
            }
          />
          <Sparkline
            values={history}
            label="Download rate over the last two minutes"
            token="--chart-2"
          />
        </div>
      )}
    </MetricCard>
  );
}
