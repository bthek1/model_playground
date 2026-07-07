// Dataset configuration as a popup: train/test sizes plus the MNIST loader with
// its progress bar. Pulled out of the training route so the network-viz stage
// stays uncluttered — this is a modal the user opens on demand.

import { Download, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type {
  DatasetStatus,
  TrainingSettings,
} from "@/hooks/useLinearTraining";

import { NumberField } from "./controls";

export function DatasetDialog({
  trigger,
  settings,
  set,
  disabled,
  datasetStatus,
  datasetProgress,
  datasetError,
  poolCount,
  loadData,
}: {
  trigger: ReactNode;
  settings: TrainingSettings;
  set: <K extends keyof TrainingSettings>(k: K, v: TrainingSettings[K]) => void;
  disabled: boolean;
  datasetStatus: DatasetStatus;
  datasetProgress: number;
  datasetError: string | null;
  poolCount: number;
  loadData: (count: number) => void;
}) {
  return (
    <Dialog>
      <DialogTrigger render={trigger as React.ReactElement} />
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Dataset</DialogTitle>
          <DialogDescription>
            MNIST is fetched from a CDN and decoded in your browser.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Train images"
            value={settings.trainSize}
            step={1000}
            min={100}
            disabled={disabled}
            onChange={(v) => set("trainSize", Math.round(v))}
          />
          <NumberField
            label="Test images"
            value={settings.testSize}
            step={500}
            min={100}
            disabled={disabled}
            onChange={(v) => set("testSize", Math.round(v))}
          />
        </div>

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
          <p className="text-sm text-muted-foreground">
            Loaded {poolCount.toLocaleString()} images (28×28, greyscale).
          </p>
        )}
        {datasetError && (
          <p className="text-sm text-destructive">{datasetError}</p>
        )}
      </DialogContent>
    </Dialog>
  );
}
