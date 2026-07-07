// Hyperparameters (plain mini-batch SGD) as a popup dialog. Fields are locked
// while a run is in flight so the user can't mutate the live training config.

import type { ReactNode } from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { TrainingSettings } from "@/hooks/useLinearTraining";

import { NumberField } from "./controls";

export function HyperparamsDialog({
  trigger,
  settings,
  set,
  disabled,
}: {
  trigger: ReactNode;
  settings: TrainingSettings;
  set: <K extends keyof TrainingSettings>(k: K, v: TrainingSettings[K]) => void;
  disabled: boolean;
}) {
  return (
    <Dialog>
      <DialogTrigger render={trigger as React.ReactElement} />
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Hyperparameters</DialogTitle>
          <DialogDescription>Plain mini-batch SGD.</DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Learning rate"
            value={settings.learningRate}
            step={0.05}
            min={0.001}
            disabled={disabled}
            onChange={(v) => set("learningRate", v)}
          />
          <NumberField
            label="Batch size"
            value={settings.batchSize}
            step={16}
            min={1}
            disabled={disabled}
            onChange={(v) => set("batchSize", Math.round(v))}
          />
          <NumberField
            label="Epochs"
            value={settings.epochs}
            step={1}
            min={1}
            disabled={disabled}
            onChange={(v) => set("epochs", Math.round(v))}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
