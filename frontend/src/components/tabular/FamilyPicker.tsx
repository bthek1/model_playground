// SELECT for the Tabular pages: which rung of the ladder, and its knobs.
//
// The counterpart of `ModelPicker`, and deliberately not that component: there
// is no download to quote, no cache to probe and no model id to resolve on a
// Hub. What a user needs before pressing FIT is different — **how long it will
// take on this many rows**, and **where the arithmetic will run**, which is the
// one sentence this category exists to say out loud.
//
// Every control here is a choice, not an action. Changing the family or moving
// a slider fits nothing; it invalidates nothing already on screen either. Only
// FIT spends.

import { Cpu, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  describeDuration,
  estimateFitMs,
  MAX_BOOST_DEPTH,
  MAX_TREE_DEPTH,
} from "@/tabular/limits";
import { familyInfo, KNOB_LABELS, type FamilyInfo } from "@/tabular/families";
import type { Family, Hyperparams } from "@/tabular/types";

export function FamilyPicker({
  families,
  value,
  onChange,
  hp,
  onHp,
  seed,
  onSeed,
  rows,
  classes,
  disabled = false,
}: {
  families: FamilyInfo[];
  value: Family;
  onChange: (family: Family) => void;
  hp: Hyperparams;
  onHp: (hp: Hyperparams) => void;
  seed: number;
  onSeed: (seed: number) => void;
  /** Training rows, for the time estimate. Zero before a dataset is loaded. */
  rows: number;
  classes: number;
  disabled?: boolean;
}) {
  const info = familyInfo(value);
  const estimate = rows > 0 ? estimateFitMs(value, rows, hp, classes) : null;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {families.map((f) => (
          <Button
            key={f.id}
            size="sm"
            variant={f.id === value ? "default" : "outline"}
            aria-pressed={f.id === value}
            disabled={disabled}
            onClick={() => onChange(f.id)}
          >
            {f.label}
          </Button>
        ))}
      </div>

      <p className="text-xs leading-snug text-muted-foreground">{info.blurb}</p>

      <p
        data-testid="family-compute"
        className="flex items-start gap-1.5 text-xs text-muted-foreground"
      >
        {info.compute === "gpu" ? (
          <Sparkles className="mt-0.5 size-3.5 shrink-0" />
        ) : (
          <Cpu className="mt-0.5 size-3.5 shrink-0" />
        )}
        <span>
          <span className="font-medium text-foreground">
            {info.compute === "gpu" ? "Runs on your GPU." : "Runs on your CPU."}
          </span>{" "}
          {info.computeNote}
        </span>
      </p>

      {estimate != null && (
        <p data-testid="fit-estimate" className="text-xs text-muted-foreground">
          Estimated fit: <strong>{describeDuration(estimate)}</strong> on{" "}
          {rows.toLocaleString()} training rows.
        </p>
      )}

      <div className="space-y-2.5 border-t pt-3">
        {info.knobs.map((knob) => (
          <Knob
            key={knob}
            knob={knob}
            family={value}
            value={hp[knob]}
            disabled={disabled}
            onChange={(v) => onHp({ ...hp, [knob]: v })}
          />
        ))}
        <div className="space-y-1">
          <label
            htmlFor="tabular-seed"
            className="flex items-baseline justify-between text-xs font-medium"
          >
            Random seed
            <span className="font-mono text-xs text-muted-foreground tabular-nums">
              {seed}
            </span>
          </label>
          <input
            id="tabular-seed"
            type="number"
            value={seed}
            disabled={disabled}
            onChange={(e) => onSeed(Number(e.target.value) || 0)}
            className="h-8 w-full rounded-md border bg-background px-2 text-xs tabular-nums"
          />
          <p className="text-xs text-muted-foreground">
            Drives the split, the bagging and the importance shuffles. It is on
            screen so a fit can be repeated exactly — a head-to-head between two
            families on two different splits compares the splits.
          </p>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Nothing is downloaded and nothing is fitted until you press Fit.
      </p>
    </div>
  );
}

function Knob({
  knob,
  family,
  value,
  onChange,
  disabled,
}: {
  knob: keyof Hyperparams;
  family: Family;
  value: number;
  onChange: (v: number) => void;
  disabled: boolean;
}) {
  const meta = KNOB_LABELS[knob];
  // Boosting's depth cap is lower than the forest's, and the number came from
  // Phase 0's measurement rather than from taste — see `limits.ts`.
  const max =
    knob === "maxDepth"
      ? family === "boosting"
        ? MAX_BOOST_DEPTH
        : MAX_TREE_DEPTH
      : meta.max;
  const id = `hp-${knob}`;
  return (
    <div className="space-y-1">
      <label
        htmlFor={id}
        className="flex items-baseline justify-between text-xs font-medium"
      >
        {meta.label}
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {value}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={meta.min}
        max={max}
        step={meta.step}
        value={Math.min(value, max)}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
      <p className="text-xs leading-snug text-muted-foreground">{meta.hint}</p>
    </div>
  );
}
