// A readable schematic of the model's architecture — Linear Softmax Classifier,
// 784 → 10 — so the whole thing makes intuitive sense at a glance:
//
//   INPUT (28×28 = 784)  ──►  z = W·x + b  ──►  softmax  ──►  10 class probs
//
// The middle stage shows *every* parameter: each output class is a 28×28 image of
// its 784 weights (red = +ve / for the digit, blue = −ve / against), plus its
// bias — the ModelWeights templates. It reads the live weights, so the templates
// sharpen into digit shapes as training runs.

import { useMemo } from "react";

import { ModelWeights } from "./ModelWeights";

const IMAGE_SIZE = 784;
const SIDE = 28;

export function ModelArchitecture({
  weights,
  bias,
  epoch,
  numClasses = 10,
}: {
  weights: Float32Array;
  bias?: Float32Array;
  epoch?: number;
  numClasses?: number;
}) {
  const hasWeights = weights.length >= IMAGE_SIZE * numClasses;
  const trained = useMemo(
    () => hasWeights && weights.some((v) => v !== 0),
    [weights, hasWeights],
  );

  const weightParams = IMAGE_SIZE * numClasses;
  const total = weightParams + numClasses;

  return (
    <div className="space-y-4">
      {/* Header + parameter budget */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-semibold">
            Linear Softmax Classifier{" "}
            <span className="text-muted-foreground">
              · 784 → {numClasses} · MNIST
            </span>
          </p>
          <p className="font-mono text-xs text-muted-foreground">
            z = W·x + b · p = softmax(z) · argmax(p) = digit
          </p>
        </div>
        <div className="flex gap-1.5 font-mono text-[11px]">
          <ParamChip label="W 784×10" value={weightParams.toLocaleString()} />
          <ParamChip label="b 10" value={String(numClasses)} />
          <ParamChip label="total" value={total.toLocaleString()} accent />
        </div>
      </div>

      {/* Flow: INPUT ──► WEIGHTS (10 × 28×28) ──► OUTPUT */}
      <div className="flex flex-col items-stretch gap-5 xl:flex-row xl:items-center">
        <Stage title="INPUT LAYER" sub="28×28 = 784 px">
          <div className="mx-auto w-40">
            <div
              aria-label="28×28 input pixel grid"
              className="aspect-square w-full rounded border bg-muted/30"
              style={{
                backgroundImage:
                  "linear-gradient(to right, rgba(148,163,184,0.3) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.3) 1px, transparent 1px)",
                backgroundSize: `${100 / SIDE}% ${100 / SIDE}%`,
              }}
            />
            <p className="mt-1.5 text-center font-mono text-xs text-muted-foreground">
              x₀ … x₇₈₃
            </p>
          </div>
        </Stage>

        <Arrow label="W·x + b" />

        <Stage
          title="WEIGHTS + BIAS"
          sub="10 classes × 784 weights (+ 10 biases) — every parameter"
        >
          <ModelWeights
            weights={weights}
            bias={bias}
            numClasses={numClasses}
            epoch={epoch}
            tileSize="size-32"
          />
        </Stage>

        <Arrow label="softmax" />

        <Stage title="OUTPUT" sub="10 logits → probabilities">
          <div className="space-y-2">
            {Array.from({ length: numClasses }, (_, c) => (
              <div key={c} className="flex items-center gap-2">
                <span className="font-mono text-xs text-muted-foreground">
                  z{c}+b{c}→
                </span>
                <span className="flex size-10 items-center justify-center rounded-full border-2 font-mono text-lg font-semibold tabular-nums">
                  {c}
                </span>
              </div>
            ))}
          </div>
        </Stage>
      </div>

      {!trained && (
        <p className="text-xs text-muted-foreground italic">
          Start training to watch each weight template sharpen into a digit shape.
        </p>
      )}
    </div>
  );
}

function ParamChip({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 ${
        accent ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"
      }`}
    >
      {label} <span className="font-semibold">{value}</span>
    </span>
  );
}

function Stage({
  title,
  sub,
  children,
  className,
}: {
  title: string;
  sub: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-lg border bg-card/60 p-2.5 ${className ?? ""}`}>
      <p className="text-[11px] font-semibold tracking-wide">{title}</p>
      <p className="mb-2 text-[10px] text-muted-foreground">{sub}</p>
      {children}
    </div>
  );
}

function Arrow({ label }: { label: string }) {
  return (
    <div className="flex shrink-0 flex-col items-center justify-center gap-1 text-muted-foreground">
      <span className="font-mono text-xs">{label}</span>
      {/* Big long-arrow connector; rotates to point down when stages stack. */}
      <span
        aria-hidden
        className="rotate-90 text-5xl leading-none xl:rotate-0"
      >
        ⟶
      </span>
    </div>
  );
}
