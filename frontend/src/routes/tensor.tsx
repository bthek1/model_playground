// Tensor Arithmetic — the reference view for the `custom` / tensor-ops task
// category. Per the model-visualization standard, tensor ops emphasise
// *structure*: the operands and the operation are laid out as a left-to-right
// dataflow schematic (Stage ─Arrow─▶ Stage), and the result is rendered both as
// a diverging heatmap (red = +, blue = −, alpha = magnitude) and as the raw
// numeric grid. See docs/standards/model-visualization.md.

import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Sigma } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Arrow, ParamChip, Stage } from "@/components/viz/schematic";
import {
  DivergingLegend,
  HeatmapTile,
  useMaxAbs,
} from "@/components/viz/heatmap";
import { useTensorOp } from "@/hooks/useTensorOp";
import { formatMatrix, parseMatrix } from "@/lib/matrix";
import { isBinaryOp, tensorOpSymbol } from "@/webgpu/tensorops";
import type { TensorOp, TensorOpJob, TensorOpResult } from "@/webgpu/types";

export const Route = createFileRoute("/tensor")({
  component: TensorArithmeticPage,
});

interface OpMeta {
  op: TensorOp;
  label: string;
  hint: string;
}

const OPS: OpMeta[] = [
  { op: "add", label: "Add", hint: "A + B — element-wise, same shape" },
  { op: "sub", label: "Subtract", hint: "A − B — element-wise, same shape" },
  { op: "mul", label: "Multiply", hint: "A ⊙ B — Hadamard, same shape" },
  { op: "div", label: "Divide", hint: "A ÷ B — element-wise, same shape" },
  { op: "matmul", label: "Matmul", hint: "A · B — inner dims must match" },
  { op: "transpose", label: "Transpose", hint: "Aᵀ — flips A's shape" },
  { op: "scale", label: "Scale", hint: "s · A — multiply A by a scalar" },
];

const SAMPLE_A = "1 2 3\n4 5 6";
const SAMPLE_B = "7 8 9\n10 11 12";

/** Parsed shape as `r×c`, or `—` while the text isn't a valid matrix. */
function useShape(text: string): string {
  return useMemo(() => {
    try {
      const { rows, cols } = parseMatrix(text);
      return `${rows}×${cols}`;
    } catch {
      return "—";
    }
  }, [text]);
}

function TensorArithmeticPage() {
  const [op, setOp] = useState<TensorOp>("add");
  const [textA, setTextA] = useState(SAMPLE_A);
  const [textB, setTextB] = useState(SAMPLE_B);
  const [scalar, setScalar] = useState("2");
  const [parseError, setParseError] = useState<string | null>(null);
  const { running, result, error, run, reset } = useTensorOp();

  const meta = OPS.find((o) => o.op === op)!;
  const needsB = isBinaryOp(op);
  const needsScalar = op === "scale";
  const shownError = parseError ?? error;

  const shapeA = useShape(textA);
  const shapeB = useShape(textB);

  // Label for the arrow feeding the result stage — reads as "A + B = …".
  const resultArrow =
    op === "transpose" ? "Aᵀ" : needsScalar ? "= s·A" : "= C";

  const onRun = () => {
    // Parse operands on the CPU first; a bad shape/number is a local error and
    // never reaches the GPU worker.
    let job: TensorOpJob;
    try {
      const a = parseMatrix(textA);
      job = { op, a: a.data, aRows: a.rows, aCols: a.cols };
      if (needsB) {
        const b = parseMatrix(textB);
        job.b = b.data;
        job.bRows = b.rows;
        job.bCols = b.cols;
      }
      if (needsScalar) {
        const s = Number(scalar);
        if (!Number.isFinite(s)) throw new Error(`"${scalar}" is not a number.`);
        job.scalar = s;
      }
    } catch (e) {
      reset();
      setParseError(e instanceof Error ? e.message : String(e));
      return;
    }
    setParseError(null);
    void run(job);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Sigma className="size-6" /> Tensor Arithmetic
        </h1>
        <p className="text-sm text-muted-foreground">
          Basic matrix and tensor operations, computed on your GPU via raw
          WebGPU in a Web Worker. Pick an operation, edit the operands — rows on
          separate lines, values separated by spaces or commas — then Compute.
        </p>
      </div>

      {/* Operation selector */}
      <div className="space-y-2">
        <div className="flex flex-wrap gap-2">
          {OPS.map((o) => (
            <Button
              key={o.op}
              variant={o.op === op ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setOp(o.op);
                setParseError(null);
                reset();
              }}
            >
              <span className="font-mono">{tensorOpSymbol(o.op)}</span>
              {o.label}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ParamChip label="op" value={tensorOpSymbol(op)} accent />
          <ParamChip label="arity" value={needsB ? "binary" : "unary"} />
          <p className="font-mono text-xs text-muted-foreground">{meta.hint}</p>
        </div>
      </div>

      {/* Dataflow schematic: A ─(op)─▶ [B | scalar] ─(=)─▶ Result */}
      <div className="flex flex-col items-stretch gap-4 xl:flex-row xl:items-start">
        <Stage title="Matrix A" sub={`shape ${shapeA}`} className="flex-1">
          <MatrixTextarea value={textA} onChange={setTextA} />
        </Stage>

        {needsB && (
          <>
            <Arrow label={tensorOpSymbol(op)} />
            <Stage title="Matrix B" sub={`shape ${shapeB}`} className="flex-1">
              <MatrixTextarea value={textB} onChange={setTextB} />
            </Stage>
          </>
        )}

        {needsScalar && (
          <>
            <Arrow label="×" />
            <Stage title="Scalar" sub="s · A" className="flex-1">
              <div className="grid gap-1.5">
                <Label htmlFor="scalar">Multiply every element of A by</Label>
                <input
                  id="scalar"
                  value={scalar}
                  onChange={(e) => setScalar(e.target.value)}
                  className="h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 font-mono text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
                />
              </div>
            </Stage>
          </>
        )}

        <Arrow label={resultArrow} />

        <Stage
          title="Result"
          sub={
            result
              ? `${result.rows}×${result.cols} · ${result.gpuTimeMs.toFixed(2)} ms on GPU`
              : "compute to run on the GPU"
          }
          className="flex-1"
        >
          {result ? (
            <ResultView result={result} />
          ) : (
            <p className="py-6 text-center text-sm text-muted-foreground">
              Press{" "}
              <span className="font-medium text-foreground">Compute</span> to
              evaluate {resultArrow === "Aᵀ" ? "Aᵀ" : "the operation"} on your
              GPU.
            </p>
          )}
        </Stage>
      </div>

      {/* Transport */}
      <div className="flex items-center gap-3">
        <Button onClick={onRun} disabled={running}>
          {running ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Computing…
            </>
          ) : (
            "Compute"
          )}
        </Button>
        {shownError && <p className="text-sm text-destructive">{shownError}</p>}
      </div>

      {/* Raw numeric dump (progressive disclosure: the heatmap leads, numbers
          back it up) */}
      {result && (
        <Stage title="Result values" sub="raw row-major output">
          <MatrixGrid data={result.data} rows={result.rows} cols={result.cols} />
        </Stage>
      )}
    </div>
  );
}

function MatrixTextarea({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      rows={5}
      spellCheck={false}
      className="w-full resize-y rounded-lg border border-input bg-transparent px-2.5 py-2 font-mono text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
    />
  );
}

/** Diverging heatmap of the result tensor + its legend. */
function ResultView({ result }: { result: TensorOpResult }) {
  const maxAbs = useMaxAbs(result.data);
  // Keep the true aspect ratio, capped so a wide/tall result stays readable.
  const px = 176 / Math.max(result.rows, result.cols);

  return (
    <div className="space-y-3">
      <div className="flex justify-center py-1">
        <HeatmapTile
          values={result.data}
          rows={result.rows}
          cols={result.cols}
          maxAbs={maxAbs}
          style={{
            width: `${result.cols * px}px`,
            height: `${result.rows * px}px`,
          }}
          aria-label="Result value heatmap"
        />
      </div>
      <DivergingLegend maxAbs={maxAbs} posLabel="positive" negLabel="negative" />
    </div>
  );
}

function MatrixGrid({
  data,
  rows,
  cols,
}: {
  data: Float32Array;
  rows: number;
  cols: number;
}) {
  const grid = formatMatrix(data, rows, cols);
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 font-mono text-sm">
        <tbody>
          {grid.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td
                  key={c}
                  className="min-w-14 rounded-md bg-muted px-2.5 py-1 text-right tabular-nums"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
