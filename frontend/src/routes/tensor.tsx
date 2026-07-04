import { createFileRoute } from "@tanstack/react-router";
import { Loader2, Sigma } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useTensorOp } from "@/hooks/useTensorOp";
import { formatMatrix, parseMatrix } from "@/lib/matrix";
import { cn } from "@/lib/utils";
import { isBinaryOp, tensorOpSymbol } from "@/webgpu/tensorops";
import type { TensorOp, TensorOpJob } from "@/webgpu/types";

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
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Sigma className="size-6" /> Tensor Arithmetic
        </h1>
        <p className="text-sm text-muted-foreground">
          Basic matrix and tensor operations, computed on your GPU via raw
          WebGPU in a Web Worker. Enter matrices below — rows on separate lines,
          values separated by spaces or commas.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Operation</CardTitle>
          <CardDescription>{meta.hint}</CardDescription>
        </CardHeader>
        <CardContent>
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
        </CardContent>
      </Card>

      <div className={cn("grid gap-6", needsB && "md:grid-cols-2")}>
        <Card>
          <CardHeader>
            <CardTitle>Matrix A</CardTitle>
          </CardHeader>
          <CardContent>
            <MatrixTextarea value={textA} onChange={setTextA} />
          </CardContent>
        </Card>

        {needsB && (
          <Card>
            <CardHeader>
              <CardTitle>Matrix B</CardTitle>
            </CardHeader>
            <CardContent>
              <MatrixTextarea value={textB} onChange={setTextB} />
            </CardContent>
          </Card>
        )}
      </div>

      {needsScalar && (
        <Card>
          <CardHeader>
            <CardTitle>Scalar</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid max-w-xs gap-1.5">
              <Label htmlFor="scalar">Multiply every element of A by</Label>
              <input
                id="scalar"
                value={scalar}
                onChange={(e) => setScalar(e.target.value)}
                className="h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
              />
            </div>
          </CardContent>
        </Card>
      )}

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

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>
              Result{" "}
              <span className="text-sm font-normal text-muted-foreground">
                {result.rows}×{result.cols} · {result.gpuTimeMs.toFixed(2)} ms
                on GPU
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <MatrixGrid
              data={result.data}
              rows={result.rows}
              cols={result.cols}
            />
          </CardContent>
        </Card>
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
  const shape = useMemo(() => {
    try {
      const { rows, cols } = parseMatrix(value);
      return `${rows}×${cols}`;
    } catch {
      return "—";
    }
  }, [value]);

  return (
    <div className="space-y-1.5">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={5}
        spellCheck={false}
        className="w-full resize-y rounded-lg border border-input bg-transparent px-2.5 py-2 font-mono text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
      />
      <p className="text-right text-xs text-muted-foreground">shape {shape}</p>
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
