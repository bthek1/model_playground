import { Gauge } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useGpuBenchmark } from "@/hooks/useGpuBenchmark";

const BENCH_SIZE = 512;

function BenchmarkPanel() {
  const { running, result, error, run } = useGpuBenchmark(BENCH_SIZE);

  return (
    <div className="space-y-3 text-sm">
      <p className="text-muted-foreground">
        Runs a {BENCH_SIZE}×{BENCH_SIZE} matrix multiply on the GPU (in a Web
        Worker) using the raw WGSL kernel in{" "}
        <code>src/webgpu/shaders/matmul.wgsl</code>.
      </p>
      <Button onClick={run} disabled={running}>
        {running ? "Running…" : "Run GPU benchmark"}
      </Button>

      {error && <p className="text-destructive">Error: {error}</p>}

      {result && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 font-mono text-xs">
          <dt className="text-muted-foreground">Throughput</dt>
          <dd>{result.gflops.toFixed(1)} GFLOP/s</dd>
          <dt className="text-muted-foreground">Wall time</dt>
          <dd>{result.gpuTimeMs.toFixed(2)} ms</dd>
          <dt className="text-muted-foreground">Correctness</dt>
          <dd className={result.correct ? "" : "text-destructive"}>
            {result.correct ? "verified vs CPU" : "MISMATCH"}
          </dd>
        </dl>
      )}
    </div>
  );
}

/** Dashboard card that verifies the WebGPU compute pipeline end to end. */
export function BenchmarkCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Gauge className="size-4" /> Compute Benchmark
        </CardTitle>
        <CardDescription>Verify the WebGPU pipeline end to end.</CardDescription>
      </CardHeader>
      <CardContent>
        <BenchmarkPanel />
      </CardContent>
    </Card>
  );
}
