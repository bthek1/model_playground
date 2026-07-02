import { createFileRoute } from "@tanstack/react-router";
import { Cpu, Gauge, Layers } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useGpuBenchmark } from "@/hooks/useGpuBenchmark";
import { useModels } from "@/hooks/useModels";
import { useWebGPU } from "@/hooks/useWebGPU";

export const Route = createFileRoute("/playground")({
  component: PlaygroundPage,
});

const BENCH_SIZE = 512;

function CapabilityPanel() {
  const { capabilities, loading } = useWebGPU();

  if (loading) {
    return <p className="text-sm text-muted-foreground">Probing WebGPU…</p>;
  }
  if (!capabilities || capabilities.status !== "ready") {
    return (
      <p className="text-sm text-destructive">
        WebGPU is unavailable ({capabilities?.status ?? "unknown"}). Use a
        Chromium-based browser with WebGPU enabled to run models here.
      </p>
    );
  }

  const { adapter, features, limits, isFallbackAdapter } = capabilities;
  return (
    <div className="space-y-4 text-sm">
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1">
        <dt className="text-muted-foreground">Vendor</dt>
        <dd>{adapter?.vendor}</dd>
        <dt className="text-muted-foreground">Architecture</dt>
        <dd>{adapter?.architecture || "—"}</dd>
        <dt className="text-muted-foreground">Fallback adapter</dt>
        <dd>{isFallbackAdapter ? "yes (software)" : "no (hardware)"}</dd>
      </dl>

      <div>
        <p className="mb-1 text-muted-foreground">Limits</p>
        <ul className="space-y-0.5 font-mono text-xs">
          {Object.entries(limits).map(([key, value]) => (
            <li key={key}>
              {key}: {value.toLocaleString()}
            </li>
          ))}
        </ul>
      </div>

      {features.length > 0 && (
        <div>
          <p className="mb-1 text-muted-foreground">Features</p>
          <div className="flex flex-wrap gap-1">
            {features.map((f) => (
              <span
                key={f}
                className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

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

function ModelCatalog() {
  const { data: models, isLoading, isError } = useModels();

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading catalog…</p>;
  }
  if (isError) {
    return (
      <p className="text-sm text-destructive">Could not load the catalog.</p>
    );
  }
  if (!models || models.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No models yet. Add one via the Django admin or{" "}
        <code>POST /api/registry/models/</code>. See
        docs/guides/adding-a-model.md.
      </p>
    );
  }

  return (
    <ul className="divide-y">
      {models.map((model) => (
        <li key={model.id} className="flex items-baseline gap-3 py-2 text-sm">
          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs uppercase">
            {model.task}
          </span>
          <span className="font-medium">{model.name}</span>
          <span className="truncate text-muted-foreground">
            {model.description}
          </span>
        </li>
      ))}
    </ul>
  );
}

function PlaygroundPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Model Playground</h1>
        <p className="text-sm text-muted-foreground">
          Run models directly on your GPU via raw WebGPU. Nothing is sent to a
          server — the backend only serves the model catalog.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="size-4" /> GPU Capabilities
            </CardTitle>
            <CardDescription>What this browser exposes.</CardDescription>
          </CardHeader>
          <CardContent>
            <CapabilityPanel />
          </CardContent>
        </Card>

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
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="size-4" /> Model Catalog
          </CardTitle>
          <CardDescription>
            Registered models, served from the backend registry.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ModelCatalog />
        </CardContent>
      </Card>
    </div>
  );
}
