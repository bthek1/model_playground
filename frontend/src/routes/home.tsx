// The in-app home / dashboard. It's the first thing a signed-in user sees and
// surfaces the WebGPU capability details up front — everything here runs in the
// browser; the backend only serves the model catalog.

import { createFileRoute } from "@tanstack/react-router";

import { BenchmarkCard } from "@/components/home/BenchmarkCard";
import { GpuCapabilitiesCard } from "@/components/home/GpuCapabilitiesCard";
import { ModelCatalogCard } from "@/components/home/ModelCatalogCard";

export const Route = createFileRoute("/home")({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Model Playground</h1>
        <p className="text-sm text-muted-foreground">
          Run ML models directly on your GPU via raw WebGPU. Nothing is sent to a
          server — the backend only serves the model catalog.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <GpuCapabilitiesCard />
        <BenchmarkCard />
      </div>

      <ModelCatalogCard />
    </div>
  );
}
