// The in-app home / dashboard. It's the first thing a signed-in user sees and
// surfaces the WebGPU capability details up front — everything here runs in the
// browser; the backend only serves the model catalog.

import { createFileRoute, Link } from "@tanstack/react-router";

import { AnalyticsNote } from "@/components/analytics/AnalyticsNote";
import { BenchmarkCard } from "@/components/home/BenchmarkCard";
import { GpuCapabilitiesCard } from "@/components/home/GpuCapabilitiesCard";
import { ModelCatalogCard } from "@/components/home/ModelCatalogCard";
import { TaskIndex } from "@/components/home/TaskIndex";
import { BACKEND_ENABLED } from "@/lib/features";
import { HOME_LEAD, HOME_PRIVACY } from "@/seo/pages";

export const Route = createFileRoute("/home")({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      {/* Two paragraphs, not one: the static HTML paints these words as a
          single larger paragraph, and keeping each of these smaller is what
          stops React's paint from becoming a later LCP (#67). */}
      <div className="space-y-1 text-sm text-muted-foreground">
        <h1 className="mb-1 text-2xl font-semibold text-foreground">Model Playground</h1>
        <p>{HOME_LEAD}</p>
        <p>
          {HOME_PRIVACY}
          {BACKEND_ENABLED && " — the backend only serves the model catalog"}{" "}
          (<Link to="/privacy" className="underline underline-offset-2">privacy</Link>).{" "}
          <AnalyticsNote />
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <GpuCapabilitiesCard />
        <BenchmarkCard />
      </div>

      {/* The registry is the only thing here that needs /api (#57). */}
      {BACKEND_ENABLED && <ModelCatalogCard />}

      <TaskIndex />
    </div>
  );
}
