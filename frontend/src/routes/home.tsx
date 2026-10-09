// The in-app home / dashboard. It's the first thing a signed-in user sees and
// surfaces the WebGPU capability details up front — everything here runs in the
// browser; the backend only serves the model catalog.

import { createFileRoute, Link } from "@tanstack/react-router";

import { AnalyticsNote } from "@/components/analytics/AnalyticsNote";
import { BenchmarkCard } from "@/components/home/BenchmarkCard";
import { GpuCapabilitiesCard } from "@/components/home/GpuCapabilitiesCard";
import { ModelCatalogCard } from "@/components/home/ModelCatalogCard";
import { TaskIndex } from "@/components/home/TaskIndex";
import { taskCategories } from "@/components/layout/taskTaxonomy";
import { BACKEND_ENABLED } from "@/lib/features";

export const Route = createFileRoute("/home")({
  component: HomePage,
});

const TASK_COUNT = taskCategories.reduce((n, c) => n + c.tasks.length, 0);

function HomePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Model Playground</h1>
        <p className="text-sm text-muted-foreground">
          Run machine-learning models in your browser, on your own GPU —
          speech recognition, object detection, translation, text generation
          and {TASK_COUNT - 4} more tasks across audio, vision, language,
          graphs, tabular data and reinforcement learning. The weights come to
          you; nothing you give a model is sent to a server
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
