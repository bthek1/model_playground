// Generic landing page for a task in the sidebar taxonomy. Real routes are
// mapped directly (see taskTaxonomy.ts); every other task falls through to here.
//
// It renders the same four-slot shell as an implemented task, with every slot
// empty (docs/standards/model-page-pattern.md §7). That is deliberate: an
// unimplemented task should read as *this page without a model yet*, not as a
// different kind of page. It also shows the user exactly what will appear here
// when the task lands.

import { createFileRoute, Link } from "@tanstack/react-router";
import { Construction } from "lucide-react";

import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { categoryForPath, tasksBySlug } from "@/components/layout/taskTaxonomy";

export const Route = createFileRoute("/tasks/$slug")({
  component: TaskPlaceholderPage,
});

/** Muted stand-in for a slot that has nothing behind it yet. */
function EmptySlot({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded-md border border-dashed px-3 py-2 text-sm text-muted-foreground">
      {children}
    </p>
  );
}

function TaskPlaceholderPage() {
  const { slug } = Route.useParams();
  const taskItem = tasksBySlug[slug];

  if (!taskItem) {
    return (
      <div className="mx-auto max-w-2xl p-8">
        <Card>
          <CardHeader>
            <CardTitle>Unknown task</CardTitle>
            <CardDescription>
              No task matches <code>{slug}</code>.{" "}
              <Link to="/playground" className="underline">
                Back to Playground
              </Link>
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  const category = categoryForPath(taskItem.to);

  return (
    <ModelPage
      icon={Construction}
      title={taskItem.label}
      description={
        <>
          {category ? `${category} · ` : ""}In-browser inference for this task is
          on the roadmap. When it lands it will run here the same way every other
          task does — pick a model, load it, run it, see the output.
        </>
      }
      select={<EmptySlot>No models in the catalogue for this task yet.</EmptySlot>}
      load={
        <EmptySlot>
          Nothing to load — models are downloaded to your browser and run
          locally, and none is wired up for {taskItem.label} yet.
        </EmptySlot>
      }
      run={<EmptySlot>The input surface for this task isn't built yet.</EmptySlot>}
      output={
        // The real OutputPanel, not a lookalike card — the placeholder should be
        // this page with nothing in it, which means the same components.
        <OutputPanel
          title={
            <>
              <Construction className="size-4" /> Not available yet
            </>
          }
          running={false}
          empty={
            <>
              <strong className="text-foreground">{taskItem.label}</strong> is on
              the roadmap. In the meantime, the audio tasks and the theory tools
              are live —{" "}
              <Link to="/playground" className="underline">
                back to Playground
              </Link>
              .
            </>
          }
        />
      }
    />
  );
}
