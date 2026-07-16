// Generic landing page for a task in the sidebar taxonomy. Real routes are
// mapped directly (see taskTaxonomy.ts); every other task falls through to here
// and renders a "not yet available" placeholder rather than a 404.

import { createFileRoute, Link } from "@tanstack/react-router";
import { Construction } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { categoryForPath, tasksBySlug } from "@/components/layout/taskTaxonomy";

export const Route = createFileRoute("/tasks/$slug")({
  component: TaskPlaceholderPage,
});

function TaskPlaceholderPage() {
  const { slug } = Route.useParams();
  const taskItem = tasksBySlug[slug];

  if (!taskItem) {
    return (
      <div className="mx-auto max-w-2xl">
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
    <div className="mx-auto max-w-2xl">
      <Card>
        <CardHeader>
          {category && (
            <CardDescription className="uppercase tracking-wide">
              {category}
            </CardDescription>
          )}
          <CardTitle className="text-2xl">{taskItem.label}</CardTitle>
          <CardDescription>
            This task isn't available in the playground yet.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
          <Construction className="h-5 w-5 shrink-0" />
          <span>
            In-browser inference for <strong>{taskItem.label}</strong> is on the
            roadmap. Check back soon.
          </span>
        </CardContent>
      </Card>
    </div>
  );
}
