// The inference workspace. GPU capability details and the compute benchmark now
// live on the Home dashboard (routes/home.tsx); this page is where you pick a
// model and run it in the browser.

import { createFileRoute } from "@tanstack/react-router";
import { Cpu } from "lucide-react";

import { ModelCatalogCard } from "@/components/home/ModelCatalogCard";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export const Route = createFileRoute("/playground")({
  component: PlaygroundPage,
});

function PlaygroundPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div>
        <h1 className="mb-1 text-2xl font-semibold">Playground</h1>
        <p className="text-sm text-muted-foreground">
          Pick a model and run inference in the browser. See your GPU details on
          the{" "}
          <a href="/home" className="underline">
            Home dashboard
          </a>
          .
        </p>
      </div>

      <ModelCatalogCard />

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4" /> Run
          </CardTitle>
          <CardDescription>
            In-browser inference is wired up per task — see the sidebar.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Select a task from the sidebar to run a model. Text generation and
          other inference surfaces are added as their kernels land.
        </CardContent>
      </Card>
    </div>
  );
}
