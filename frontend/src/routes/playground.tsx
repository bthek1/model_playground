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

/** One step of the shared task-page pipeline. */
function Step({
  n,
  title,
  children,
}: {
  n: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <li className="flex gap-2.5">
      <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[0.65rem] font-semibold text-muted-foreground tabular-nums">
        {n}
      </span>
      <span>
        <span className="font-medium text-foreground">{title}</span> — {children}
      </span>
    </li>
  );
}

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
            <Cpu className="size-4" /> How a task page works
          </CardTitle>
          <CardDescription>
            Every task in the sidebar is the same four steps.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <ol className="grid gap-2 sm:grid-cols-2">
            <Step n={1} title="Select">
              Pick a model from the task's catalogue. Its download size is quoted
              before anything is fetched.
            </Step>
            <Step n={2} title="Load">
              Press Load and the weights stream into your browser, then stay
              cached. Nothing downloads until you ask.
            </Step>
            <Step n={3} title="Run">
              Give it an input — text, a mic take, a file, a matrix. The model
              runs in a Web Worker on your GPU, or your CPU if there's no WebGPU.
            </Step>
            <Step n={4} title="Output">
              The result appears with whatever it needs: playback, a download, a
              heatmap, a transcript.
            </Step>
          </ol>
          <p>
            Pick a task from the sidebar to start. Tasks without a model yet show
            the same page with empty steps.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
