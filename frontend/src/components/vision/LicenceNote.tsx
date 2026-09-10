// A model's licence, stated where the model is chosen.
//
// Most checkpoints in this app are Apache-2.0 or MIT and the line is a fact
// nobody needs to act on. One is not: `briaai/RMBG-1.4` is non-commercial only,
// and it is a model someone would reasonably pick, use, and ship — which is
// exactly the mistake a licence buried in a model card lets you make.
//
// So the permissive case renders one quiet line and the restricted case renders
// a warning the same colour as the size-before-load guardrail, for the same
// reason: it is a cost the user is about to take on without being told.

import { AlertTriangle, Scale } from "lucide-react";

import type { ModelLicence } from "@/vision/backgroundRemoval";

export function LicenceNote({ licence }: { licence: ModelLicence }) {
  if (licence.commercial) {
    return (
      <p
        data-testid="model-licence"
        className="flex items-center gap-1.5 text-xs text-muted-foreground"
      >
        <Scale className="size-3.5 shrink-0" />
        <span>
          Licence:{" "}
          <a
            href={licence.url}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2"
          >
            {licence.name}
          </a>{" "}
          — commercial use permitted.
        </span>
      </p>
    );
  }

  return (
    <p
      data-testid="model-licence"
      className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500"
    >
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
      <span>
        <span className="font-medium">{licence.name}</span> —{" "}
        {licence.note ?? "not licensed for commercial use."}{" "}
        <a
          href={licence.url}
          target="_blank"
          rel="noreferrer"
          className="underline underline-offset-2"
        >
          Read the licence
        </a>
        .
      </span>
    </p>
  );
}
