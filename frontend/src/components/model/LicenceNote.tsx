// A model's licence, stated where the model is chosen.
//
// Most checkpoints in this app are Apache-2.0 or MIT and the line is a fact
// nobody needs to act on. Some are not — RMBG-1.4 and the SegFormer heads are
// non-commercial, DINOv3 has Meta's own terms, two NLI heads state none at all —
// and each is a model someone would reasonably pick, use, and ship, which is
// exactly the mistake a licence buried in a model card lets you make.
//
// So the permissive case renders one quiet line and the restricted case renders
// a warning the same colour as the size-before-load guardrail, for the same
// reason: it is a cost the user is about to take on without being told.
//
// It started on `/background-removal` (#24) and moved here when every picker
// gained a licence line (#62) — moved rather than copied, like `backend.ts`.

import { AlertTriangle, Scale } from "lucide-react";

import { TERMS_LABEL, type ModelLicence } from "@/model/licences";

export function LicenceNote({
  licence,
  /** Shown before the licence when an entry is a pair, so the half is named. */
  subject,
}: {
  licence: ModelLicence;
  subject?: string;
}) {
  const prefix = subject ? `${subject}: ` : "";

  if (licence.terms === "permissive") {
    return (
      <p
        data-testid="model-licence"
        data-terms={licence.terms}
        className="flex items-center gap-1.5 text-xs text-muted-foreground"
      >
        <Scale className="size-3.5 shrink-0" />
        <span>
          {prefix}Licence:{" "}
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
      data-terms={licence.terms}
      className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500"
    >
      <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
      <span>
        {prefix}
        <span className="font-medium">
          {TERMS_LABEL[licence.terms]} ({licence.name})
        </span>{" "}
        — {licence.note ?? FALLBACK[licence.terms]}{" "}
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

const FALLBACK: Record<ModelLicence["terms"], string> = {
  permissive: "",
  "non-commercial": "not licensed for commercial use.",
  custom: "the publisher's own terms apply.",
  unstated: "its publisher states no licence, so its terms are unknown.",
};
