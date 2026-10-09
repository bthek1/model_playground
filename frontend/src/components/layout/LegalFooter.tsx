// The legal links, on every page (#62). One slim row under the workbench rather
// than a footer below scrolled content: task pages clamp to the viewport, so a
// footer after `main` would never be reached on them.

import { Link } from "@tanstack/react-router";

import { CONTACT_EMAIL } from "@/legal/site";
import { cn } from "@/lib/utils";

const LEGAL_LINKS = [
  { to: "/privacy", label: "Privacy" },
  { to: "/terms", label: "Terms" },
  { to: "/licences", label: "Licences" },
  { to: "/accessibility", label: "Accessibility" },
] as const;

export function LegalFooter({ className }: { className?: string }) {
  return (
    <footer
      data-testid="legal-footer"
      className={cn(
        "flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-t px-4 py-1.5 text-xs text-muted-foreground",
        className,
      )}
    >
      <nav aria-label="Legal" className="flex flex-wrap gap-x-4 gap-y-1">
        {LEGAL_LINKS.map((l) => (
          <Link
            key={l.to}
            to={l.to}
            className="hover:text-foreground hover:underline underline-offset-2"
            activeProps={{ className: "text-foreground", "aria-current": "page" }}
          >
            {l.label}
          </Link>
        ))}
        <a
          href={`mailto:${CONTACT_EMAIL}`}
          className="hover:text-foreground hover:underline underline-offset-2"
        >
          Contact
        </a>
      </nav>
      <span className="ml-auto hidden sm:inline">
        Models run in your browser. Outputs are machine-generated and can be wrong.
      </span>
    </footer>
  );
}
