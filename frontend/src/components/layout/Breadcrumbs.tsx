// Home › Category › Task, in the navbar of every task page (#65).
//
// The same trail the build writes into each route's BreadcrumbList JSON-LD, so
// a search result's breadcrumb and the page's own agree. The category links to
// its section of the home page's task index — the in-app route from one task
// to its siblings, since the sidebar starts collapsed.

import { Link, useRouterState } from "@tanstack/react-router";
import { ChevronRight } from "lucide-react";

import { NOTES_PATH } from "@/notes/meta";
import { categoryAnchor, pageForPath } from "@/seo/pages";

export function Breadcrumbs({ className }: { className?: string }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const page = pageForPath(pathname);
  // A note's parent is the notes index, as in its BreadcrumbList (#72).
  if (!page?.category && page?.kind !== "note") return null;

  return (
    <nav aria-label="Breadcrumb" className={className}>
      <ol className="flex min-w-0 items-center gap-1 text-sm text-muted-foreground">
        <li className="shrink-0">
          {page.category ? (
            <Link
              to="/home"
              hash={categoryAnchor(page.category)}
              className="hover:text-foreground hover:underline underline-offset-2"
            >
              {page.category}
            </Link>
          ) : (
            <Link
              to={NOTES_PATH}
              className="hover:text-foreground hover:underline underline-offset-2"
            >
              Notes
            </Link>
          )}
        </li>
        <li aria-hidden="true" className="shrink-0">
          <ChevronRight className="h-3.5 w-3.5" />
        </li>
        <li className="truncate text-foreground" aria-current="page">
          {page.heading}
        </li>
      </ol>
    </nav>
  );
}
