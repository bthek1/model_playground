// The notes that cite this task page (#72), under its "About this page"
// section: the way back from a page to the write-up of what it demonstrates.
// The build writes the same links into the static HTML (`seo/render.ts`).
//
// Reads the path from the router if there is one, and renders nothing if not:
// the task-route tests mount the page component without a router, and a
// missing link list is the right answer there.

import { Link, useRouter } from "@tanstack/react-router";

import { notePath, notesFor } from "@/notes/meta";
import { NOTES_HEADING } from "@/seo/render";

export function RelatedNotes() {
  const router = useRouter({ warn: false });
  const path = router?.state.location.pathname;
  const notes = path ? notesFor(path) : [];
  if (notes.length === 0) return null;

  return (
    <div data-testid="related-notes" className="space-y-1">
      <h3 className="font-medium text-foreground">{NOTES_HEADING}</h3>
      <ul className="space-y-1">
        {notes.map((n) => (
          <li key={n.slug}>
            <Link to={notePath(n.slug)} className="underline underline-offset-2">
              {n.title}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
