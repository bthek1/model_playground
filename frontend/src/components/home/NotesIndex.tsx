// The notes, on home under the task index (#72): body-text links from the
// most-linked page to every write-up, as the static HTML has them.

import { Link } from "@tanstack/react-router";

import { NOTES, NOTES_PATH, notePath } from "@/notes/meta";

export function NotesIndex() {
  return (
    <section id="notes" aria-labelledby="notes-heading" className="scroll-mt-4 space-y-2">
      <h2 id="notes-heading" className="text-lg font-semibold">
        <Link to={NOTES_PATH} className="underline-offset-2 hover:underline">
          Notes
        </Link>
      </h2>
      <ul className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
        {NOTES.map((n) => (
          <li key={n.slug} className="text-sm">
            <Link to={notePath(n.slug)} className="font-medium underline-offset-2 hover:underline">
              {n.title}
            </Link>
            <span className="text-muted-foreground"> — {n.description}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
