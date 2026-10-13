// The notes index (#72): every write-up, newest first, each with the sentence
// that says what problem it answers. The build writes the same list into this
// page's static HTML (`seo/render.ts`).

import { createFileRoute, Link } from "@tanstack/react-router";

import { NOTES, notePath } from "@/notes/meta";

export const Route = createFileRoute("/notes")({
  component: NotesIndexPage,
});

function NotesIndexPage() {
  return (
    <article className="mx-auto max-w-3xl space-y-6 pb-8 text-sm leading-relaxed">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold">Notes</h1>
        <p className="text-muted-foreground">
          Problems found while building this site, written up with the error message, the cause
          and the fix. Each links to the page where you can see it.
        </p>
      </header>
      <ul className="space-y-4" data-testid="notes-index">
        {NOTES.map((n) => (
          <li key={n.slug} className="space-y-1">
            <Link to={notePath(n.slug)} className="font-medium underline underline-offset-2">
              {n.title}
            </Link>
            <p className="text-muted-foreground">{n.description}</p>
            <p className="text-xs text-muted-foreground">
              <time dateTime={n.published}>{n.published}</time>
            </p>
          </li>
        ))}
      </ul>
    </article>
  );
}
