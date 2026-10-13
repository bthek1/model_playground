// One note (#72). `notes_` un-nests it from `/notes`, so the index is a page of
// its own rather than a layout around every article.
//
// The body is loaded by the route's loader, one chunk per note through
// `import.meta.glob`: the index and the metadata are small and live in the
// entry, the articles do not. Each page's HTML preloads its own body chunk
// (`scripts/seoPages.ts`), so the loader resolves from cache and the app does
// not mount an empty article over the static one.

import { createFileRoute, Link, notFound } from "@tanstack/react-router";

import { NoteBlocks } from "@/components/notes/NoteBlocks";
import { OPERATOR } from "@/legal/site";
import { NOTES, NOTES_PATH } from "@/notes/meta";
import type { NoteBlock } from "@/notes/types";
import { pageForPath } from "@/seo/pages";

const BODIES = import.meta.glob<{ body: readonly NoteBlock[] }>([
  "../notes/bodies/*.ts",
  "!../notes/bodies/index.ts",
]);

export const Route = createFileRoute("/notes_/$slug")({
  loader: async ({ params }) => {
    const meta = NOTES.find((n) => n.slug === params.slug);
    const load = BODIES[`../notes/bodies/${params.slug}.ts`];
    if (!meta || !load) throw notFound();
    return { meta, body: (await load()).body };
  },
  component: NotePage,
});

function NotePage() {
  const { meta, body } = Route.useLoaderData();
  const related = meta.related.flatMap((path) => {
    const page = pageForPath(path);
    return page ? [page] : [];
  });

  return (
    <article className="mx-auto max-w-3xl space-y-4 pb-8 text-sm leading-relaxed">
      <header className="space-y-2">
        <p className="text-xs text-muted-foreground">
          <Link to={NOTES_PATH} className="underline underline-offset-2">
            Notes
          </Link>
        </p>
        <h1 className="text-2xl font-semibold">{meta.title}</h1>
        <p className="text-xs text-muted-foreground" data-testid="note-byline">
          By {OPERATOR} · <time dateTime={meta.published}>{meta.published}</time>
          {meta.modified !== meta.published && (
            <>
              {" "}
              · updated <time dateTime={meta.modified}>{meta.modified}</time>
            </>
          )}
        </p>
      </header>
      <div className="space-y-3 text-muted-foreground" data-testid="note-body">
        <NoteBlocks blocks={body} />
      </div>
      {related.length > 0 && (
        <section className="space-y-2 border-t pt-4">
          <h2 className="text-base font-semibold">Try it</h2>
          <ul className="flex flex-wrap gap-x-4 gap-y-1">
            {related.map((p) => (
              <li key={p.path}>
                <Link to={p.path} className="underline underline-offset-2">
                  {p.heading}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
