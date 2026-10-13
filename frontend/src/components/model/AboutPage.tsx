import type { PageCopy } from "@/seo/copy/types";
import { ABOUT_HEADING } from "@/seo/render";

/**
 * A task page's own paragraphs (#71), below the workbench: what the task is,
 * what runs it and what it cannot do, in the words someone would search with.
 *
 * Below rather than above, because the page is a tool first and the workbench
 * fills the viewport at `md` — the result must never be pushed under the fold
 * (page-pattern §4). It still has to be *in the app*: Google indexes the page
 * after rendering it, and the static HTML's copy of these words is replaced
 * when the app mounts. The build writes the same strings into that HTML.
 *
 * The heading does not repeat the page's name: the `<h1>` already carries it,
 * and a second heading matching it makes "the page's heading" ambiguous.
 */
export function AboutPage({ copy }: { copy: PageCopy }) {
  return (
    // Deliberately unnamed, so it is not a `region`: the four bands are the
    // page's regions, and the route tests count them (page-pattern §8).
    <section
      data-testid="about-page"
      className="mx-auto mt-10 w-full max-w-3xl space-y-3 border-t pt-6 text-sm text-muted-foreground"
    >
      <h2 className="text-base font-medium text-foreground">{ABOUT_HEADING}</h2>
      {copy.about.map((p) => (
        <p key={p.slice(0, 40)} className="leading-relaxed">
          {p}
        </p>
      ))}
    </section>
  );
}
