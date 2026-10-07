// The shell the four legal pages share (#62): a readable column, the title, the
// date it last changed and the contact, then numbered-by-heading sections.
//
// These are prose pages, not task pages — no SELECT/LOAD/RUN/OUTPUT, no worker,
// nothing in the sidebar taxonomy. They live in the app shell so the footer
// that links them is always one click from any model.

import type { ReactNode } from "react";

import { CONTACT_EMAIL, LAST_UPDATED } from "@/legal/site";
import { cn } from "@/lib/utils";

export function LegalPage({
  title,
  intro,
  children,
  dated = true,
}: {
  title: string;
  intro: ReactNode;
  children: ReactNode;
  /** The privacy notice and terms carry a revision date; the licence list does not. */
  dated?: boolean;
}) {
  return (
    <article className="mx-auto max-w-3xl space-y-8 pb-8 text-sm leading-relaxed">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold">{title}</h1>
        {dated && (
          <p data-testid="legal-updated" className="text-xs text-muted-foreground">
            Last updated <time dateTime={LAST_UPDATED}>{LAST_UPDATED}</time>
          </p>
        )}
        <div className="space-y-2 text-muted-foreground">{intro}</div>
      </header>
      {children}
    </article>
  );
}

export function Section({
  title,
  children,
  id,
}: {
  title: string;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section aria-labelledby={id ? `${id}-h` : undefined} className="space-y-3">
      <h2 id={id ? `${id}-h` : undefined} className="text-base font-semibold">
        {title}
      </h2>
      {children}
    </section>
  );
}

/** A table that scrolls inside itself, so a phone never scrolls the page sideways. */
export function LegalTable({
  caption,
  head,
  children,
  testId,
}: {
  caption: string;
  head: readonly string[];
  children: ReactNode;
  testId?: string;
}) {
  return (
    <div className="overflow-x-auto rounded-md border">
      <table data-testid={testId} className="w-full text-left text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-muted/50">
          <tr>
            {head.map((h) => (
              <th key={h} scope="col" className="px-3 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y">{children}</tbody>
      </table>
    </div>
  );
}

export function Cell({ children, className }: { children: ReactNode; className?: string }) {
  return <td className={cn("px-3 py-2 align-top", className)}>{children}</td>;
}

export function ExternalLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
      {children}
    </a>
  );
}

export function ContactLink() {
  return (
    <a href={`mailto:${CONTACT_EMAIL}`} className="underline underline-offset-2">
      {CONTACT_EMAIL}
    </a>
  );
}
