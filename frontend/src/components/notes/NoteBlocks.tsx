// A note's body as elements (#72) — the same blocks the build writes into the
// page's static HTML (`seo/render.ts` `noteBodyHtml`), so the article a
// crawler reads and the one the app shows are one source.

import { Link } from "@tanstack/react-router";

import { parseInline } from "@/notes/inline";
import type { NoteBlock } from "@/notes/types";

const linkClass = "underline underline-offset-2";

export function Inline({ text }: { text: string }) {
  return (
    <>
      {parseInline(text).map((t, i) => {
        if (t.kind === "code") {
          return (
            <code key={i} className="rounded bg-muted px-1 font-mono text-[0.9em]">
              {t.text}
            </code>
          );
        }
        if (t.kind === "link") {
          return t.href.startsWith("/") ? (
            <Link key={i} to={t.href} className={linkClass}>
              {t.text}
            </Link>
          ) : (
            <a key={i} href={t.href} target="_blank" rel="noreferrer" className={linkClass}>
              {t.text}
            </a>
          );
        }
        return <span key={i}>{t.text}</span>;
      })}
    </>
  );
}

export function NoteBlocks({ blocks }: { blocks: readonly NoteBlock[] }) {
  return (
    <>
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "p":
            return (
              <p key={i}>
                <Inline text={b.text} />
              </p>
            );
          case "h2":
            return (
              <h2 key={i} className="pt-2 text-base font-semibold text-foreground">
                <Inline text={b.text} />
              </h2>
            );
          case "ul":
            return (
              <ul key={i} className="list-disc space-y-1 pl-5">
                {b.items.map((item, j) => (
                  <li key={j}>
                    <Inline text={item} />
                  </li>
                ))}
              </ul>
            );
          case "code":
            return (
              <pre key={i} className="overflow-x-auto rounded-md border bg-muted p-3 text-xs">
                <code>{b.text}</code>
              </pre>
            );
        }
      })}
    </>
  );
}
