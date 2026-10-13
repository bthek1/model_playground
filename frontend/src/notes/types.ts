/**
 * Notes (#72): short technical articles, each about one thing building this
 * site turned up — an error message, a missing field, a quantization that is
 * wrong rather than worse. They answer searches people make word for word, and
 * each links to the page that demonstrates it.
 *
 * Plain data, never JSX or Markdown: the build runs this in Node and writes the
 * whole article into the page's static HTML, and the app renders the same
 * blocks, so the two cannot say different things. Relative imports only.
 */

export interface NoteMeta {
  /** The URL segment: `/notes/<slug>`. Lower-case kebab, no dots. */
  slug: string;
  /**
   * The headline: the `<h1>`, and the `<title>` before the site suffix. At most
   * 51 characters, like a task's search title, and led by the words someone
   * with the problem would type.
   */
  title: string;
  /** The meta description, 70–150 characters. */
  description: string;
  /** ISO dates. `modified` moves only when the article's substance does. */
  published: string;
  modified: string;
  /** The task routes that demonstrate the finding; each links back. */
  related: readonly string[];
}

/**
 * One block of an article. Text carries two inline forms, parsed by
 * `inline.ts`: `` `code` `` and `[label](/path)` (or an `https://` URL).
 */
export type NoteBlock =
  | { kind: "p"; text: string }
  | { kind: "h2"; text: string }
  | { kind: "ul"; items: readonly string[] }
  | { kind: "code"; text: string };
