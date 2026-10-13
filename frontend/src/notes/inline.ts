/**
 * The two inline forms a note's text may use: `` `code` `` and
 * `[label](href)`. Parsed into tokens so the build can write them as HTML
 * and the app as elements, from one reading of the string.
 *
 * Deliberately not Markdown: two forms is all the articles need, and a
 * parser this small is one whose output a test can enumerate.
 */

export type InlineToken =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "link"; text: string; href: string };

const PATTERN = /`([^`]+)`|\[([^\]]+)\]\(((?:\/|https:\/\/)[^)\s]*)\)/g;

export function parseInline(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let last = 0;
  for (const m of text.matchAll(PATTERN)) {
    if (m.index > last) tokens.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[1] !== undefined) tokens.push({ kind: "code", text: m[1] });
    else tokens.push({ kind: "link", text: m[2], href: m[3] });
    last = m.index + m[0].length;
  }
  if (last < text.length) tokens.push({ kind: "text", text: text.slice(last) });
  return tokens;
}

/** The text as a reader sees it, markup removed — for word counts and search. */
export function plainText(text: string): string {
  return parseInline(text)
    .map((t) => t.text)
    .join("");
}
