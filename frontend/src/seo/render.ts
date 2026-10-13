/**
 * The build-time half of the site's SEO (#63, #64, #65): pure string functions
 * that turn the built `index.html` and the page list into one HTML file per
 * route, a sitemap and a robots.txt. `scripts/seoPages.ts` does the file I/O.
 *
 * Why a file per route at all: every URL used to be served the same 1.8 KB
 * shell — one title, one description, an empty `<div id="root">`. Google fills
 * that in after rendering the JavaScript, late; link-preview bots and most
 * other crawlers never do. Each route's own file carries its title,
 * description, canonical, Open Graph tags, JSON-LD and a short static summary
 * of the page, and the app then mounts over it exactly as it did before.
 *
 * Pure and relative-imported for the same reason as `pages.ts`. A task page's
 * own words (`seo/copy/`) are passed **in** rather than imported: the app
 * imports this module for `structuredData`, so an import here would put every
 * page's copy in the entry chunk (#71).
 */

import { AUTHOR_URL, OPERATOR, SOURCE_URL } from "../legal/site";
import { APP_NAME } from "../lib/site";
import { taskCategories } from "../components/layout/taskTaxonomy";
import {
  canonicalUrl,
  categoryAnchor,
  HOME_LEAD,
  HOME_PRIVACY,
  SITE_PAGES,
  siblingsOf,
  type SitePage,
} from "./pages";
import type { PageCopy } from "./copy/types";
import { parseInline } from "../notes/inline";
import { NOTES, NOTES_PATH, notePath, notesFor } from "../notes/meta";
import type { NoteBlock } from "../notes/types";

const OG_IMAGE_PATH = "/og-image.png";

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Replace a `<meta {attr}="{key}" content="…">` tag's content, whatever its
 * whitespace, or add the tag before `</head>` when the template has none.
 */
function setMeta(html: string, attr: string, key: string, value: string): string {
  const tag = `<meta ${attr}="${key}" content="${escapeHtml(value)}" />`;
  const existing = new RegExp(
    `<meta\\s+${attr}="${key.replace(/[.:]/g, "\\$&")}"\\s+content="[^"]*"\\s*/?>`,
  );
  return existing.test(html)
    ? html.replace(existing, () => tag)
    : html.replace("</head>", () => `  ${tag}\n  </head>`);
}

/** Point `og:image` and `twitter:image` at an absolute URL: the OG spec needs one. */
export function absolutiseImages(html: string, origin: string): string {
  const image = canonicalUrl(OG_IMAGE_PATH, origin);
  return setMeta(setMeta(html, "property", "og:image", image), "name", "twitter:image", image);
}

/** The about page, which the author's `@id` points into. */
export const ABOUT_PATH = "/about";

/**
 * The person behind the site (#72): named, with a profile a search engine can
 * match to them. Who made a page is one of the things Google's quality raters
 * are told to look for, and nothing on the site said.
 */
export function author(origin: string): object {
  return {
    "@type": "Person",
    "@id": `${canonicalUrl(ABOUT_PATH, origin)}#author`,
    name: OPERATOR,
    url: AUTHOR_URL,
    sameAs: [AUTHOR_URL],
  };
}

/**
 * Schema.org data for the page: the app itself on home; elsewhere a breadcrumb,
 * and on a task page the page itself as a `WebPage` that is part of the app
 * (#71), so its title and description are claims about *this* URL. Home and
 * the about page name the author (#72).
 */
export function structuredData(page: SitePage, origin: string): object {
  const home = canonicalUrl("/home", origin);
  if (page.kind === "home") {
    return {
      "@context": "https://schema.org",
      "@type": "WebApplication",
      name: APP_NAME,
      url: home,
      description: page.description,
      applicationCategory: "DeveloperApplication",
      operatingSystem: "Any",
      browserRequirements:
        "Requires JavaScript. WebGPU recommended; most models fall back to WebAssembly.",
      isAccessibleForFree: true,
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
      author: author(origin),
      subjectOf: {
        "@type": "SoftwareSourceCode",
        codeRepository: SOURCE_URL,
        programmingLanguage: "TypeScript",
      },
    };
  }
  const crumbs: { name: string; item?: string }[] = [{ name: APP_NAME, item: home }];
  if (page.category) {
    crumbs.push({ name: page.category, item: `${home}#${categoryAnchor(page.category)}` });
  }
  if (page.kind === "note") {
    crumbs.push({ name: "Notes", item: canonicalUrl(NOTES_PATH, origin) });
  }
  const url = canonicalUrl(page.path, origin);
  crumbs.push({ name: page.heading, item: url });
  const breadcrumb = {
    "@type": "BreadcrumbList",
    itemListElement: crumbs.map((c, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: c.name,
      item: c.item,
    })),
  };
  if (page.kind === "note") {
    return {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "TechArticle",
          "@id": url,
          url,
          mainEntityOfPage: url,
          headline: page.heading,
          name: page.title,
          description: page.description,
          datePublished: page.published,
          dateModified: page.modified,
          author: author(origin),
          isPartOf: { "@type": "WebApplication", name: APP_NAME, url: home },
          breadcrumb,
        },
      ],
    };
  }
  if (page.path === ABOUT_PATH) {
    return {
      "@context": "https://schema.org",
      "@graph": [
        {
          "@type": "AboutPage",
          "@id": url,
          url,
          name: page.title,
          description: page.description,
          isPartOf: { "@type": "WebApplication", name: APP_NAME, url: home },
          mainEntity: author(origin),
          breadcrumb,
        },
      ],
    };
  }
  if (page.kind !== "task") return { "@context": "https://schema.org", ...breadcrumb };
  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": url,
        url,
        name: page.title,
        headline: page.heading,
        description: page.description,
        isPartOf: { "@type": "WebApplication", name: APP_NAME, url: home },
        breadcrumb,
      },
    ],
  };
}

/** JSON for a `<script>` body: `<` escaped so the text cannot close the tag. */
export function scriptJson(value: object): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

function link(path: string, text: string): string {
  return `<a href="${escapeHtml(path)}" class="underline underline-offset-2">${escapeHtml(text)}</a>`;
}

function taskList(pages: readonly SitePage[]): string {
  const items = pages
    .map(
      (p) =>
        `<li>${link(p.path, p.heading)} <span class="text-muted-foreground">— ${escapeHtml(p.description)}</span></li>`,
    )
    .join("");
  return `<ul class="space-y-1 text-sm">${items}</ul>`;
}

/**
 * A task page's siblings as bare links. Their descriptions used to come too,
 * and made each page mostly other pages' text: Google then quoted the Depth
 * Estimation sentence as the snippet for Mask Generation (#71).
 */
function siblingLinks(pages: readonly SitePage[]): string {
  const items = pages.map((p) => `<li>${link(p.path, p.heading)}</li>`).join("");
  return `<ul class="flex flex-wrap gap-x-4 gap-y-1 text-sm">${items}</ul>`;
}

/** A note's inline markup as HTML: escaped text, `code`, and links. */
export function inlineHtml(text: string): string {
  return parseInline(text)
    .map((t) =>
      t.kind === "code"
        ? `<code class="rounded bg-muted px-1 font-mono text-[0.9em]">${escapeHtml(t.text)}</code>`
        : t.kind === "link"
          ? link(t.href, t.text)
          : escapeHtml(t.text),
    )
    .join("");
}

/** A note's body, block by block — the whole article, so no crawler needs JS (#72). */
export function noteBodyHtml(blocks: readonly NoteBlock[]): string {
  return blocks
    .map((b) => {
      switch (b.kind) {
        case "p":
          return `<p>${inlineHtml(b.text)}</p>`;
        case "h2":
          return `<h2 class="text-lg font-medium">${inlineHtml(b.text)}</h2>`;
        case "ul":
          return `<ul class="list-disc space-y-1 pl-5">${b.items.map((i) => `<li>${inlineHtml(i)}</li>`).join("")}</ul>`;
        case "code":
          return `<pre class="overflow-x-auto rounded-md bg-muted p-3 text-xs"><code>${escapeHtml(b.text)}</code></pre>`;
      }
    })
    .join("");
}

/** The notes that cite a task page, as links — the way back from page to write-up. */
function relatedNotes(path: string): string {
  const notes = notesFor(path);
  if (notes.length === 0) return "";
  const items = notes.map((n) => `<li>${link(notePath(n.slug), n.title)}</li>`).join("");
  return `<section><h2 class="text-lg font-medium">${escapeHtml(NOTES_HEADING)}</h2><ul class="space-y-1 text-sm">${items}</ul></section>`;
}

/** Every note as a link and its description: the notes index, and home's section. */
function notesList(): string {
  const items = NOTES.map(
    (n) =>
      `<li>${link(notePath(n.slug), n.title)} <span class="text-muted-foreground">— ${escapeHtml(n.description)}</span></li>`,
  ).join("");
  return `<ul class="space-y-1 text-sm">${items}</ul>`;
}

/** The heading over a task page's related notes, shared with the app's `RelatedNotes`. */
export const NOTES_HEADING = "Notes";

/** A task page's own paragraphs, under the same heading the app gives them. */
function aboutSection(copy: PageCopy): string {
  const paras = copy.about.map((p) => `<p>${escapeHtml(p)}</p>`).join("");
  return `<section class="space-y-2 text-sm"><h2 class="text-lg font-medium">${escapeHtml(ABOUT_HEADING)}</h2>${paras}</section>`;
}

/**
 * What a crawler that runs no JavaScript sees, and what everyone sees for the
 * moment before the app mounts over it. `createRoot().render` replaces the
 * children of `#root` (this is not hydration), so none of it has to match the
 * React tree — but it must say the same things the page does, and on a task
 * page it says them in the same words: the tagline and the about paragraphs
 * are the strings the route renders (#71).
 */
export function staticBody(page: SitePage, copy?: PageCopy, note?: readonly NoteBlock[]): string {
  const parts: string[] = [];
  if (page.kind !== "home") {
    const trail = [link("/home", APP_NAME)];
    if (page.category) {
      trail.push(link(`/home#${categoryAnchor(page.category)}`, page.category));
    }
    if (page.kind === "note") trail.push(link(NOTES_PATH, "Notes"));
    trail.push(escapeHtml(page.heading));
    parts.push(
      `<nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">${trail.join(" › ")}</nav>`,
    );
  }
  parts.push(`<h1 class="text-2xl font-semibold">${escapeHtml(page.heading)}</h1>`);
  // Home's paragraph is the app's intro, not the meta description: it is the
  // page's LCP element, and it must stay larger than anything React paints (#67).
  parts.push(
    page.kind === "home"
      ? `<p class="text-muted-foreground">${escapeHtml(`${HOME_LEAD} ${HOME_PRIVACY}`)} (${link("/privacy", "privacy")}).</p>`
      : page.kind === "note"
        ? `<p class="text-xs text-muted-foreground">By ${escapeHtml(OPERATOR)} · <time datetime="${escapeHtml(page.published ?? "")}">${escapeHtml(page.published ?? "")}</time></p>`
        : `<p class="text-muted-foreground">${escapeHtml(copy?.tagline ?? page.description)}</p>`,
  );
  parts.push(
    `<noscript><p>${escapeHtml(APP_NAME)} runs its models in your browser, so it needs JavaScript (and, for most models, WebGPU).</p></noscript>`,
  );

  if (page.kind === "home") {
    for (const category of taskCategories) {
      const pages = SITE_PAGES.filter((p) => p.category === category.label);
      parts.push(
        `<section id="${categoryAnchor(category.label)}"><h2 class="text-lg font-medium">${escapeHtml(category.label)}</h2>${taskList(pages)}</section>`,
      );
    }
    parts.push(
      `<section id="notes"><h2 class="text-lg font-medium">${link(NOTES_PATH, "Notes")}</h2>${notesList()}</section>`,
    );
  } else if (page.kind === "note") {
    if (note) parts.push(`<article class="space-y-3">${noteBodyHtml(note)}</article>`);
  } else if (page.path === NOTES_PATH) {
    parts.push(notesList());
  } else if (page.category) {
    if (copy) parts.push(aboutSection(copy));
    parts.push(relatedNotes(page.path));
    parts.push(
      `<nav aria-label="More in ${escapeHtml(page.category)}"><h2 class="text-lg font-medium">More in ${escapeHtml(page.category)}</h2>${siblingLinks(siblingsOf(page))}</nav>`,
    );
  }
  return `<main class="mx-auto max-w-3xl space-y-4 p-8">${parts.join("")}</main>`;
}

/** The about section's heading, shared with the app's `AboutPage`. */
export const ABOUT_HEADING = "About this page";

/** The built `index.html`, rewritten for one route. Throws on a template it cannot read. */
export function renderPageHtml(
  template: string,
  page: SitePage,
  origin: string,
  copy?: PageCopy,
  note?: readonly NoteBlock[],
): string {
  if (!/<title>[^<]*<\/title>/.test(template) || !template.includes('<div id="root"></div>')) {
    throw new Error("seo: index.html has no <title> or no empty #root to fill");
  }
  const url = canonicalUrl(page.path, origin);
  let html = template.replace(/<title>[^<]*<\/title>/, () => `<title>${escapeHtml(page.title)}</title>`);
  html = setMeta(html, "name", "description", page.description);
  html = setMeta(html, "property", "og:title", page.title);
  html = setMeta(html, "property", "og:description", page.description);
  html = setMeta(html, "property", "og:url", url);
  html = setMeta(html, "name", "twitter:title", page.title);
  html = setMeta(html, "name", "twitter:description", page.description);
  html = absolutiseImages(html, origin);
  const head = [
    `<link rel="canonical" href="${escapeHtml(url)}" />`,
    `<script type="application/ld+json" id="structured-data">${scriptJson(structuredData(page, origin))}</script>`,
  ];
  html = html.replace("</head>", () => `  ${head.join("\n    ")}\n  </head>`);
  return html.replace('<div id="root"></div>', () => `<div id="root">${staticBody(page, copy, note)}</div>`);
}

/**
 * `/404.html`: the shell the edge serves, with a 404 status, for a path that
 * has no page (#63). The app still mounts and renders its not-found page; this
 * is what a crawler reads first. `noindex` as well as the status, so a
 * misconfigured edge that sent it as a 200 still keeps it out of the index.
 */
export function renderNotFoundHtml(template: string, origin: string): string {
  if (!/<title>[^<]*<\/title>/.test(template) || !template.includes('<div id="root"></div>')) {
    throw new Error("seo: index.html has no <title> or no empty #root to fill");
  }
  let html = template.replace(
    /<title>[^<]*<\/title>/,
    () => `<title>${escapeHtml(suffix404)}</title>`,
  );
  html = setMeta(html, "name", "robots", "noindex");
  html = absolutiseImages(html, origin);
  const body = `<main class="mx-auto max-w-3xl space-y-4 p-8"><h1 class="text-2xl font-semibold">Page not found</h1><p class="text-muted-foreground">There is no page at this address.</p><p>${link("/home", "Browse every task")}</p></main>`;
  return html.replace('<div id="root"></div>', () => `<div id="root">${body}</div>`);
}

const suffix404 = `Page not found · ${APP_NAME}`;

/**
 * Everything a crawler reads on a page, as one string: title, description,
 * canonical, JSON-LD and the static body. Not the built HTML, which names
 * hashed `/assets/…` chunks that change on most deploys while nothing a search
 * engine sees has. The build hashes this into `seo-manifest.json`, and the
 * IndexNow step (#69) submits only pages whose digest moved.
 */
export function crawlableContent(
  page: SitePage,
  origin: string,
  copy?: PageCopy,
  note?: readonly NoteBlock[],
): string {
  return JSON.stringify([
    page.title,
    page.description,
    canonicalUrl(page.path, origin),
    structuredData(page, origin),
    staticBody(page, copy, note),
  ]);
}

/**
 * Page path → digest of its crawlable content. The hash and the page copy are
 * injected: this module also runs in the browser, where neither belongs.
 */
export function seoManifest(
  pages: readonly SitePage[],
  origin: string,
  hash: (text: string) => string,
  copyFor: (path: string) => PageCopy | undefined = () => undefined,
  noteFor: (path: string) => readonly NoteBlock[] | undefined = () => undefined,
): Record<string, string> {
  return Object.fromEntries(
    pages.map((p) => [p.path, hash(crawlableContent(p, origin, copyFor(p.path), noteFor(p.path)))]),
  );
}

/**
 * The sitemap. No `<lastmod>`: every page would carry the build date whether
 * or not it changed, and Google ignores a lastmod that is not accurate.
 */
export function renderSitemap(pages: readonly SitePage[], origin: string): string {
  const urls = pages
    .map((p) => `  <url><loc>${escapeHtml(canonicalUrl(p.path, origin))}</loc></url>`)
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;
}

/** Everything is crawlable except the analytics relay, which is not a page. */
export function renderRobots(origin: string): string {
  return [
    "User-agent: *",
    "Allow: /",
    "Disallow: /ingest/",
    "",
    `Sitemap: ${canonicalUrl("/sitemap.xml", origin)}`,
    "",
  ].join("\n");
}
