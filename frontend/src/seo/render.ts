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
 * Pure and relative-imported for the same reason as `pages.ts`.
 */

import { APP_NAME } from "../lib/site";
import { taskCategories } from "../components/layout/taskTaxonomy";
import {
  canonicalUrl,
  categoryAnchor,
  SITE_PAGES,
  siblingsOf,
  type SitePage,
} from "./pages";

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

/** Schema.org data for the page: the app itself on home, a breadcrumb elsewhere. */
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
    };
  }
  const crumbs: { name: string; item?: string }[] = [{ name: APP_NAME, item: home }];
  if (page.category) {
    crumbs.push({ name: page.category, item: `${home}#${categoryAnchor(page.category)}` });
  }
  crumbs.push({ name: page.heading, item: canonicalUrl(page.path, origin) });
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: crumbs.map((c, i) => ({
      "@type": "ListItem",
      position: i + 1,
      name: c.name,
      item: c.item,
    })),
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
 * What a crawler that runs no JavaScript sees, and what everyone sees for the
 * moment before the app mounts over it. `createRoot().render` replaces the
 * children of `#root` (this is not hydration), so none of it has to match the
 * React tree — but it must say the same things the page does.
 */
export function staticBody(page: SitePage): string {
  const parts: string[] = [];
  if (page.kind !== "home") {
    const trail = [link("/home", APP_NAME)];
    if (page.category) {
      trail.push(link(`/home#${categoryAnchor(page.category)}`, page.category));
    }
    trail.push(escapeHtml(page.heading));
    parts.push(
      `<nav aria-label="Breadcrumb" class="text-sm text-muted-foreground">${trail.join(" › ")}</nav>`,
    );
  }
  parts.push(`<h1 class="text-2xl font-semibold">${escapeHtml(page.heading)}</h1>`);
  parts.push(`<p class="text-muted-foreground">${escapeHtml(page.description)}</p>`);
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
  } else if (page.category) {
    parts.push(
      `<section><h2 class="text-lg font-medium">More in ${escapeHtml(page.category)}</h2>${taskList(siblingsOf(page))}</section>`,
    );
  }
  return `<main class="mx-auto max-w-3xl space-y-4 p-8">${parts.join("")}</main>`;
}

/** The built `index.html`, rewritten for one route. Throws on a template it cannot read. */
export function renderPageHtml(template: string, page: SitePage, origin: string): string {
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
  return html.replace('<div id="root"></div>', () => `<div id="root">${staticBody(page)}</div>`);
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
