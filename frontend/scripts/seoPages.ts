/**
 * Vite plugin: the build's SEO output (#63, #64).
 *
 *   dist/<route>/index.html   one per indexable page, from `src/seo/`
 *   dist/sitemap.xml          every indexable page
 *   dist/robots.txt           allow all but /ingest/, and name the sitemap
 *
 * and an absolute `og:image` in the root `index.html` (the OG spec needs one).
 * The CloudFront function (`infra/site/spa-rewrite.js`) serves a route's own
 * file when it has one and the root shell otherwise; a test in `src/seo/` keeps
 * its route set equal to `SITE_PAGES`. `vite preview` gets the same mapping,
 * because the static E2E pass (`E2E_STATIC=1`) runs against it and its own SPA
 * fallback would serve the root shell for `/asr` — testing a page production
 * never serves. The dev server writes nothing and serves the shell for every
 * route, as before.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Plugin } from "vite";

import { SITE_URL } from "../src/lib/site";
import { SITE_PAGES } from "../src/seo/pages";
import {
  absolutiseImages,
  renderPageHtml,
  renderRobots,
  renderSitemap,
} from "../src/seo/render";

const ROUTES = new Set(SITE_PAGES.map((p) => p.path));

export function seoPages({ origin = SITE_URL }: { origin?: string } = {}): Plugin {
  let outDir = "dist";
  return {
    name: "seo-pages",
    configResolved(config) {
      outDir = resolve(config.root, config.build.outDir);
    },
    configurePreviewServer(server) {
      // What `infra/site/spa-rewrite.js` does at the edge: an indexable route
      // is served its own file. Added directly, so it runs before preview's
      // static handler and SPA fallback.
      server.middlewares.use((req, _res, next) => {
        const url = new URL(req.url ?? "/", "http://preview");
        if (ROUTES.has(url.pathname)) req.url = `${url.pathname}/index.html${url.search}`;
        next();
      });
    },
    transformIndexHtml(html) {
      return absolutiseImages(html, origin);
    },
    writeBundle() {
      const template = readFileSync(join(outDir, "index.html"), "utf8");
      for (const page of SITE_PAGES) {
        const dir = join(outDir, page.path);
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, "index.html"), renderPageHtml(template, page, origin));
      }
      writeFileSync(join(outDir, "sitemap.xml"), renderSitemap(SITE_PAGES, origin));
      writeFileSync(join(outDir, "robots.txt"), renderRobots(origin));
    },
  };
}
