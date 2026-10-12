/**
 * Vite plugin: the build's SEO output (#63, #64).
 *
 *   dist/<route>/index.html   one per indexable page, from `src/seo/`
 *   dist/sitemap.xml          every indexable page
 *   dist/robots.txt           allow all but /ingest/, and name the sitemap
 *   dist/404.html             the shell for a path with no page, `noindex`
 *   dist/seo-manifest.json    page → digest of what a crawler reads (#69)
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

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { Plugin } from "vite";

import { SITE_URL } from "../src/lib/site";
import { APP_ONLY_PATHS, SITE_PAGES } from "../src/seo/pages";
import {
  absolutiseImages,
  renderNotFoundHtml,
  renderPageHtml,
  renderRobots,
  renderSitemap,
  seoManifest,
} from "../src/seo/render";

const ROUTES = new Set(SITE_PAGES.map((p) => p.path));
const APP_ONLY = new Set(APP_ONLY_PATHS);

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
      // A path with no page gets /404.html with a 404, as `object-status.js`
      // makes it at the edge; `/` is left to the app (the edge 301s it), and
      // `/ingest/*` is its own CloudFront behaviour that never reaches here.
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url ?? "/", "http://preview");
        const path = url.pathname;
        const isFile = path.slice(path.lastIndexOf("/") + 1).includes(".");
        if (ROUTES.has(path)) {
          req.url = `${path}/index.html${url.search}`;
        } else if (
          !isFile &&
          path !== "/" &&
          !APP_ONLY.has(path) &&
          !path.startsWith("/ingest/")
        ) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "text/html; charset=utf-8");
          res.end(readFileSync(join(outDir, "404.html")));
          return;
        }
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
      writeFileSync(join(outDir, "404.html"), renderNotFoundHtml(template, origin));
      const sha256 = (text: string) => createHash("sha256").update(text).digest("hex");
      writeFileSync(
        join(outDir, "seo-manifest.json"),
        `${JSON.stringify(seoManifest(SITE_PAGES, origin, sha256), null, 2)}\n`,
      );
    },
  };
}
