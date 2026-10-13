/**
 * Facts about the site itself: its name, its public address and the sentence
 * that describes it.
 *
 * Pure data with no imports, because two very different callers read it: the
 * app (tab titles, the head tags it keeps in step on navigation) and the build
 * (`src/seo/`, run inside `vite.config.ts` by Node, which has no `@/` alias and
 * no DOM). Anything added here must stay importable from both.
 */

/** The product name. Import it rather than retyping the string. */
export const APP_NAME = "Model Playground";

/**
 * The public origin every canonical URL, `og:url` and sitemap entry is written
 * against — the static deployment (#57, deployment.md §10). A build for another
 * host overrides it with `SITE_URL` in the build environment; the app itself
 * never reads this to decide where to send a request.
 */
export const SITE_URL = "https://playground.benedictthekkel.com";

/**
 * The site-wide meta description. `index.html` carries the same text in its
 * description, `og:description` and `twitter:description`, and so does
 * `site.webmanifest` — a test keeps all four equal (#72), because only the
 * first used to be checked and the other three drifted. It is what a page
 * with no description of its own — sign-in, a 404 — falls back to.
 */
export const SITE_DESCRIPTION =
  "Run machine-learning models in your browser on your own GPU: speech, vision, text, graphs, tables and RL. Your inputs stay on your device.";
