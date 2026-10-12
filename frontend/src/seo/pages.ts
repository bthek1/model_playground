/**
 * Every page the site wants indexed, with the title, description and place in
 * the taxonomy a search engine is told about (#63, #64, #65).
 *
 * Built from the sidebar taxonomy rather than listed: the sitemap, the per-route
 * HTML the build writes, the edge function's route set, the tab title and the
 * home page's task index all read this one list, and a route added to the
 * taxonomy reaches every one of them without a second edit.
 *
 * Imports are relative, never `@/`: `vite.config.ts` runs this module in Node
 * at build time, where the alias does not exist.
 */

import { taskCategories } from "../components/layout/taskTaxonomy";
import { APP_NAME, SITE_DESCRIPTION, SITE_URL } from "../lib/site";

export type PageKind = "home" | "task" | "static";

export interface SitePage {
  /** The route, e.g. `/asr`. Lower-case, no trailing slash. */
  path: string;
  kind: PageKind;
  /** The `<title>` — for a task, the label plus "in your browser". */
  title: string;
  /** The page's `<h1>`: the taxonomy label, or the static page's name. */
  heading: string;
  /** The meta description, 70–150 characters (about 920 px, #68). */
  description: string;
  /** The owning taxonomy category, for a task page. */
  category?: string;
}

/** The home page's title: the product, then what it does. */
export const HOME_TITLE = `${APP_NAME}: ML models in your browser, on your GPU`;

/** `"Privacy"` → `"Privacy · Model Playground"`. */
export function suffixTitle(name: string): string {
  return `${name} · ${APP_NAME}`;
}

/** A task's title: the Hub task name first, so open tabs stay distinguishable. */
export function taskTitle(label: string): string {
  return suffixTitle(`${label} in your browser`);
}

/** `"Computer Vision"` → `"computer-vision"`, the home page's anchor for it. */
export function categoryAnchor(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

const STATIC_PAGES: SitePage[] = [
  {
    path: "/privacy",
    kind: "static",
    title: suffixTitle("Privacy"),
    heading: "Privacy",
    description:
      "What Model Playground sends where: models run in your browser, so your inputs stay on your device. The hosts your browser contacts, and why.",
  },
  {
    path: "/terms",
    kind: "static",
    title: suffixTitle("Terms of use"),
    heading: "Terms of use",
    description:
      "Terms of use for Model Playground, a free in-browser machine-learning playground: no warranty, machine-generated output, acceptable use.",
  },
  {
    path: "/licences",
    kind: "static",
    title: suffixTitle("Licences"),
    heading: "Licences",
    description:
      "The licence of every model, dataset and bundled package Model Playground uses, with non-commercial restrictions shown where they apply.",
  },
  {
    path: "/accessibility",
    kind: "static",
    title: suffixTitle("Accessibility"),
    heading: "Accessibility",
    description:
      "How Model Playground aims to meet WCAG 2.2 AA, what is known not to yet, and how to report an accessibility problem.",
  },
];

const TASK_COUNT = taskCategories.reduce((n, c) => n + c.tasks.length, 0);

/**
 * The home page's introduction, shared by the app and the static HTML (#67).
 * The two must agree for a reason beyond consistency: this paragraph is the
 * page's largest contentful paint. The static copy paints at first paint, and
 * `routes/home.tsx` splits the same words into two smaller paragraphs, so
 * nothing React paints later is larger and LCP is not pushed back to when the
 * entry chunk has run (1.8 s later on a throttled phone).
 */
export const HOME_LEAD = `Run machine-learning models in your browser, on your own GPU — speech recognition, object detection, translation, text generation and ${TASK_COUNT - 4} more tasks across audio, vision, language, graphs, tabular data and reinforcement learning.`;

/** The intro's second half; the app and the static page each append a privacy link. */
export const HOME_PRIVACY = "The weights come to you; nothing you give a model is sent to a server";

const HOME: SitePage = {
  path: "/home",
  kind: "home",
  title: HOME_TITLE,
  heading: APP_NAME,
  description: SITE_DESCRIPTION,
};

/**
 * The indexable pages, home first. Sign-in and sign-up are deliberately absent
 * — an account form is not a search result — and so is `/`, which the edge
 * redirects to `/home` (#63).
 */
export const SITE_PAGES: readonly SitePage[] = [
  HOME,
  ...taskCategories.flatMap((category) =>
    category.tasks.map(
      (t): SitePage => ({
        path: t.to,
        kind: "task",
        title: taskTitle(t.label),
        heading: t.label,
        description: t.description,
        category: category.label,
      }),
    ),
  ),
  ...STATIC_PAGES,
];

/**
 * Routes the app has that are not indexable: they get the plain shell at the
 * edge, with a 200. Every other path with no page gets `/404.html` and a real
 * 404 (#63). `infra/site/spa-rewrite.js` carries a copy as `APP_ROUTES`, which
 * `edgeRoutes.test.ts` keeps equal to this.
 */
export const APP_ONLY_PATHS: readonly string[] = ["/login", "/signup"];

const BY_PATH = new Map(SITE_PAGES.map((p) => [p.path, p]));

/** Lower-case and drop a trailing slash: `/ASR/` → `/asr`, `/` stays `/`. */
export function normalisePath(pathname: string): string {
  const lower = pathname.toLowerCase();
  return lower.length > 1 ? lower.replace(/\/+$/, "") || "/" : lower;
}

/** The indexable page at this pathname, if it is one. */
export function pageForPath(pathname: string): SitePage | undefined {
  return BY_PATH.get(normalisePath(pathname));
}

/** The absolute canonical URL of a path. */
export function canonicalUrl(path: string, origin: string = SITE_URL): string {
  return `${origin.replace(/\/+$/, "")}${path}`;
}

/** The task pages that share a category with this one, itself excluded. */
export function siblingsOf(page: SitePage): SitePage[] {
  if (!page.category) return [];
  return SITE_PAGES.filter(
    (p) => p.category === page.category && p.path !== page.path,
  );
}
