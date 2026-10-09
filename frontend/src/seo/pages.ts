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
  /** The meta description, 70–160 characters. */
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
