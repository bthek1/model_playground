/**
 * Per-route document titles.
 *
 * Twenty-odd task routes shared one tab title before this, so a user with
 * several tasks open could not tell them apart. The names are *not* re-listed
 * here: the sidebar taxonomy already maps every task to its route and its label,
 * and re-typing them would be a second source of truth that drifts the first
 * time a task is renamed.
 *
 * Pure on purpose — the hook that calls it is three lines, and the mapping is
 * what is worth testing.
 */

import { APP_NAME } from "@/components/layout/Logo";
import { taskCategories } from "@/components/layout/taskTaxonomy";

/** Routes that aren't tasks, so aren't in the taxonomy. */
const STATIC_TITLES: Record<string, string> = {
  "/": "",
  "/home": "Home",
  "/login": "Sign in",
  "/signup": "Create account",
};

function suffix(name: string): string {
  return name ? `${name} · ${APP_NAME}` : APP_NAME;
}

/** Route path -> task label, built once from the taxonomy. */
const BY_ROUTE = new Map<string, string>();
/** Task slug -> label, for the generic `/tasks/$slug` placeholder. */
const BY_SLUG = new Map<string, string>();

for (const category of taskCategories) {
  for (const t of category.tasks) {
    if (!BY_ROUTE.has(t.to)) BY_ROUTE.set(t.to, t.label);
    BY_SLUG.set(t.slug, t.label);
  }
}

/**
 * The tab title for a pathname. Unknown paths fall back to the bare product
 * name rather than inventing a label from the URL.
 */
export function titleForPath(pathname: string): string {
  // Tolerate a trailing slash on everything but the root.
  const path =
    pathname.length > 1 && pathname.endsWith("/")
      ? pathname.slice(0, -1)
      : pathname;

  const staticTitle = STATIC_TITLES[path];
  if (staticTitle !== undefined) return suffix(staticTitle);

  const byRoute = BY_ROUTE.get(path);
  if (byRoute) return suffix(byRoute);

  if (path.startsWith("/tasks/")) {
    const label = BY_SLUG.get(path.slice("/tasks/".length));
    if (label) return suffix(label);
  }

  return APP_NAME;
}
