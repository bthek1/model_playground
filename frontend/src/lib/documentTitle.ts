/**
 * Per-route document titles.
 *
 * Twenty-odd task routes shared one tab title before this, so a user with
 * several tasks open could not tell them apart. The names are *not* re-listed
 * here: an indexable page's title comes from `seo/pages.ts`, which builds it
 * from the sidebar taxonomy — the same string the build writes into that
 * route's own `index.html` (#64), so the tab and the search result agree.
 *
 * Pure on purpose — the hook that calls it is three lines, and the mapping is
 * what is worth testing.
 */

import { APP_NAME } from "@/lib/site";
import { normalisePath, pageForPath, suffixTitle } from "@/seo/pages";

/** Routes that are not indexable pages, so are not in `SITE_PAGES`. */
const UNLISTED_TITLES: Record<string, string> = {
  "/": APP_NAME,
  "/login": suffixTitle("Sign in"),
  "/signup": suffixTitle("Create account"),
};

/**
 * The tab title for a pathname. Unknown paths fall back to the bare product
 * name rather than inventing a label from the URL.
 */
export function titleForPath(pathname: string): string {
  const page = pageForPath(pathname);
  if (page) return page.title;
  return UNLISTED_TITLES[normalisePath(pathname)] ?? APP_NAME;
}
