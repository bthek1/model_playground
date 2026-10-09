import { useEffect } from "react";
import { useRouterState } from "@tanstack/react-router";

import { titleForPath } from "@/lib/documentTitle";
import { SITE_DESCRIPTION, SITE_URL } from "@/lib/site";
import { canonicalUrl, pageForPath } from "@/seo/pages";
import { scriptJson, structuredData } from "@/seo/render";

/**
 * The origin the build wrote its canonicals against. The build makes `og:image`
 * absolute in every HTML file it emits (`scripts/seoPages.ts`), so its origin
 * is the build's `SITE_URL` — read back here rather than threaded through a
 * second variable that could disagree with it.
 */
function buildOrigin(doc: Document): string {
  const image = doc.querySelector<HTMLMetaElement>('meta[property="og:image"]');
  try {
    return new URL(image?.content ?? "").origin;
  } catch {
    return SITE_URL;
  }
}

function setMeta(doc: Document, attr: "name" | "property", key: string, value: string) {
  let el = doc.head.querySelector<HTMLMetaElement>(`meta[${attr}="${key}"]`);
  if (!el) {
    el = doc.createElement("meta");
    el.setAttribute(attr, key);
    doc.head.appendChild(el);
  }
  el.content = value;
}

/**
 * Bring the head in line with `pathname`: the same title, description,
 * canonical and JSON-LD the build wrote into that route's own `index.html`
 * (#64). A visitor who lands on `/asr` and clicks through to `/depth` never
 * fetches `/depth/index.html`, so without this the head would keep describing
 * the page they arrived on. A path that is not an indexable page loses its
 * canonical and structured data rather than keeping another page's.
 */
export function syncHead(pathname: string, doc: Document = document): void {
  const page = pageForPath(pathname);
  const title = titleForPath(pathname);
  const description = page?.description ?? SITE_DESCRIPTION;
  const origin = buildOrigin(doc);

  doc.title = title;
  setMeta(doc, "name", "description", description);
  setMeta(doc, "property", "og:title", title);
  setMeta(doc, "property", "og:description", description);
  setMeta(doc, "name", "twitter:title", title);
  setMeta(doc, "name", "twitter:description", description);

  const canonical = doc.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  const ogUrl = doc.head.querySelector('meta[property="og:url"]');
  const data = doc.getElementById("structured-data");
  if (!page) {
    canonical?.remove();
    ogUrl?.remove();
    data?.remove();
    return;
  }

  const url = canonicalUrl(page.path, origin);
  if (canonical) {
    canonical.href = url;
  } else {
    const link = doc.createElement("link");
    link.rel = "canonical";
    link.href = url;
    doc.head.appendChild(link);
  }
  setMeta(doc, "property", "og:url", url);

  let script = data;
  if (!script) {
    script = doc.createElement("script");
    script.id = "structured-data";
    script.setAttribute("type", "application/ld+json");
    doc.head.appendChild(script);
  }
  script.textContent = scriptJson(structuredData(page, origin));
}

/**
 * Keeps the document head in step with the current route. Mounted once, in
 * the root route, so it covers every page including the ones that render
 * without the app shell.
 */
export function useDocumentHead() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    syncHead(pathname);
  }, [pathname]);
}
