import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { SITE_DESCRIPTION } from "@/lib/site";
import { HOME_LEAD, HOME_PRIVACY, pageForPath, SITE_PAGES } from "./pages";
import {
  absolutiseImages,
  renderNotFoundHtml,
  renderPageHtml,
  renderRobots,
  renderSitemap,
  scriptJson,
  structuredData,
} from "./render";

// The source template: the same tags the built one carries, minus the script.
const TEMPLATE = readFileSync(join(__dirname, "../../index.html"), "utf8");
const ORIGIN = "https://example.test";

/** Parse the rendered file as a browser would, so assertions read the DOM. */
function parse(html: string): Document {
  return new DOMParser().parseFromString(html, "text/html");
}
const meta = (doc: Document, sel: string) =>
  doc.querySelector<HTMLMetaElement>(`meta[${sel}]`)?.content;

describe("index.html", () => {
  it("carries the same description the app falls back to", () => {
    expect(meta(parse(TEMPLATE), 'name="description"')).toBe(SITE_DESCRIPTION);
  });
});

describe("renderPageHtml", () => {
  it.each(SITE_PAGES.map((p) => [p.path, p] as const))(
    "writes %s its own head and a static summary",
    (path, page) => {
      const doc = parse(renderPageHtml(TEMPLATE, page, ORIGIN));
      const url = `${ORIGIN}${path}`;
      expect(doc.title).toBe(page.title);
      expect(meta(doc, 'name="description"')).toBe(page.description);
      expect(meta(doc, 'property="og:title"')).toBe(page.title);
      expect(meta(doc, 'property="og:description"')).toBe(page.description);
      expect(meta(doc, 'property="og:url"')).toBe(url);
      expect(meta(doc, 'name="twitter:title"')).toBe(page.title);
      expect(meta(doc, 'property="og:image"')).toBe(`${ORIGIN}/og-image.png`);
      expect(meta(doc, 'name="twitter:image"')).toBe(`${ORIGIN}/og-image.png`);
      expect(doc.querySelectorAll('link[rel="canonical"]')).toHaveLength(1);
      expect(doc.querySelector<HTMLLinkElement>('link[rel="canonical"]')?.getAttribute("href")).toBe(url);
      expect(doc.querySelectorAll('meta[name="description"]')).toHaveLength(1);
      const h1s = doc.querySelectorAll("#root h1");
      expect(h1s).toHaveLength(1);
      expect(h1s[0].textContent).toBe(page.heading);
      const data = doc.getElementById("structured-data");
      expect(JSON.parse(data!.textContent!)).toEqual(structuredData(page, ORIGIN));
    },
  );

  it("links home to every task, under its category's anchor", () => {
    const doc = parse(renderPageHtml(TEMPLATE, pageForPath("/home")!, ORIGIN));
    const hrefs = [...doc.querySelectorAll("#root a")].map((a) => a.getAttribute("href"));
    for (const p of SITE_PAGES.filter((p) => p.kind === "task")) {
      expect(hrefs).toContain(p.path);
    }
    expect(doc.getElementById("computer-vision")?.querySelector("h2")?.textContent).toBe(
      "Computer Vision",
    );
  });

  // Home's paragraph is its LCP element (#67): the app's intro, whole, at
  // first paint — so React's two smaller paragraphs never replace it as LCP.
  it("paints home's intro as one paragraph holding both of the app's", () => {
    const doc = parse(renderPageHtml(TEMPLATE, pageForPath("/home")!, ORIGIN));
    const p = doc.querySelector("#root main > p");
    expect(p?.textContent).toBe(`${HOME_LEAD} ${HOME_PRIVACY} (privacy).`);
    expect(p?.querySelector("a")?.getAttribute("href")).toBe("/privacy");
  });

  // #71: the page's own words, the same strings its route renders, and its
  // siblings as bare links — their descriptions made each page mostly theirs.
  it("paints a task's tagline and about paragraphs, and its siblings as bare links", () => {
    const copy = {
      tagline: "Turn speech into text.",
      about: ["First paragraph.", "Second & last."],
      aliases: [],
    };
    const doc = parse(renderPageHtml(TEMPLATE, pageForPath("/asr")!, ORIGIN, copy));
    const root = doc.getElementById("root")!;
    expect(root.querySelector("h1 + p")?.textContent).toBe("Turn speech into text.");
    const paras = [...root.querySelectorAll("section p")].map((p) => p.textContent);
    expect(paras).toEqual(["First paragraph.", "Second & last."]);
    expect(root.textContent).not.toContain(pageForPath("/vad")!.description);
    expect(root.querySelector('a[href="/vad"]')?.textContent).toBe("Voice Activity Detection");
  });

  it("falls back to the description for a page with no copy", () => {
    const doc = parse(renderPageHtml(TEMPLATE, pageForPath("/asr")!, ORIGIN));
    expect(doc.querySelector("#root h1 + p")?.textContent).toBe(pageForPath("/asr")!.description);
  });

  it("links a task to home, its category and its siblings", () => {
    const doc = parse(renderPageHtml(TEMPLATE, pageForPath("/asr")!, ORIGIN));
    const hrefs = [...doc.querySelectorAll("#root a")].map((a) => a.getAttribute("href"));
    expect(hrefs).toEqual(expect.arrayContaining(["/home", "/home#audio", "/vad", "/text-to-speech"]));
    expect(hrefs).not.toContain("/asr");
  });

  it("escapes what it writes", () => {
    const page = { ...pageForPath("/asr")!, title: 'A <b>"bold"</b> & title' };
    const doc = parse(renderPageHtml(TEMPLATE, page, ORIGIN));
    expect(doc.title).toBe('A <b>"bold"</b> & title');
    expect(doc.querySelector("b")).toBeNull();
  });

  it("refuses a template it cannot fill, rather than shipping it unchanged", () => {
    expect(() =>
      renderPageHtml("<html><head></head><body></body></html>", pageForPath("/asr")!, ORIGIN),
    ).toThrow(/no <title>/);
  });
});

describe("renderNotFoundHtml", () => {
  // The edge serves this, with a 404, for a path with no page (#63).
  it("is the shell with a not-found title, noindex and no canonical", () => {
    const doc = parse(renderNotFoundHtml(TEMPLATE, ORIGIN));
    expect(doc.title).toBe("Page not found · Model Playground");
    expect(meta(doc, 'name="robots"')).toBe("noindex");
    expect(doc.querySelector('link[rel="canonical"]')).toBeNull();
    expect(doc.getElementById("structured-data")).toBeNull();
    expect(doc.querySelector("#root h1")?.textContent).toBe("Page not found");
    expect(doc.querySelector('#root a[href="/home"]')).not.toBeNull();
    expect(meta(doc, 'property="og:image"')).toBe(`${ORIGIN}/og-image.png`);
  });

  it("keeps the app's script, so the not-found page still mounts", () => {
    const withScript = TEMPLATE.replace("</head>", '<script type="module" src="/x.js"></script></head>');
    expect(renderNotFoundHtml(withScript, ORIGIN)).toContain('<script type="module" src="/x.js">');
  });
});

describe("structuredData", () => {
  it("describes home as a free web application", () => {
    const data = structuredData(pageForPath("/home")!, ORIGIN) as Record<string, unknown>;
    expect(data["@type"]).toBe("WebApplication");
    expect(data.url).toBe(`${ORIGIN}/home`);
    expect(data.offers).toEqual({ "@type": "Offer", price: "0", priceCurrency: "USD" });
  });

  it("gives a static page a breadcrumb only", () => {
    const data = structuredData(pageForPath("/privacy")!, ORIGIN) as Record<string, unknown>;
    expect(data["@type"]).toBe("BreadcrumbList");
  });

  // #71: the task page itself is the thing described, with its own title and
  // description, as part of the app — not only a position in a breadcrumb.
  it("describes a task as a WebPage that is part of the app", () => {
    const page = pageForPath("/depth")!;
    const data = structuredData(page, ORIGIN) as { "@graph": Record<string, unknown>[] };
    const [webPage] = data["@graph"];
    expect(webPage).toMatchObject({
      "@type": "WebPage",
      "@id": `${ORIGIN}/depth`,
      url: `${ORIGIN}/depth`,
      name: page.title,
      description: page.description,
      isPartOf: { "@type": "WebApplication", url: `${ORIGIN}/home` },
    });
  });

  it("gives a task a three-step breadcrumb: home, category, task", () => {
    const data = structuredData(pageForPath("/super-resolution")!, ORIGIN) as {
      "@graph": { breadcrumb: { itemListElement: unknown[] } }[];
    };
    expect(data["@graph"][0].breadcrumb.itemListElement).toEqual([
      { "@type": "ListItem", position: 1, name: "Model Playground", item: `${ORIGIN}/home` },
      { "@type": "ListItem", position: 2, name: "Computer Vision", item: `${ORIGIN}/home#computer-vision` },
      { "@type": "ListItem", position: 3, name: "Image to Image", item: `${ORIGIN}/super-resolution` },
    ]);
  });

  it("cannot close its own <script> tag", () => {
    expect(scriptJson({ x: "</script><script>alert(1)" })).not.toContain("</script>");
  });
});

describe("absolutiseImages", () => {
  it("makes the share image absolute, which the OG spec requires", () => {
    const doc = parse(absolutiseImages(TEMPLATE, ORIGIN));
    expect(meta(doc, 'property="og:image"')).toBe(`${ORIGIN}/og-image.png`);
    expect(meta(doc, 'name="twitter:image"')).toBe(`${ORIGIN}/og-image.png`);
  });
});

describe("renderSitemap / renderRobots", () => {
  it("lists every indexable page by absolute URL", () => {
    const xml = renderSitemap(SITE_PAGES, ORIGIN);
    const locs = [...xml.matchAll(/<loc>([^<]*)<\/loc>/g)].map((m) => m[1]);
    expect(locs).toEqual(SITE_PAGES.map((p) => `${ORIGIN}${p.path}`));
    expect(xml).toContain('xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"');
  });

  it("allows everything but the analytics relay, and names the sitemap", () => {
    const robots = renderRobots(ORIGIN);
    expect(robots).toMatch(/^User-agent: \*$/m);
    expect(robots).toMatch(/^Disallow: \/ingest\/$/m);
    expect(robots).toMatch(new RegExp(`^Sitemap: ${ORIGIN}/sitemap.xml$`, "m"));
    expect(robots).not.toMatch(/^Disallow: \/$/m);
  });
});
