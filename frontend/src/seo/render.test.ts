import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { SITE_DESCRIPTION } from "@/lib/site";
import { pageForPath, SITE_PAGES } from "./pages";
import {
  absolutiseImages,
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

describe("structuredData", () => {
  it("describes home as a free web application", () => {
    const data = structuredData(pageForPath("/home")!, ORIGIN) as Record<string, unknown>;
    expect(data["@type"]).toBe("WebApplication");
    expect(data.url).toBe(`${ORIGIN}/home`);
    expect(data.offers).toEqual({ "@type": "Offer", price: "0", priceCurrency: "USD" });
  });

  it("gives a task a three-step breadcrumb: home, category, task", () => {
    const data = structuredData(pageForPath("/super-resolution")!, ORIGIN) as {
      itemListElement: { position: number; name: string; item: string }[];
    };
    expect(data.itemListElement).toEqual([
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
