/**
 * Renders the share cards (#72): one 1200×630 JPEG per indexable page under
 * `public/og/`, and the site-wide `public/og-image.png` for paths with none.
 *
 *     npm run og          (or: just fe-og)
 *
 * Run it after adding a page, or changing a page's heading or description —
 * `src/seo/render.test.ts` fails, naming the page, when a card is missing.
 *
 * A link to `/depth` and a link to `/translation` used to unfurl as the same
 * card in Slack, X, LinkedIn and Discord. Each now carries its own heading and
 * description, read from `SITE_PAGES` — the same strings as the page's title
 * and meta tags, so a card cannot describe a page differently from its head.
 *
 * Like `generate-icons.mjs`, it renders with the Playwright Chromium the E2E
 * suite already installs. `src/seo/` is TypeScript with extensionless imports,
 * so it is loaded through Vite's own module runner rather than a second TS
 * toolchain. JPEG at quality 82: a dark gradient with text is ~40 KB that way
 * and several times that as PNG, across fifty-odd files.
 */

import { chromium } from "@playwright/test";
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const svg = readFileSync(resolve(root, "public/favicon.svg"), "utf8");
const font = readFileSync(
  resolve(root, "node_modules/@fontsource-variable/geist/files/geist-latin-wght-normal.woff2"),
).toString("base64");

const INK = "#0a0a0a";

const vite = await createServer({
  root,
  configFile: false,
  logLevel: "error",
  server: { middlewareMode: true, hmr: false },
  appType: "custom",
});
const { SITE_PAGES } = await vite.ssrLoadModule("/src/seo/pages.ts");
const { ogImagePath, OG_IMAGE_PATH } = await vite.ssrLoadModule("/src/seo/render.ts");
const { APP_NAME, SITE_DESCRIPTION } = await vite.ssrLoadModule("/src/lib/site.ts");
const { OPERATOR } = await vite.ssrLoadModule("/src/legal/site.ts");
await vite.close();

const escape = (t) =>
  t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** One card: the brand row, a kicker, the heading, the description, a footer line. */
function card({ kicker, heading, description, footer }) {
  // Long headings step down a size rather than wrapping to three lines.
  const size = heading.length > 34 ? 60 : 72;
  return `
    <style>
      @font-face { font-family: Geist; src: url(data:font/woff2;base64,${font}) format("woff2"); font-weight: 100 900; }
      .clamp { display: -webkit-box; -webkit-box-orient: vertical; overflow: hidden; }
    </style>
    <div style="width:1200px;height:630px;box-sizing:border-box;padding:72px 88px;color:#fff;
                font-family:Geist,sans-serif;display:flex;flex-direction:column;gap:22px;
                background:radial-gradient(120% 120% at 100% 0%, #1e3a8a 0%, ${INK} 55%)">
      <div style="display:flex;align-items:center;gap:18px">
        <div style="width:56px;height:56px">${svg}</div>
        <div style="font-size:30px;font-weight:600;letter-spacing:-0.01em;color:#dbe4f7">${escape(APP_NAME)}</div>
      </div>
      <div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:18px">
        ${kicker ? `<div style="font-size:26px;font-weight:500;color:#7fa0ff;text-transform:uppercase;letter-spacing:0.08em">${escape(kicker)}</div>` : ""}
        <div class="clamp" style="-webkit-line-clamp:2;font-size:${size}px;line-height:1.08;font-weight:650;letter-spacing:-0.025em">${escape(heading)}</div>
        <div class="clamp" style="-webkit-line-clamp:3;font-size:31px;line-height:1.35;color:#c9d4ee;max-width:1000px">${escape(description)}</div>
      </div>
      <div style="font-size:24px;color:#7f92bb">${escape(footer)}</div>
    </div>`;
}

const IN_BROWSER = "Runs in your browser, on your own GPU · nothing is uploaded";

function cardFor(page) {
  switch (page.kind) {
    case "home":
      return { kicker: "", heading: "ML models in your browser", description: page.description, footer: IN_BROWSER };
    case "task":
      return { kicker: page.category, heading: page.heading, description: page.description, footer: IN_BROWSER };
    case "note":
      return { kicker: "Notes", heading: page.heading, description: page.description, footer: `By ${OPERATOR}` };
    default:
      return { kicker: "", heading: page.heading, description: page.description, footer: IN_BROWSER };
  }
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });

async function shoot(body, type) {
  await page.setContent(`<body style="margin:0">${body}</body>`);
  await page.evaluate(() => document.fonts.ready);
  return page.screenshot(type === "jpeg" ? { type: "jpeg", quality: 82 } : { type: "png" });
}

const outDir = resolve(root, "public/og");
// Rebuilt from scratch, so a removed page's card does not linger in the deploy.
rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

let total = 0;
for (const p of SITE_PAGES) {
  const buffer = await shoot(card(cardFor(p)), "jpeg");
  writeFileSync(resolve(root, "public", ogImagePath(p.path).slice(1)), buffer);
  total += buffer.length;
}

const site = await shoot(
  card({ kicker: "", heading: "ML models in your browser", description: SITE_DESCRIPTION, footer: IN_BROWSER }),
  "png",
);
writeFileSync(resolve(root, "public", OG_IMAGE_PATH.slice(1)), site);

await browser.close();
const count = readdirSync(outDir).length;
console.log(
  `Wrote ${count} page cards to public/og/ (${(total / 1024).toFixed(0)} KB, ` +
    `${(total / 1024 / count).toFixed(0)} KB each) and ${OG_IMAGE_PATH.slice(1)} (${(site.length / 1024).toFixed(0)} KB)`,
);
