/**
 * Renders the raster brand assets from `public/favicon.svg`.
 *
 * The SVG is the source of truth; everything else in `public/` that carries the
 * mark is generated from it by this script, so the assets can be regenerated
 * after a logo change instead of being opaque binaries of unknown provenance.
 *
 *     npm run icons          (or: just fe-icons)
 *
 * Rendering uses the Playwright Chromium that the E2E suite already installs —
 * no image toolchain to add, and no rasteriser dependency in package.json.
 * `favicon.ico` is a PNG wrapped in a single-entry ICO container, which every
 * browser from IE11 onward accepts.
 */

import { chromium } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const svg = readFileSync(resolve(root, "public/favicon.svg"), "utf8");

const BRAND = "#184be2";

/** Screenshot one HTML body at an exact pixel size. */
async function render(browser, { width, height, body, background = "#ffffff" }) {
  const page = await browser.newPage({ viewport: { width, height } });
  await page.setContent(
    `<body style="margin:0;width:${width}px;height:${height}px;background:${background};` +
      `font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif">${body}</body>`,
  );
  const buffer = await page.screenshot({ omitBackground: background === "transparent" });
  await page.close();
  return buffer;
}

/** Wrap a PNG in a single-image ICO container. */
function pngToIco(png, size) {
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(1, 4); // one image

  const entry = Buffer.alloc(16);
  entry.writeUInt8(size >= 256 ? 0 : size, 0); // 0 means 256
  entry.writeUInt8(size >= 256 ? 0 : size, 1);
  entry.writeUInt8(0, 2); // palette size
  entry.writeUInt8(0, 3); // reserved
  entry.writeUInt16LE(1, 4); // colour planes
  entry.writeUInt16LE(32, 6); // bits per pixel
  entry.writeUInt32LE(png.length, 8);
  entry.writeUInt32LE(header.length + entry.length, 12);

  return Buffer.concat([header, entry, png]);
}

const browser = await chromium.launch();
const written = [];

function write(name, buffer) {
  writeFileSync(resolve(root, "public", name), buffer);
  written.push(`${name} (${(buffer.length / 1024).toFixed(1)} KB)`);
}

// Square icons: the tile is already rounded, so it renders edge to edge.
for (const [name, size] of [
  ["apple-touch-icon.png", 180],
  ["icon-192.png", 192],
  ["icon-512.png", 512],
]) {
  write(
    name,
    await render(browser, {
      width: size,
      height: size,
      background: "transparent",
      body: `<div style="width:${size}px;height:${size}px">${svg}</div>`,
    }),
  );
}

// Maskable: Android crops to a circle inscribed in the central 80%, so the mark
// sits at 60% on a full-bleed ground. A rounded tile here would get its corners
// shaved off twice.
write(
  "icon-512-maskable.png",
  await render(browser, {
    width: 512,
    height: 512,
    background: BRAND,
    body:
      `<div style="width:512px;height:512px;display:flex;align-items:center;justify-content:center">` +
      `<div style="width:308px;height:308px">${svg.replace(
        /<rect[^>]*\/>/,
        "",
      )}</div></div>`,
  }),
);

// The social card (`og-image.png`) and each page's own card are rendered by
// `generate-og-cards.mjs` (#72), from the same strings as the page heads.

// Legacy tab icon.
const ico32 = await render(browser, {
  width: 32,
  height: 32,
  background: "transparent",
  body: `<div style="width:32px;height:32px">${svg}</div>`,
});
write("favicon.ico", pngToIco(ico32, 32));

await browser.close();
console.log("Generated from public/favicon.svg:\n  " + written.join("\n  "));
