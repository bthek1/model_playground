/**
 * Text contrast, computed from the colours that ship (#67): the theme tokens in
 * `src/index.css` and Tailwind's own palette in `tailwindcss/theme.css`, not
 * from copies of them. Lighthouse found `--muted-foreground` at 3.98:1 on a
 * `bg-destructive/10` status box and `text-amber-600` at 3.1:1 on white — both
 * under WCAG AA's 4.5:1 for body text, and both invisible to every other test.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

const SRC = resolve(__dirname, "..");
const CSS = readFileSync(join(SRC, "index.css"), "utf8");
const PALETTE = readFileSync(resolve(SRC, "../node_modules/tailwindcss/theme.css"), "utf8");

type Rgb = [number, number, number];

/** oklch → sRGB in [0, 1], gamut-clipped (Björn Ottosson's matrices). */
function oklch(l: number, c: number, h: number): Rgb {
  const a = c * Math.cos((h * Math.PI) / 180);
  const b = c * Math.sin((h * Math.PI) / 180);
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const lin = [
    4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
  ];
  return lin.map((x) => {
    const v = Math.min(Math.max(x, 0), 1);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055;
  }) as Rgb;
}

function parseOklch(value: string): Rgb {
  const m = value.match(/oklch\(\s*([\d.]+)(%?)\s+([\d.]+)\s+([\d.]+)\s*\)/);
  if (!m) throw new Error(`not an oklch() colour: ${value}`);
  return oklch(Number(m[1]) / (m[2] ? 100 : 1), Number(m[3]), Number(m[4]));
}

/** A token's value inside one block of index.css (`:root` or `.dark`). */
function token(block: ":root" | ".dark", name: string): Rgb {
  const start = CSS.indexOf(`${block} {`);
  const body = CSS.slice(start, CSS.indexOf("\n}", start));
  const m = body.match(new RegExp(`--${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`--${name} not found in ${block}`);
  return parseOklch(m[1]);
}

function palette(name: string): Rgb {
  const m = PALETTE.match(new RegExp(`--color-${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`--color-${name} not in Tailwind's palette`);
  return parseOklch(m[1]);
}

const mix = (fg: Rgb, bg: Rgb, alpha: number): Rgb =>
  fg.map((f, i) => f * alpha + bg[i] * (1 - alpha)) as Rgb;

function luminance(rgb: Rgb): number {
  const [r, g, b] = rgb.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: Rgb, b: Rgb): number {
  const [lo, hi] = [luminance(a), luminance(b)].sort((x, y) => x - y);
  return (hi + 0.05) / (lo + 0.05);
}

const AA = 4.5;

describe("theme contrast (WCAG AA, 4.5:1)", () => {
  // Calibrates the arithmetic against a number Lighthouse reported: the old
  // token, #737373, on bg-destructive/10 over white measured 3.98.
  it("reproduces Lighthouse's measurement of the old token", () => {
    const old = oklch(0.556, 0, 0);
    const tint = mix(token(":root", "destructive"), token(":root", "background"), 0.1);
    expect(contrast(old, tint)).toBeCloseTo(3.96, 1);
  });

  for (const theme of [":root", ".dark"] as const) {
    describe(theme === ":root" ? "light" : "dark", () => {
      const fg = token(theme, "muted-foreground");
      const surfaces: [string, Rgb][] = [
        ["background", token(theme, "background")],
        ["card", token(theme, "card")],
        ["muted", token(theme, "muted")],
      ];
      for (const [name, bg] of surfaces) {
        it(`muted-foreground on ${name}`, () => {
          expect(contrast(fg, bg)).toBeGreaterThanOrEqual(AA);
        });
      }
      // The status boxes: `bg-*/10` over the card, with muted body text.
      const statuses: [string, Rgb][] = [
        ["destructive", token(theme, "destructive")],
        ["emerald-500", palette("emerald-500")],
        ["amber-500", palette("amber-500")],
      ];
      for (const [name, colour] of statuses) {
        it(`muted-foreground on bg-${name}/10`, () => {
          const bg = mix(colour, token(theme, "card"), 0.1);
          expect(contrast(fg, bg)).toBeGreaterThanOrEqual(AA);
        });
      }
    });
  }

  // The light-mode status text colours, each with its dark: partner untouched.
  for (const name of ["amber-700", "emerald-700"]) {
    it(`text-${name} on white and on its own /10 tint`, () => {
      const white = token(":root", "background");
      expect(contrast(palette(name), white)).toBeGreaterThanOrEqual(AA);
      expect(contrast(palette(name), mix(palette(name), white, 0.1))).toBeGreaterThan(4.3);
    });
  }

  it("selected-row hint text on the primary button", () => {
    const primary = token(":root", "primary");
    const hint = mix(token(":root", "primary-foreground"), primary, 0.85);
    expect(contrast(hint, primary)).toBeGreaterThanOrEqual(AA);
  });
});

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [path] : [];
  });
}

describe("no failing text colour comes back", () => {
  // Each of these measured under 4.5:1 in light mode (#67). A `dark:` variant
  // is fine — the dark surfaces are dark — so only the bare class is refused.
  const REFUSED = [
    /(?<![\w:-])text-amber-600\b/,
    /(?<![\w:-])text-emerald-600\b/,
    /text-primary-foreground\/[1-7]\d\b/,
  ];
  it.each(REFUSED.map((re) => [re.source, re] as const))("%s", (_, re) => {
    const offenders = sourceFiles(SRC).filter((f) => re.test(readFileSync(f, "utf8")));
    expect(offenders.map((f) => f.slice(SRC.length + 1))).toEqual([]);
  });
});
