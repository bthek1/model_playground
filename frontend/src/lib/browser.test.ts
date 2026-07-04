import { afterEach, describe, expect, it, vi } from "vitest";

import { detectBrowser } from "./browser";

// detectBrowser reads navigator.userAgentData (Chromium) then falls back to
// navigator.userAgent. We stub the whole navigator per case; only the two
// fields the function touches need to be present.
afterEach(() => {
  vi.unstubAllGlobals();
});

function setNavigator(nav: object) {
  vi.stubGlobal("navigator", nav);
}

describe("detectBrowser — userAgentData (structured)", () => {
  it("prefers a specific brand over Chromium and skips the greasing entry", () => {
    setNavigator({
      userAgentData: {
        brands: [
          { brand: "Not/A)Brand", version: "99" },
          { brand: "Chromium", version: "149" },
          { brand: "Google Chrome", version: "149" },
        ],
      },
    });
    expect(detectBrowser()).toEqual({ name: "Google Chrome", version: "149" });
  });

  it("falls back to Chromium when no vendor-specific brand is present", () => {
    setNavigator({
      userAgentData: {
        brands: [
          { brand: "Not.A/Brand", version: "8" },
          { brand: "Chromium", version: "149" },
        ],
      },
    });
    expect(detectBrowser()).toEqual({ name: "Chromium", version: "149" });
  });

  it("ignores an empty brands array and parses the UA string instead", () => {
    setNavigator({
      userAgentData: { brands: [] },
      userAgent:
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    });
    expect(detectBrowser()).toEqual({ name: "Chrome", version: "149.0.0.0" });
  });
});

describe("detectBrowser — userAgent fallback", () => {
  const cases: Array<[string, string, string, string]> = [
    [
      "Edge (matched before Chrome)",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.91",
      "Edge",
      "120.0.2210.91",
    ],
    [
      "Edge on iOS",
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 EdgiOS/120.0.0.0 Mobile/15E148 Safari/604.1",
      "Edge",
      "120.0.0.0",
    ],
    [
      "Opera (matched before Chrome)",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
      "Opera",
      "106.0.0.0",
    ],
    [
      "Samsung Internet",
      "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
      "Samsung Internet",
      "23.0",
    ],
    [
      "Firefox on Linux",
      "Mozilla/5.0 (X11; Linux x86_64; rv:152.0) Gecko/20100101 Firefox/152.0",
      "Firefox",
      "152.0",
    ],
    [
      "Firefox on iOS",
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/125.0 Mobile/15E148 Safari/605.1.15",
      "Firefox",
      "125.0",
    ],
    [
      "Chrome (matched before Safari)",
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
      "Chrome",
      "149.0.0.0",
    ],
    [
      "Chrome on iOS",
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
      "Chrome",
      "120.0.6099.119",
    ],
    [
      "Safari (no Chrome token)",
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
      "Safari",
      "17.4",
    ],
  ];

  it.each(cases)("parses %s", (_label, ua, name, version) => {
    setNavigator({ userAgent: ua });
    expect(detectBrowser()).toEqual({ name, version });
  });

  it("returns Unknown when nothing matches", () => {
    setNavigator({ userAgent: "some random non-browser agent" });
    expect(detectBrowser()).toEqual({ name: "Unknown", version: "" });
  });

  it("returns Unknown when navigator has no useful fields", () => {
    setNavigator({});
    expect(detectBrowser()).toEqual({ name: "Unknown", version: "" });
  });
});
