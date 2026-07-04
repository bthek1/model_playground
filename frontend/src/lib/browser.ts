// Best-effort browser name + version detection for display in the UI.
// Prefers the structured `navigator.userAgentData` (Chromium) and falls back to
// parsing the classic user-agent string. Order matters: Edge/Opera/Brave all
// embed "Chrome" in their UA, so they must be matched before plain Chrome.

export interface BrowserInfo {
  name: string;
  version: string;
}

const UNKNOWN: BrowserInfo = { name: "Unknown", version: "" };

// Ordered [displayName, regex-capturing-version]. First match wins.
const UA_RULES: Array<[string, RegExp]> = [
  ["Edge", /Edg(?:e|A|iOS)?\/([\d.]+)/],
  ["Opera", /OPR\/([\d.]+)/],
  ["Samsung Internet", /SamsungBrowser\/([\d.]+)/],
  ["Firefox", /(?:Firefox|FxiOS)\/([\d.]+)/],
  // Chrome must come before Safari (Chrome's UA also contains "Safari").
  ["Chrome", /(?:Chrome|CriOS)\/([\d.]+)/],
  ["Safari", /Version\/([\d.]+).*Safari/],
];

interface UAData {
  brands?: Array<{ brand: string; version: string }>;
}

function fromUserAgentData(): BrowserInfo | null {
  const uaData = (navigator as Navigator & { userAgentData?: UAData })
    .userAgentData;
  const brands = uaData?.brands;
  if (!brands?.length) return null;

  // Ignore the intentionally-random "Not(A:Brand" greasing entry.
  const real = brands.filter((b) => !/not.a.brand/i.test(b.brand));
  // Prefer a specific brand over the generic "Chromium" umbrella.
  const preferred =
    real.find((b) => !/chromium/i.test(b.brand)) ?? real[0] ?? null;
  if (!preferred) return null;
  return { name: preferred.brand, version: preferred.version };
}

export function detectBrowser(): BrowserInfo {
  if (typeof navigator === "undefined") return UNKNOWN;

  const structured = fromUserAgentData();
  if (structured) return structured;

  const ua = navigator.userAgent ?? "";
  for (const [name, re] of UA_RULES) {
    const match = ua.match(re);
    if (match) return { name, version: match[1] };
  }
  return UNKNOWN;
}
