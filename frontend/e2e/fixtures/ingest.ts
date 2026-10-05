// Same-origin analytics, intercepted (#60). Every page answers `/ingest/**`
// itself — 200, empty body — and keeps a decoded copy of what was sent, so:
//
//  * no test ever reaches PostHog, even the static pass, whose bundle carries
//    CI's real project key;
//  * a spec can read exactly what would have left the tab.
//
// posthog-js compresses its batches: a raw gzip body (with or without
// `?compression=gzip-js`), or a base64 form field. Both are decoded here, because a privacy
// assertion that greps an undecoded payload passes vacuously.

import { gunzipSync } from "node:zlib";
import type { Page, Request } from "@playwright/test";

export interface IngestedEvent {
  event: string;
  properties: Record<string, unknown>;
}

export interface Ingest {
  /** Every decoded event so far. */
  events: IngestedEvent[];
  /** Every decoded body as text — what the privacy assertion greps. */
  bodies: string[];
  /** Absolute URLs of every /ingest request. */
  urls: string[];
}

export function decodeBody(request: Request): string {
  const raw = request.postDataBuffer();
  if (!raw) return "";
  const compression = new URL(request.url()).searchParams.get("compression");
  // Sniff the gzip magic number too: the SDK version in use sends gzip with
  // no `compression` parameter at all, and an undecoded body greps clean.
  const gzipped = raw.length > 2 && raw[0] === 0x1f && raw[1] === 0x8b;
  if (compression === "gzip-js" || gzipped) return gunzipSync(raw).toString("utf8");
  const text = raw.toString("utf8");
  if (compression === "base64" || text.startsWith("data=")) {
    const data = new URLSearchParams(text).get("data") ?? "";
    return Buffer.from(data, "base64").toString("utf8");
  }
  return text;
}

function parseEvents(body: string): IngestedEvent[] {
  try {
    const parsed = JSON.parse(body);
    const list = Array.isArray(parsed) ? parsed : (parsed.batch ?? [parsed]);
    return list.filter((e: unknown) => e && typeof (e as IngestedEvent).event === "string");
  } catch {
    return [];
  }
}

export async function interceptIngest(page: Page): Promise<Ingest> {
  const ingest: Ingest = { events: [], bodies: [], urls: [] };
  // A predicate, not a glob: a glob like `**/ingest/**` would also match a
  // dev-server module URL under /src/ (the `**/api/**` trap, e2e-testing.md).
  await page.route(
    (url) => url.pathname === "/ingest" || url.pathname.startsWith("/ingest/"),
    async (route) => {
      const request = route.request();
      ingest.urls.push(request.url());
      const body = decodeBody(request);
      ingest.bodies.push(body);
      ingest.events.push(...parseEvents(body));
      await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
    },
  );
  return ingest;
}
