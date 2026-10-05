// The analytics facade (#60). Everything outside `src/analytics/` calls
// `track` / `pageview` and nothing else.
//
// Disabled — no key in the build, the user opted out, or no `window` (a Worker
// realm) — every call returns at once and **nothing is imported**: the SDK
// chunk is never even requested, which is what keeps every "no outbound
// request" assertion in the test suites meaning what it says. Enabled, the
// first call schedules the SDK for idle time and queues until it resolves.

import { ANALYTICS_ENABLED, POSTHOG_HOST, POSTHOG_KEY } from "@/lib/features";

import type { Client } from "./client";
import { sanitize, type AnalyticsEvent, type Props } from "./schema";

export { classifyError } from "./schema";
export type { AnalyticsEvent, ErrorKind, Props } from "./schema";

/** Its own key: `persistence: "memory"` forgets an SDK-level opt-out on reload. */
export const OPT_OUT_KEY = "mp.analytics.optOut";
const MAX_QUEUE = 100;

let client: Client | null = null;
let loading = false;
let failed = false;
let queue: Array<[AnalyticsEvent, Record<string, unknown>]> = [];
let route: string | null = null;

/** Whether this build ships analytics at all — the opt-out switch's gate. */
export const analyticsAvailable = (): boolean =>
  ANALYTICS_ENABLED && typeof window !== "undefined";

export function isOptedOut(): boolean {
  try {
    return localStorage.getItem(OPT_OUT_KEY) === "1";
  } catch {
    return false;
  }
}

export function setOptedOut(out: boolean): void {
  try {
    if (out) localStorage.setItem(OPT_OUT_KEY, "1");
    else localStorage.removeItem(OPT_OUT_KEY);
  } catch {
    // A blocked storage still honours the choice for this page's lifetime
    // through the client below; it just will not survive a reload.
  }
  if (out) {
    queue = [];
    client?.optOut();
  } else {
    client?.optIn();
  }
}

const active = (): boolean => analyticsAvailable() && !failed && !isOptedOut();

/**
 * `webgpu_status`, once per session. Adapter-level on purpose: acquiring a
 * device to report `ready` would cost a GPU device on every visit, for a number
 * the model events already carry (`backend` on `model_load_ready`). This is the
 * same `requestAdapter()` `useBackendProbe` makes on every model page.
 */
export async function probeWebGPU(): Promise<Props> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { webgpuStatus: "unsupported" };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { webgpuStatus: "no-adapter" };
    return { webgpuStatus: "adapter", shaderF16: adapter.features.has("shader-f16") };
  } catch {
    return { webgpuStatus: "no-adapter" };
  }
}

function load(): void {
  if (loading) return;
  loading = true;
  const start = () => {
    import("./client")
      .then(({ createClient }) => {
        client = createClient(POSTHOG_KEY, POSTHOG_HOST);
        if (isOptedOut()) client.optOut();
        const pending = queue;
        queue = [];
        if (!isOptedOut()) for (const [e, p] of pending) client.capture(e, p);
        return probeWebGPU().then((props) => track("webgpu_status", props));
      })
      .catch(() => {
        // An ad-blocked chunk or an offline tab: stop trying, drop the queue,
        // and never surface it — analytics must not be a way for a page to fail.
        failed = true;
        queue = [];
      });
  };
  if (typeof requestIdleCallback === "function") requestIdleCallback(start, { timeout: 4000 });
  else setTimeout(start, 1);
}

/** Record that the app did something. Props pass the allowlist or are dropped. */
export function track(event: AnalyticsEvent, props: Props = {}): void {
  if (!active()) return;
  const properties = sanitize({ ...props, route: route ?? undefined });
  if (client) {
    client.capture(event, properties);
    return;
  }
  if (queue.length < MAX_QUEUE) queue.push([event, properties]);
  load();
}

/** A navigation, by route *pattern* (`/text-classification`), never the URL. */
export function pageview(pattern: string): void {
  route = pattern;
  track("$pageview", {});
}

/** Test seam: forget the module's state between cases. */
export function __resetAnalyticsForTests(): void {
  client = null;
  loading = false;
  failed = false;
  queue = [];
  route = null;
}
