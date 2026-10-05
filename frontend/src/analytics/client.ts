// The ONLY module that imports posthog-js (#60), and only ever by dynamic
// `import()` from `./index.ts` — one static import anywhere else folds ~50 KB
// gzip into the entry chunk (CLAUDE.md's INEFFECTIVE_DYNAMIC_IMPORT rule), and
// `conventions.test.ts` fails naming the file.
//
// The build is `module.slim.no-external`: the core SDK with no extensions and
// **no external-script loader**, so it cannot fetch session-recording,
// surveys or toolbar code from PostHog's asset host however the project is
// configured server-side. 51 KB gzip, against 98 KB for the default build.

import posthog from "posthog-js/dist/module.slim.no-external";
import type {
  CaptureResult,
  PostHogConfig,
} from "posthog-js/dist/module.slim.no-external";

import { sanitize } from "./schema";

/**
 * SDK-added properties that are allowed through `before_send`. Everything
 * else the SDK attaches is dropped — notably `$current_url`'s query string and
 * hash, `$referrer`, `$initial_*` and `$set_once`, any of which can carry a URL
 * the user typed or followed. A strict list rather than a denylist, so a
 * property a future SDK version adds is dropped until someone reads it.
 */
export const SDK_PROPS = [
  "token",
  "distinct_id",
  "$device_id",
  "$session_id",
  "$window_id",
  "$pageview_id",
  "$insert_id",
  "$time",
  "$lib",
  "$lib_version",
  "$browser",
  "$browser_version",
  "$os",
  "$os_version",
  "$device_type",
  "$screen_height",
  "$screen_width",
  "$viewport_height",
  "$viewport_width",
  "$timezone",
  "$browser_language",
  "$process_person_profile",
  "$is_identified",
] as const;

const sdkAllowed = new Set<string>(SDK_PROPS);

/**
 * The final gate. Rebuilds the property bag from the two allowlists, and
 * replaces the URL with origin + route *pattern* (the app's own `route`
 * property), so neither a query string nor a hash can leave.
 */
export function beforeSend(event: CaptureResult | null): CaptureResult | null {
  if (!event) return null;
  const source = (event.properties ?? {}) as Record<string, unknown>;
  const properties: Record<string, unknown> = {};
  for (const key of Object.keys(source)) {
    if (sdkAllowed.has(key)) properties[key] = source[key];
  }
  Object.assign(properties, sanitize(source));
  const route = typeof properties.route === "string" ? properties.route : "/";
  const origin = typeof location !== "undefined" ? location.origin : "";
  properties.$current_url = origin + route;
  properties.$pathname = route;
  return {
    ...event,
    properties: properties as CaptureResult["properties"],
    $set: undefined,
    $set_once: undefined,
  };
}

/** Every option is a decision; `client.test.ts` pins them one by one. */
export function initOptions(host: string): Partial<PostHogConfig> {
  return {
    api_host: host,
    // Nothing on disk: no cookie, no localStorage, no consent banner. A
    // returning visitor counts as new — accepted (#60, Phase 0).
    persistence: "memory",
    person_profiles: "never",
    // Autocapture records element text, which here can be the user's sentence.
    autocapture: false,
    rageclick: false,
    capture_dead_clicks: false,
    capture_heatmaps: false,
    capture_performance: false,
    capture_pageview: false, // the router sends route *patterns*
    capture_pageleave: false,
    disable_surveys: true,
    disable_product_tours: true,
    disable_conversations: true,
    disable_web_experiments: true,
    disable_external_dependency_loading: true,
    // No /flags round-trip: remote config could otherwise switch a capture
    // feature on from the PostHog UI without a code change here.
    advanced_disable_flags: true,
    save_referrer: false,
    respect_dnt: true,
    mask_all_text: true,
    mask_all_element_attributes: true,
    mask_personal_data_properties: true,
    before_send: beforeSend,
  };
}

export interface Client {
  capture(event: string, properties: Record<string, unknown>): void;
  optOut(): void;
  optIn(): void;
}

export function createClient(key: string, host: string): Client {
  posthog.init(key, initOptions(host));
  return {
    capture: (event, properties) => {
      posthog.capture(event, properties);
    },
    optOut: () => posthog.opt_out_capturing(),
    optIn: () => posthog.opt_in_capturing(),
  };
}
