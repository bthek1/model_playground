import { beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.hoisted(() => ({
  init: vi.fn(),
  capture: vi.fn(),
  opt_out_capturing: vi.fn(),
  opt_in_capturing: vi.fn(),
}));
vi.mock("posthog-js/dist/module.slim.no-external", () => ({ default: sdk }));

import { beforeSend, createClient, initOptions, SDK_PROPS } from "./client";

beforeEach(() => vi.clearAllMocks());

describe("init options (#60) — each one a decision", () => {
  const options = initOptions("/ingest");

  it("goes to the same-origin proxy", () => {
    expect(options.api_host).toBe("/ingest");
  });

  it("keeps nothing on disk and builds no person profiles", () => {
    expect(options.persistence).toBe("memory");
    expect(options.person_profiles).toBe("never");
  });

  it("captures nothing on its own — the app sends what it means to", () => {
    expect(options).toMatchObject({
      autocapture: false,
      rageclick: false,
      capture_dead_clicks: false,
      capture_exceptions: false,
      capture_heatmaps: false,
      capture_performance: false,
      capture_pageview: false,
      capture_pageleave: false,
    });
  });

  it("records no sessions and loads no remote features or scripts", () => {
    expect(options).toMatchObject({
      disable_session_recording: true,
      disable_surveys: true,
      disable_product_tours: true,
      disable_web_experiments: true,
      disable_external_dependency_loading: true,
      advanced_disable_flags: true,
    });
  });

  it("respects Do Not Track and masks as defence in depth", () => {
    expect(options).toMatchObject({
      respect_dnt: true,
      save_referrer: false,
      mask_all_text: true,
      mask_all_element_attributes: true,
      mask_personal_data_properties: true,
      before_send: beforeSend,
    });
  });

  it("createClient hands exactly these options to the SDK", () => {
    const client = createClient("phc_test", "/ingest");
    expect(sdk.init).toHaveBeenCalledWith("phc_test", initOptions("/ingest"));
    client.capture("model_load_ready", { modelId: "m" });
    expect(sdk.capture).toHaveBeenCalledWith("model_load_ready", { modelId: "m" });
    client.optOut();
    expect(sdk.opt_out_capturing).toHaveBeenCalled();
  });
});

describe("beforeSend — the last gate", () => {
  const event = (properties: Record<string, unknown>) =>
    beforeSend({
      uuid: "u",
      event: "$pageview",
      properties,
      $set: { $initial_current_url: "https://x/?q=secret" },
      $set_once: { $initial_referrer: "https://search?q=secret" },
    } as never)!;

  it("replaces the URL with origin + route pattern: no query, no hash", () => {
    const out = event({
      route: "/text-classification",
      $current_url: "https://site/text-classification?text=my+secret#frag",
      $pathname: "/text-classification",
    });
    expect(out.properties.$current_url).toBe(`${location.origin}/text-classification`);
    expect(out.properties.$pathname).toBe("/text-classification");
    expect(JSON.stringify(out)).not.toMatch(/secret|frag/);
  });

  it("drops referrers, $set and $set_once, and every unknown SDK property", () => {
    const out = event({
      route: "/asr",
      $referrer: "https://search?q=secret",
      $referring_domain: "search",
      $raw_user_agent: "UA",
      $some_future_property: "x",
      $browser: "Chrome",
      token: "phc_test",
    });
    expect(out.properties).toEqual({
      route: "/asr",
      $browser: "Chrome",
      token: "phc_test",
      $current_url: `${location.origin}/asr`,
      $pathname: "/asr",
    });
    expect(out.$set).toBeUndefined();
    expect(out.$set_once).toBeUndefined();
  });

  it("re-applies the app allowlist, so nothing a caller slipped past track() survives", () => {
    const out = event({ route: "/asr", text: "my secret", modelId: "a b c" });
    expect(out.properties).not.toHaveProperty("text");
    expect(out.properties).not.toHaveProperty("modelId");
  });

  it("keeps the identity the SDK needs and nothing URL-shaped", () => {
    expect(SDK_PROPS).toContain("token");
    expect(SDK_PROPS).toContain("distinct_id");
    for (const p of SDK_PROPS) expect(p).not.toMatch(/url|referr|title|host|pathname/);
  });

  it("passes a dropped event through as dropped", () => {
    expect(beforeSend(null)).toBeNull();
  });
});
