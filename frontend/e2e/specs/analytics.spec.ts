import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// Anonymous analytics (#60), against the shipped static bundle with a key in
// it. The fixture answers every /ingest request locally, so nothing here
// reaches PostHog; what it checks is what *would* have.
//
// The assertion that matters is the privacy one: a distinctive sentence typed
// into a model page and a CSV with distinctive column names, fitted for real,
// must appear in no decoded payload. Several pages promise "nothing is
// uploaded", and an analytics call is an upload.
//
// Needs a build with VITE_POSTHOG_KEY set: `just fe-e2e-static` builds with a
// throwaway key and sets E2E_ANALYTICS=1; CI sets it when the repository
// variable exists.
test.skip(
  !process.env.E2E_STATIC || !process.env.E2E_ANALYTICS,
  "static pass with an analytics build only (E2E_STATIC=1 E2E_ANALYTICS=1)",
);

// PostHog drops events from anything that looks like a bot — a `HeadlessChrome`
// user agent or brand, or `navigator.webdriver` — so under Playwright the SDK
// captures nothing and every assertion below would pass on an empty payload.
// Look like a browser for this file only. (The flip side is real: headless
// crawlers on the live site produce no events.)
test.use({
  userAgent:
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
});
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "webdriver", { get: () => false });
    Object.defineProperty(navigator, "userAgentData", { get: () => undefined });
  });
});

const SENTENCE = "Zanzibar quokkas audit my marmalade ledger";
const COLUMN = "zq_secret_salary_column";
const FILE = "zq-private-payroll.csv";

function csv(): string {
  const rows = [`${COLUMN},zq_other_feature,zq_target_label`];
  for (let i = 0; i < 80; i++) {
    const x = (i * 37) % 100;
    rows.push(`${x},${(i * 11) % 7},${x > 50 ? "yes" : "no"}`);
  }
  return rows.join("\n");
}

test.describe("analytics (#60)", () => {
  test("pageviews carry the route pattern, and stay on this origin", async ({ page, ingest }) => {
    const offOrigin: string[] = [];
    page.on("request", (req) => {
      if (/posthog\.com/.test(new URL(req.url()).hostname)) offOrigin.push(req.url());
    });

    const routes = () =>
      ingest.events.filter((e) => e.event === "$pageview").map((e) => e.properties.route);

    // Wait for each one: a full page load discards whatever the previous page
    // had not yet sent (the SDK is lazy and batches), and in-app navigation —
    // the real case — never reloads.
    await page.goto("/home");
    await expect.poll(routes, { timeout: 20_000 }).toContain("/home");
    await page.goto("/text-classification?text=zq-query-secret#zq-hash-secret");
    await expect.poll(routes, { timeout: 20_000 }).toContain("/text-classification");

    // In-app navigation: a pageview per route change, by pattern.
    await page.getByRole("link", { name: "Fill Mask" }).first().click();
    await expect.poll(routes, { timeout: 20_000 }).toContain("/fill-mask");

    const origin = new URL(page.url()).origin;
    for (const url of ingest.urls) expect(new URL(url).origin).toBe(origin);
    expect(offOrigin).toEqual([]);
    // The URL the SDK reports is origin + pattern: the query and hash never leave.
    expect(ingest.bodies.join("\n")).not.toMatch(/zq-query-secret|zq-hash-secret/);
  });

  test("a load emits the funnel, with a category for its failure", async ({ page, ingest }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-classification");
    await model.load();
    await expect
      .poll(() => ingest.events.map((e) => e.event), { timeout: 30_000 })
      .toEqual(expect.arrayContaining(["model_load_started", "model_load_failed"]));
    const failed = ingest.events.find((e) => e.event === "model_load_failed")!;
    expect(failed.properties).toMatchObject({ route: "/text-classification" });
    expect(typeof failed.properties.modelId).toBe("string");
    expect(typeof failed.properties.errorKind).toBe("string");
  });

  test("neither a typed sentence nor a fitted CSV appears in any payload", async ({
    page,
    ingest,
  }) => {
    test.setTimeout(120_000);
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/text-classification");
    await page.getByLabel("Text to classify").fill(SENTENCE);
    await model.load(); // fails on the blocked download — still an event

    await page.goto("/tabular-classification");
    await page.getByLabel("CSV file").setInputFiles({
      name: FILE,
      mimeType: "text/csv",
      buffer: Buffer.from(csv()),
    });
    await expect(page.getByTestId("dataset-summary")).toContainText("80 rows");
    await page.getByRole("button", { name: /random forest/i }).click();
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 90_000 });

    await expect
      .poll(() => ingest.events.map((e) => e.event), { timeout: 30_000 })
      .toEqual(expect.arrayContaining(["feature_used", "model_load_started"]));

    const everything = ingest.bodies.join("\n");
    expect(everything.length).toBeGreaterThan(0); // the probe is live
    for (const secret of [SENTENCE, "quokka", COLUMN, "zq_other_feature", "zq_target_label", FILE, "payroll"]) {
      expect(everything, `"${secret}" left the tab`).not.toContain(secret);
    }
    const fit = ingest.events.find((e) => e.event === "feature_used")!;
    expect(fit.properties).toMatchObject({ feature: "tabular_classification", family: expect.any(String) });
  });

  test("states what it collects, and the opt-out survives a reload", async ({ page, ingest }) => {
    await page.goto("/home");
    await expect(page.getByTestId("analytics-note")).toContainText(
      "Anonymous counts of which pages and models are used — never your inputs or results.",
    );
    const toggle = page.getByRole("switch", { name: /anonymous usage analytics/i });
    if (!(await toggle.isVisible())) {
      await page.getByRole("button", { name: "Toggle system panel" }).click();
    }
    await expect(toggle).toBeChecked();
    await toggle.click();
    await expect(toggle).not.toBeChecked();

    await page.reload();
    const before = ingest.events.length;
    await page.goto("/asr");
    await page.waitForTimeout(3_000);
    expect(ingest.events.length).toBe(before);
  });
});
