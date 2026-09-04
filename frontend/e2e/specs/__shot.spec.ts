import { expect, test } from "../fixtures/base";

const SHOTS = "/tmp/claude-1000/-home-bthek1-model-playground/912d27e2-f04d-46cc-b90a-7888375e6cf9/scratchpad";

test("shots", async ({ page, mockApi }) => {
  await mockApi();
  await page.route((u) => u.hostname.endsWith("huggingface.co"), (r) => r.abort());
  for (const [name, w, h] of [["desktop", 1440, 900], ["md", 1000, 800], ["phone", 375, 812]] as const) {
    await page.setViewportSize({ width: w, height: h });
    for (const route of ["/text-to-speech", "/asr", "/tensor"]) {
      await page.goto(route);
      await expect(page.getByTestId("slot-4")).toBeVisible();
      await page.waitForTimeout(300);
      await page.screenshot({ path: `${SHOTS}/${name}${route.replace(/\//g, "-")}.png` });
    }
  }
});
