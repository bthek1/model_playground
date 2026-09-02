import { disableWebGPU } from "../../fixtures/webgpu";
import { expect, test } from "../../fixtures/base";

test.describe("WebGPU capability reporting", () => {
  test("reports GPU access when a device can be acquired", async ({
    page,
    mockApi,
    webgpuStatus,
  }) => {
    test.skip(
      webgpuStatus !== "ready",
      `No GPU device in this browser (status: ${webgpuStatus}).`,
    );
    await mockApi();
    await page.goto("/home");

    await expect(page.getByText("GPU accessible")).toBeVisible();
    // The details block only renders on "ready".
    await expect(page.getByText("Adapter", { exact: true })).toBeVisible();
    await expect(page.getByText("Limits", { exact: true })).toBeVisible();
  });

  // Runs everywhere, GPU or not — this is the assertion that matters on CI.
  test("degrades gracefully when navigator.gpu is absent", async ({
    page,
    mockApi,
  }) => {
    await disableWebGPU(page);
    await mockApi();

    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto("/home");

    await expect(page.getByText("GPU Capabilities")).toBeVisible();
    // Either the plain unsupported label or the insecure-context variant.
    await expect(
      page.getByText(/WebGPU unsupported|WebGPU hidden|WebGPU disabled/),
    ).toBeVisible();
    // The app is still usable, not a blank crash screen.
    await expect(
      page.getByRole("heading", { name: "Model Playground" }),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });

  test("the tensor page still renders without WebGPU", async ({
    page,
    mockApi,
  }) => {
    await disableWebGPU(page);
    await mockApi();
    await page.goto("/tensor");

    await expect(
      page.getByRole("heading", { name: "Tensor Arithmetic" }),
    ).toBeVisible();
    // Scoped to <main>: the sidebar's "Computer Vision" button also matches.
    await expect(
      page.locator("main").getByRole("button", { name: "Compute", exact: true }),
    ).toBeVisible();
  });
});
