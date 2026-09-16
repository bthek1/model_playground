import { expect, test } from "../../fixtures/base";

// The /graph-classification route, end to end in a real browser.
//
// The assertion that matters is **above the majority baseline**, not above zero
// and not above chance. PROTEINS is 663 enzymes to 450 non-enzymes, so a model
// that ignores the molecule entirely scores 0.598 — and that is precisely what a
// broken readout, a mis-built disjoint union or a dead gradient produces, because
// the class prior is the easiest thing in the dataset to learn. "An accuracy
// appeared" and even "an accuracy above 0.5" would both pass with the page
// completely broken.
//
// This is also the one route in the repo that downloads a dataset, so the fast
// spec checks that nothing leaves the machine until the button is pressed.

test.describe("Graph classification", () => {
  test("downloads nothing until asked, and quotes the size first", async ({
    page,
    mockApi,
  }) => {
    await mockApi();

    const hub: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (url.includes("huggingface.co")) hub.push(url);
      return route.continue();
    });

    await page.goto("/graph-classification");
    await expect(
      page.getByRole("heading", { name: /graph classification/i }),
    ).toBeVisible();

    for (const step of [1, 2, 3, 4]) {
      await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    }
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByRole("button", { name: /^train$/i })).toBeDisabled();

    // The size is on screen before the click that spends it.
    await expect(page.getByText(/2\.0 MB from the Hugging Face Hub/i)).toBeVisible();
    expect(hub).toEqual([]);
  });

  test("@slow loads 1113 proteins and beats the majority baseline", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(600_000);
    await mockApi();
    await page.goto("/graph-classification");

    await page.getByRole("button", { name: /load dataset/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.getByTestId("model-ready")).toContainText("1113 graphs");
    await expect(page.getByTestId("model-ready")).toContainText("43471 nodes");

    await page.getByRole("button", { name: /^train$/i }).click();
    await expect(page.getByTestId("protein-gallery")).toBeVisible({
      timeout: 480_000,
    });
    await expect(page.getByRole("button", { name: /^train$/i })).toBeEnabled({
      timeout: 540_000,
    });

    const read = async (label: string) =>
      Number.parseFloat(
        (
          await page
            .getByText(label, { exact: true })
            .locator("xpath=following-sibling::dd[1]")
            .innerText()
        ).replace("%", ""),
      );

    const accuracy = await read("Test accuracy");
    const baseline = await read("Majority baseline");

    // 59.8 on this split, and the page computes it from the same split it scores
    // the model on — so this is the real null model, not a remembered constant.
    expect(baseline).toBeGreaterThan(55);
    expect(baseline).toBeLessThan(65);

    // The assertion. Published GNNs reach 73-76 on PROTEINS; anything at or below
    // the baseline means the model learned the class prior and nothing else.
    expect(accuracy).toBeGreaterThan(baseline + 5);

    // …and the page must be saying so, not merely be right.
    await expect(page.getByTestId("baseline-note")).toContainText(
      /points above the baseline/i,
    );
  });
});
