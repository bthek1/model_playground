import type { Locator, Page } from "@playwright/test";

import { ModelPageObject } from "./ModelPage";

/**
 * /tensor — the reference WebGPU compute surface (raw WGSL in a Web Worker).
 * A four-slot model page like the audio routes, except its LOAD stage is a
 * device probe rather than a download (model-page-pattern.md §7).
 */
export class TensorPage extends ModelPageObject {
  readonly heading: Locator;
  readonly computeButton: Locator;
  readonly resultTable: Locator;

  constructor(page: Page) {
    super(page);
    this.heading = this.main.getByRole("heading", { name: "Tensor Arithmetic" });
    this.computeButton = this.main.getByRole("button", {
      name: /^(Compute|Computing…)$/,
    });
    this.resultTable = this.main.locator("table");
  }

  async goto(): Promise<void> {
    await this.page.goto("/tensor");
  }

  /** Pick an operation from the operation selector row. */
  async selectOp(label: string): Promise<void> {
    await this.main
      .getByRole("button", { name: label, exact: true })
      .first()
      .click();
  }

  /** Matrix A is the first textarea, Matrix B the second. */
  async fillMatrixA(text: string): Promise<void> {
    await this.main.locator("textarea").first().fill(text);
  }

  async fillMatrixB(text: string): Promise<void> {
    await this.main.locator("textarea").nth(1).fill(text);
  }

  /** Read the rendered result grid back as numbers, row-major. */
  async readResult(): Promise<number[][]> {
    await this.resultTable.waitFor();
    return this.resultTable.evaluate((table) =>
      Array.from(table.querySelectorAll("tr")).map((tr) =>
        Array.from(tr.querySelectorAll("td")).map((td) =>
          Number(td.textContent?.trim()),
        ),
      ),
    );
  }
}
