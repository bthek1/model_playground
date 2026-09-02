import { TensorPage } from "../../pages/TensorPage";
import { expect, test } from "../../fixtures/base";

// Real GPU compute through the Web Worker. Skips wholesale when no GPU device
// is available, so a GPU-less runner stays green.
test.describe("WebGPU tensor compute", () => {
  test.beforeEach(({ webgpuStatus }) => {
    test.skip(
      webgpuStatus !== "ready",
      `No GPU device in this browser (status: ${webgpuStatus}).`,
    );
  });

  test("element-wise add matches a CPU reference", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const tensor = new TensorPage(page);
    await tensor.goto();

    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8, 9],
      [10, 11, 12],
    ];
    await tensor.selectOp("Add");
    await tensor.fillMatrixA(a.map((r) => r.join(" ")).join("\n"));
    await tensor.fillMatrixB(b.map((r) => r.join(" ")).join("\n"));
    await tensor.computeButton.click();

    // CPU reference — the repo rule is that every kernel is cross-checked.
    const expected = a.map((row, r) => row.map((v, c) => v + b[r][c]));
    await expect.poll(() => tensor.readResult()).toEqual(expected);
  });

  test("matmul matches a CPU reference", async ({ page, mockApi }) => {
    await mockApi();
    const tensor = new TensorPage(page);
    await tensor.goto();

    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const b = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    await tensor.selectOp("Matmul");
    await tensor.fillMatrixA(a.map((r) => r.join(" ")).join("\n"));
    await tensor.fillMatrixB(b.map((r) => r.join(" ")).join("\n"));
    await tensor.computeButton.click();

    const expected = a.map((row) =>
      b[0].map((_, j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0)),
    );
    await expect.poll(() => tensor.readResult()).toEqual(expected);
  });

  test("reports GPU time, proving the worker path ran", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const tensor = new TensorPage(page);
    await tensor.goto();
    await tensor.computeButton.click();

    await expect(page.getByText(/ms on GPU/)).toBeVisible();
  });

  test("a shape mismatch surfaces as an error, not a hang", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const tensor = new TensorPage(page);
    await tensor.goto();

    await tensor.selectOp("Add");
    await tensor.fillMatrixA("1 2 3");
    await tensor.fillMatrixB("1 2\n3 4");
    await tensor.computeButton.click();

    await expect(page.locator(".text-destructive").first()).toBeVisible();
    // The button returns to its idle state rather than spinning forever.
    await expect(tensor.computeButton).toBeEnabled();
  });
});
