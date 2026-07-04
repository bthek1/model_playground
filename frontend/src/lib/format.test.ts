import { describe, expect, it } from "vitest";

import { formatBytes, titleCase } from "./format";

describe("formatBytes", () => {
  it("keeps sub-KiB values in bytes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1023)).toBe("1023 B");
  });

  it("uses binary units and trims trailing zeros", () => {
    expect(formatBytes(1024)).toBe("1 KiB");
    expect(formatBytes(49152)).toBe("48 KiB");
    expect(formatBytes(1024 * 1024)).toBe("1 MiB");
    expect(formatBytes(1024 * 1024 * 1024)).toBe("1 GiB");
  });

  it("keeps up to two decimals for non-round values", () => {
    expect(formatBytes(1536)).toBe("1.5 KiB");
    expect(formatBytes(1024 * 1024 * 1.25)).toBe("1.25 MiB");
  });

  it("formats the real GPU limit values seen in the panel", () => {
    // maxBufferSize ≈ 4 GiB, maxStorageBufferBindingSize ≈ 2 GiB.
    expect(formatBytes(4294967292)).toBe("4 GiB");
    expect(formatBytes(2147483644)).toBe("2 GiB");
    // Firefox's more conservative 1 GiB default.
    expect(formatBytes(1073741824)).toBe("1 GiB");
  });

  it("returns non-finite or negative input stringified", () => {
    expect(formatBytes(Number.NaN)).toBe("NaN");
    expect(formatBytes(-1)).toBe("-1");
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBe("Infinity");
  });
});

describe("titleCase", () => {
  it("capitalises the first letter of each word", () => {
    expect(titleCase("turing")).toBe("Turing");
    expect(titleCase("apple m1")).toBe("Apple M1");
    expect(titleCase("")).toBe("");
  });
});
