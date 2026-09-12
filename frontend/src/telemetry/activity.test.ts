import { beforeEach, describe, expect, it } from "vitest";

import {
  activeDownload,
  consumeBusyMs,
  reportDownload,
  reportInflight,
  resetActivity,
} from "./activity";

beforeEach(() => resetActivity());

describe("busy accounting", () => {
  it("reports nothing busy before any run", () => {
    expect(consumeBusyMs(1000)).toBe(0);
  });

  it("measures a completed run", () => {
    reportInflight("asr", 1, 1000);
    reportInflight("asr", 0, 1400);
    expect(consumeBusyMs(2000)).toBe(400);
  });

  it("counts a still-running inference in every tick it spans", () => {
    reportInflight("asr", 1, 1000);
    expect(consumeBusyMs(2000)).toBe(1000);
    expect(consumeBusyMs(3000)).toBe(1000);
    reportInflight("asr", 0, 3500);
    expect(consumeBusyMs(4000)).toBe(500);
  });

  it("resets to zero once consumed", () => {
    reportInflight("asr", 1, 0);
    reportInflight("asr", 0, 100);
    expect(consumeBusyMs(200)).toBe(100);
    expect(consumeBusyMs(300)).toBe(0);
  });

  it("stays busy while any worker is busy — two models, one machine", () => {
    reportInflight("detector", 1, 0);
    reportInflight("pose", 1, 100);
    reportInflight("detector", 0, 200);
    // Still busy: `pose` has not returned.
    expect(consumeBusyMs(300)).toBe(300);
    reportInflight("pose", 0, 400);
    expect(consumeBusyMs(500)).toBe(100);
  });

  it("a teardown report of 0 closes the interval — no permanent 'busy'", () => {
    reportInflight("asr", 2, 0);
    reportInflight("asr", 0, 50); // route unmounted mid-run
    expect(consumeBusyMs(1000)).toBe(50);
    expect(consumeBusyMs(2000)).toBe(0);
  });
});

describe("download reporting", () => {
  it("is null when nothing is downloading", () => {
    expect(activeDownload()).toBeNull();
  });

  it("ignores a report with no known total — an indeterminate bar is not bytes", () => {
    reportDownload("tts", { loadedBytes: 10, totalBytes: 0 });
    expect(activeDownload()).toBeNull();
  });

  it("sums concurrent downloads", () => {
    reportDownload("detector", { loadedBytes: 10, totalBytes: 100 });
    reportDownload("pose", { loadedBytes: 20, totalBytes: 200 });
    expect(activeDownload()).toEqual({ loadedBytes: 30, totalBytes: 300 });
  });

  it("clears on completion", () => {
    reportDownload("tts", { loadedBytes: 10, totalBytes: 100 });
    reportDownload("tts", null);
    expect(activeDownload()).toBeNull();
  });
});
