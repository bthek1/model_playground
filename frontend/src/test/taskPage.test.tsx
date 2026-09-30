// The contract suite is only worth something if it fails on the pages it is
// meant to catch. A fixture page built from the real shell runs once correct —
// every check must pass — and once per injected fault, where the check aimed at
// that fault must throw.

import { cleanup, render } from "@testing-library/react";
import { Box } from "lucide-react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { ModelStatus } from "@/components/model/ModelStatus";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";

import {
  idleTask,
  loadErrorTask,
  loadingTask,
  readyTask,
  routeComponent,
  taskPageContractChecks,
  type TaskMachine,
} from "./taskPage";

type Fault =
  | "none"
  | "autoLoad"
  | "loadOnMount"
  | "loadErrorInOutput"
  | "runErrorDropped"
  | "triggerAlwaysOn"
  | "noEmptyOutput";

interface FixtureTask extends TaskMachine {
  result: string | null;
  run: (x: string) => Promise<string>;
}

let fault: Fault = "none";
let state: FixtureTask;
const hook = vi.fn<(model: string, autoLoad?: boolean) => FixtureTask>(
  () => state,
);

function FixturePage() {
  const t = fault === "autoLoad" ? hook("m", true) : hook("m");
  useEffect(() => {
    if (fault === "loadOnMount") t.load();
  }, [t]);
  const loadError = t.status === "error" ? t.error : null;
  const runError = t.status === "error" ? null : t.error;
  return (
    <ModelPage
      icon={Box}
      title="Fixture"
      description="A fixture page."
      select={<p>one model</p>}
      load={
        <ModelStatus
          status={t.status}
          backend={t.backend}
          loadProgress={t.loadProgress}
          error={loadError}
          onLoad={t.load}
          onRetry={t.retry}
          onCancel={t.cancel}
        />
      }
      run={
        <InputPanel
          ready={t.ready}
          controls={
            <Button disabled={fault === "triggerAlwaysOn" ? false : !t.ready}>
              Generate
            </Button>
          }
        >
          <p>input</p>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Result"
          running={t.running}
          error={
            fault === "loadErrorInOutput"
              ? t.error
              : fault === "runErrorDropped"
                ? loadError
                : runError
          }
          empty={fault === "noEmptyOutput" ? undefined : "Nothing yet."}
        >
          {fault === "noEmptyOutput" ? <p>stale</p> : t.result}
        </OutputPanel>
      }
    />
  );
}

const base = idleTask<FixtureTask>({
  result: null,
  run: vi.fn(async (x: string) => x),
});

const checks = taskPageContractChecks({
  render: () => render(<FixturePage />),
  hook,
  base,
  setState: (s) => {
    state = s;
  },
  trigger: /generate/i,
});

const check = (name: RegExp) => {
  const found = checks.find((c) => name.test(c.name));
  if (!found) throw new Error(`no contract check matches ${name}`);
  return found.check;
};

describe("describeTaskPageContract", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fault = "none";
    state = { ...base };
  });
  afterEach(cleanup);

  it("passes every check on a page that keeps the contract", () => {
    for (const c of checks) {
      state = { ...base };
      vi.clearAllMocks();
      expect(c.check, c.name).not.toThrow();
      cleanup();
    }
  });

  it.each([
    ["autoLoad", /loads nothing on arrival/],
    ["loadOnMount", /loads nothing on arrival/],
    ["loadErrorInOutput", /load failure lands in LOAD/],
    ["runErrorDropped", /run failure lands in OUTPUT/],
    ["triggerAlwaysOn", /GENERATE is disabled/],
    ["noEmptyOutput", /four slots/],
  ] as const)("fails a page with the %s fault", (f, name) => {
    fault = f;
    expect(check(name)).toThrow();
  });
});

describe("task state builders", () => {
  it("builds each machine state from one base, keeping its task fields", () => {
    expect(base).toMatchObject({ status: "idle", idle: true, ready: false });
    expect(readyTask(base, { result: "x" })).toMatchObject({
      status: "ready",
      idle: false,
      ready: true,
      result: "x",
      run: base.run,
    });
    expect(loadingTask(base)).toMatchObject({ status: "loading", loading: true });
    expect(loadErrorTask(base, "boom")).toMatchObject({
      status: "error",
      error: "boom",
      ready: false,
    });
  });

  it("finds the page a route registered, and refuses a route with none", () => {
    const Page = () => null;
    expect(routeComponent({ Route: { options: { component: Page } } })).toBe(
      Page,
    );
    expect(() => routeComponent({})).toThrow(/no component/);
  });
});
