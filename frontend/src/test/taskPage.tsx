// Shared set-up for task-route tests (model-page-pattern.md §8).
//
// Every task route test used to hand-write the same three things: a full
// `ModelTask` object for the mocked hook (fifteen fields, most of them `null`),
// a `ready()` spread over it, and the `Route.options.component` lookup. They
// live here once. What stays in each route's own file is everything that
// depends on how *that* page takes its input and draws its result — above all
// "choosing an input runs nothing", which is the most valuable assertion a page
// has and cannot be written generically.
//
// `describeTaskPageContract` adds the checks that *are* identical on every
// task page. It is additive: call it inside the route's own `describe`, so the
// route's `beforeEach` resets the mocks for it too.

import { fireEvent, screen, within } from "@testing-library/react";
import type { ComponentType } from "react";
import { expect, it as vitestIt, vi, type Mock } from "vitest";

import type { ModelTask } from "@/model/types";

/**
 * The fields every task hook shares — the two state machines. `backend` is
 * widened to a string because the task hooks report it that way.
 */
export type TaskMachine = Pick<
  ModelTask<unknown, unknown>,
  | "status"
  | "idle"
  | "loading"
  | "ready"
  | "progress"
  | "loadProgress"
  | "loadedInMs"
  | "load"
  | "retry"
  | "cancel"
  | "running"
  | "error"
> & { backend: string | null };

/**
 * A hook result at `idle`, with fresh `load`/`retry`/`cancel` mocks. Pass the
 * task-specific fields (`run`, `result`, anything else the hook returns); any
 * machine field can be overridden too.
 */
export function idleTask<R extends TaskMachine>(
  rest: Omit<R, keyof TaskMachine> & Partial<TaskMachine>,
): R {
  return {
    status: "idle",
    idle: true,
    loading: false,
    ready: false,
    progress: null,
    loadProgress: null,
    loadedInMs: null,
    backend: null,
    running: false,
    error: null,
    load: vi.fn(),
    retry: vi.fn(),
    cancel: vi.fn(),
    ...rest,
  } as unknown as R;
}

/** The same result at `ready`. */
export function readyTask<R extends TaskMachine>(
  base: R,
  extra: Partial<R> = {},
): R {
  return {
    ...base,
    status: "ready",
    idle: false,
    ready: true,
    ...extra,
  };
}

/** The same result mid-download. */
export function loadingTask<R extends TaskMachine>(
  base: R,
  extra: Partial<R> = {},
): R {
  return {
    ...base,
    status: "loading",
    idle: false,
    loading: true,
    ...extra,
  };
}

/** The same result after a failed load. */
export function loadErrorTask<R extends TaskMachine>(
  base: R,
  error: string,
  extra: Partial<R> = {},
): R {
  return {
    ...base,
    status: "error",
    idle: false,
    error,
    ...extra,
  };
}

/**
 * The page component a route module registered. The real `createFileRoute`
 * keeps it on `Route.options.component`, so no router mock is needed.
 */
export function routeComponent(mod: {
  Route?: { options?: { component?: unknown } };
}): ComponentType {
  const Page = mod.Route?.options?.component as ComponentType | undefined;
  if (!Page) throw new Error("route module registered no component");
  return Page;
}

export interface TaskPageContract<R extends TaskMachine> {
  /** Render the page under test. */
  render: () => void;
  /** The mocked task hook, as the route calls it. */
  hook: Mock;
  /** The idle hook result — usually the file's `base`. */
  base: R;
  /** Point the mocked hook at a new result. */
  setState: (state: R) => void;
  /** Accessible name of the GENERATE trigger. */
  trigger: RegExp;
  /** Accessible name of the LOAD action. Defaults to "Load model". */
  loadButton?: RegExp;
}

/**
 * The §8 checks that read the same on every task page: nothing loads on
 * arrival, LOAD is the one thing that does, the four slots render with an empty
 * OUTPUT, GENERATE is shut until `ready`, and each error lands in the slot that
 * produced it.
 */
export function taskPageContractChecks<R extends TaskMachine>(
  c: TaskPageContract<R>,
): { name: string; check: () => void }[] {
  const loadName = c.loadButton ?? /load model/i;
  const checks: { name: string; check: () => void }[] = [];
  const it = (name: string, check: () => void) => checks.push({ name, check });

  it("loads nothing on arrival — no autoLoad, no load()", () => {
    c.render();
    expect(c.hook).toHaveBeenCalled();
    // `autoLoad` sits at a different position per hook, so look for it
    // anywhere: no call may pass a literal `true`.
    expect(c.hook.mock.calls.flat()).not.toContain(true);
    expect(c.base.load).not.toHaveBeenCalled();
  });

  it("LOAD fires load(), from slot 2", () => {
    c.render();
    fireEvent.click(
      within(screen.getByTestId("slot-2")).getByRole("button", {
        name: loadName,
      }),
    );
    expect(c.base.load).toHaveBeenCalledTimes(1);
  });

  it("renders four slots in order, OUTPUT empty", () => {
    c.render();
    const slots = [1, 2, 3, 4].map((n) => screen.getByTestId(`slot-${n}`));
    for (let i = 1; i < slots.length; i++) {
      expect(
        slots[i - 1].compareDocumentPosition(slots[i]) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("GENERATE is disabled until a model is ready", () => {
    c.render();
    expect(screen.getByRole("button", { name: c.trigger })).toBeDisabled();
  });

  it("a load failure lands in LOAD, not OUTPUT", () => {
    c.setState(loadErrorTask(c.base, "network down"));
    c.render();
    expect(
      within(screen.getByTestId("slot-2")).getByTestId("error-note"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("slot-4")).queryByTestId("error-note"),
    ).toBeNull();
  });

  it("a run failure lands in OUTPUT, and the model stays loaded", () => {
    c.setState(readyTask(c.base, { error: "inference blew up" } as Partial<R>));
    c.render();
    expect(
      within(screen.getByTestId("slot-4")).getByTestId("error-note"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("slot-2")).queryByTestId("error-note"),
    ).toBeNull();
    expect(screen.getByTestId("model-ready")).toBeInTheDocument();
  });
  return checks;
}

/** Register the contract as tests. Call inside the route's own `describe`. */
export function describeTaskPageContract<R extends TaskMachine>(
  c: TaskPageContract<R>,
) {
  for (const { name, check } of taskPageContractChecks(c)) {
    vitestIt(`[contract] ${name}`, check);
  }
}
