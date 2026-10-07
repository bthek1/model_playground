// The legal pages (#62), rendered through the real route tree so the lazy
// route files, the app shell and its footer are all part of what is tested.
//
// The privacy notice is the page with logic: it must describe exactly what the
// *build* does, so it is rendered under each combination of the two flags that
// change what leaves the browser.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, createRouter, RouterProvider } from "@tanstack/react-router";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { taskCategories } from "@/components/layout/taskTaxonomy";
import { DATASETS } from "@/legal/datasets";
import { RECIPIENTS } from "@/legal/recipients";
import { CONTACT_EMAIL, LAST_UPDATED } from "@/legal/site";
import { STORED_ITEMS } from "@/legal/storage";
import { titleForPath } from "@/lib/documentTitle";
import { MODEL_LICENCES } from "@/model/licences";
import { routeTree } from "@/routeTree.gen";

const flags = vi.hoisted(() => ({ backend: false, analytics: false }));
vi.mock("@/lib/features", () => ({
  get BACKEND_ENABLED() {
    return flags.backend;
  },
  get ANALYTICS_ENABLED() {
    return flags.analytics;
  },
  POSTHOG_KEY: "",
  POSTHOG_HOST: "/ingest",
}));

async function renderAt(path: string) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  // Lazy route: the component arrives after its chunk resolves.
  await screen.findByRole("heading", { level: 1 }, { timeout: 3000 });
  return router;
}

afterEach(() => {
  cleanup();
  flags.backend = false;
  flags.analytics = false;
});

describe("/privacy", () => {
  it("names every service the static build's requests reach", async () => {
    await renderAt("/privacy");
    const table = screen.getByTestId("privacy-recipients");
    for (const r of RECIPIENTS.filter((x) => !x.analyticsOnly)) {
      expect(within(table).getByText(r.name), r.id).toBeInTheDocument();
    }
  });

  it("does not describe analytics in a build that sends none", async () => {
    // A sentence describing a request the build never makes is as untrue as
    // the reverse.
    await renderAt("/privacy");
    expect(screen.queryByText(/PostHog/)).toBeNull();
    expect(screen.getByTestId("privacy-analytics")).toHaveTextContent(/sends no analytics/i);
    expect(screen.queryByText("mp.analytics.optOut")).toBeNull();
  });

  it("names PostHog, what it receives and the opt-out when the build ships analytics", async () => {
    flags.analytics = true;
    await renderAt("/privacy");
    const table = screen.getByTestId("privacy-recipients");
    expect(within(table).getByText("PostHog")).toBeInTheDocument();
    expect(within(table).getByText(/configured to discard it/)).toBeInTheDocument();
    expect(screen.getByTestId("privacy-analytics")).toHaveTextContent(/never your inputs/i);
    expect(screen.getByTestId("privacy-analytics")).not.toHaveTextContent(/sends no analytics/i);
    expect(within(screen.getByTestId("privacy-storage")).getByText("mp.analytics.optOut")).toBeInTheDocument();
  });

  it("mentions accounts only in a build that has them", async () => {
    await renderAt("/privacy");
    expect(screen.queryByTestId("privacy-accounts")).toBeNull();
    expect(screen.queryByText("access_token")).toBeNull();
    cleanup();

    flags.backend = true;
    await renderAt("/privacy");
    expect(screen.getByTestId("privacy-accounts")).toHaveTextContent(/email address/);
    expect(within(screen.getByTestId("privacy-storage")).getByText("access_token")).toBeInTheDocument();
  });

  it("lists every item the static build stores on the device", async () => {
    await renderAt("/privacy");
    const table = screen.getByTestId("privacy-storage");
    for (const item of STORED_ITEMS.filter((s) => !s.backendOnly && !s.analyticsOnly)) {
      expect(within(table).getByText(item.key), item.key).toBeInTheDocument();
    }
  });

  it("states the contact, the revision date and a complaint route", async () => {
    await renderAt("/privacy");
    const contacts = screen.getAllByRole("link", { name: CONTACT_EMAIL });
    expect(contacts[0]).toHaveAttribute("href", `mailto:${CONTACT_EMAIL}`);
    expect(screen.getByTestId("legal-updated")).toHaveTextContent(LAST_UPDATED);
    expect(
      screen.getByRole("link", { name: /Office of the Australian Information Commissioner/ }),
    ).toHaveAttribute("href", expect.stringMatching(/^https:\/\/www\.oaic\.gov\.au\//));
  });
});

describe("/terms", () => {
  it("states that outputs are machine-made and not advice, and keeps consumer-law rights", async () => {
    await renderAt("/terms");
    expect(screen.getByTestId("terms-ai-output")).toHaveTextContent(/not professional advice/);
    expect(screen.getByText(/Australian Consumer Law/)).toBeInTheDocument();
    expect(screen.getByText(/as is/)).toBeInTheDocument();
  });
});

describe("/licences", () => {
  it("lists every licensed model repo, with its terms", async () => {
    await renderAt("/licences");
    const table = screen.getByTestId("licences-models");
    const rows = table.querySelectorAll("tbody tr[data-repo]");
    expect([...rows].map((r) => r.getAttribute("data-repo")).sort()).toEqual(Object.keys(MODEL_LICENCES).sort());
    const rmbg = table.querySelector('tr[data-repo="briaai/RMBG-1.4"]')!;
    expect(rmbg).toHaveAttribute("data-terms", "non-commercial");
    expect(rmbg).toHaveTextContent("Non-commercial");
  });

  it("lists every dataset, and the tabular and forecast samples", async () => {
    await renderAt("/licences");
    const table = screen.getByTestId("licences-datasets");
    for (const d of DATASETS) expect(table.querySelector(`tr[data-dataset="${d.id}"]`), d.id).not.toBeNull();
    expect(table.querySelector('tr[data-dataset="tabular-penguins"]')).toHaveTextContent("CC0");
    expect(table.querySelector('tr[data-dataset="forecast-airline"]')).not.toBeNull();
  });

  it("links the notices file the build writes", async () => {
    await renderAt("/licences");
    expect(screen.getByTestId("third-party-notices")).toHaveAttribute("href", "/THIRD-PARTY-NOTICES.txt");
  });
});

describe("/accessibility", () => {
  it("states its target and its known gaps", async () => {
    await renderAt("/accessibility");
    expect(screen.getByText(/WCAG\) 2\.2 at\s+level AA/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Known limitations" })).toBeInTheDocument();
  });
});

describe("the legal footer", () => {
  it("is on a task page, linking all four pages and the contact", async () => {
    await renderAt("/discrete-maths");
    const footer = screen.getByTestId("legal-footer");
    for (const name of ["Privacy", "Terms", "Licences", "Accessibility"]) {
      expect(within(footer).getByRole("link", { name }), name).toHaveAttribute(
        "href",
        `/${name.toLowerCase()}`,
      );
    }
    expect(within(footer).getByRole("link", { name: "Contact" })).toHaveAttribute(
      "href",
      `mailto:${CONTACT_EMAIL}`,
    );
  });
});

describe("legal routes are not tasks", () => {
  it("are absent from the sidebar taxonomy", () => {
    const routes = taskCategories.flatMap((c) => c.tasks.map((t) => t.to));
    for (const path of ["/privacy", "/terms", "/licences", "/accessibility"]) {
      expect(routes, path).not.toContain(path);
    }
  });

  it("have their own tab titles", () => {
    expect(titleForPath("/privacy")).toMatch(/^Privacy · /);
    expect(titleForPath("/terms")).toMatch(/^Terms of use · /);
    expect(titleForPath("/licences")).toMatch(/^Licences · /);
    expect(titleForPath("/accessibility")).toMatch(/^Accessibility · /);
  });
});
