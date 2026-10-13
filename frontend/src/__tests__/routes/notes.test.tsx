// The notes (#72), rendered through the real route tree: the index, an article
// whose body arrives through the route's loader, an unknown slug, and the way
// back from a task page.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, createRouter, RouterProvider } from "@tanstack/react-router";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RelatedNotes } from "@/components/notes/RelatedNotes";
import { OPERATOR } from "@/legal/site";
import { NOTES } from "@/notes/meta";
import { routeTree } from "@/routeTree.gen";

vi.mock("@/lib/features", () => ({
  BACKEND_ENABLED: false,
  ANALYTICS_ENABLED: false,
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
  await screen.findByRole("heading", { level: 1 }, { timeout: 3000 });
  return router;
}

afterEach(cleanup);

describe("/notes", () => {
  it("links every note, with what it answers", async () => {
    await renderAt("/notes");
    const list = screen.getByTestId("notes-index");
    for (const n of NOTES) {
      expect(within(list).getByRole("link", { name: n.title })).toHaveAttribute("href", `/notes/${n.slug}`);
      expect(within(list).getByText(n.description)).toBeInTheDocument();
    }
  });
});

describe("/notes/$slug", () => {
  const note = NOTES.find((n) => n.slug === "webgpu-shader-f16")!;

  it("renders the article its loader fetched, with a byline and its date", async () => {
    await renderAt(`/notes/${note.slug}`);
    expect(screen.getByRole("heading", { level: 1, name: note.title })).toBeInTheDocument();
    expect(screen.getByTestId("note-byline")).toHaveTextContent(`By ${OPERATOR} · ${note.published}`);
    const body = screen.getByTestId("note-body");
    expect(body).toHaveTextContent("Program Gather requires f16 but the device does not support it");
    expect(body.querySelector("pre code")).toHaveTextContent('features.has("shader-f16")');
    // Inline markup is rendered, never shown.
    expect(body.textContent).not.toContain("`");
    expect(within(body).getByRole("link", { name: "image question answering" })).toHaveAttribute(
      "href",
      "/image-text-to-text",
    );
  });

  it("links the pages that demonstrate it", async () => {
    await renderAt(`/notes/${note.slug}`);
    const tryIt = screen.getByRole("heading", { name: "Try it" }).closest("section")!;
    expect(within(tryIt).getAllByRole("link").map((a) => a.getAttribute("href"))).toEqual([...note.related]);
  });

  it("answers an unknown slug with the not-found page, not an empty article", async () => {
    await renderAt("/notes/no-such-note");
    expect(screen.queryByTestId("note-body")).toBeNull();
  });
});

describe("the footer", () => {
  it("links the notes from every page", async () => {
    await renderAt("/notes");
    expect(within(screen.getByTestId("legal-footer")).getByRole("link", { name: "Notes" })).toHaveAttribute(
      "href",
      "/notes",
    );
  });
});

describe("RelatedNotes", () => {
  // Task-route tests mount the page with no router; the list is simply absent.
  it("renders nothing outside a router", () => {
    const { container } = render(<RelatedNotes />);
    expect(container).toBeEmptyDOMElement();
  });
});
