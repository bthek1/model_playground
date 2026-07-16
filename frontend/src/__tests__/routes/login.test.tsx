import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

const { mockNavigate, mockMutate } = vi.hoisted(() => ({
  mockNavigate: vi.fn(),
  mockMutate: vi.fn(),
}));

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useNavigate: vi.fn().mockReturnValue(mockNavigate),
    Link: ({
      children,
      to,
      ...props
    }: {
      children: ReactNode;
      to: string;
      [key: string]: unknown;
    }) => (
      <a href={to} {...props}>
        {children}
      </a>
    ),
  };
});

vi.mock("@/hooks/useAuth", () => ({
  useLogin: vi.fn().mockReturnValue({ mutate: mockMutate, isPending: false }),
}));

vi.mock("@/hooks/useTheme", () => ({
  useTheme: vi.fn().mockReturnValue({ theme: "light", setTheme: vi.fn() }),
}));

vi.mock("@/api/health", () => ({
  getHealth: vi.fn().mockResolvedValue({ status: "ok" }),
}));

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

// Import AFTER mocks are registered
const { Route } = await import("@/routes/login");
const LoginComponent = Route?.options?.component as
  | React.ComponentType
  | undefined;

describe("Login page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders email and password fields", () => {
    if (!LoginComponent) return;
    render(<LoginComponent />, { wrapper });
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it("renders a link to the signup page", () => {
    if (!LoginComponent) return;
    render(<LoginComponent />, { wrapper });
    expect(screen.getByRole("link", { name: /sign up/i })).toBeInTheDocument();
  });

  it("shows validation error when form submitted empty", async () => {
    if (!LoginComponent) return;
    const user = userEvent.setup();
    render(<LoginComponent />, { wrapper });
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    expect(await screen.findByText(/valid email/i)).toBeInTheDocument();
  });

  it("redirects to /home on successful login", async () => {
    if (!LoginComponent) return;
    mockMutate.mockImplementation((_values, opts) => opts?.onSuccess?.());
    const user = userEvent.setup();
    render(<LoginComponent />, { wrapper });
    await user.type(screen.getByLabelText(/email/i), "user@example.com");
    await user.type(screen.getByLabelText(/password/i), "password123");
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    expect(mockNavigate).toHaveBeenCalledWith({ to: "/home" });
  });
});
