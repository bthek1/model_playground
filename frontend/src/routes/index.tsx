import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { HeroBanner } from "@/components/home/HeroBanner";
import { LegalFooter } from "@/components/layout/LegalFooter";
import { useMe } from "@/hooks/useAuth";
import { BACKEND_ENABLED } from "@/lib/features";

export const Route = createFileRoute("/")({
  // The landing page is a sign-in prompt. With no backend there is nothing to
  // sign in to, so go straight to the app — before `useMe` could run (#57).
  beforeLoad: () => {
    if (!BACKEND_ENABLED) throw redirect({ to: "/home", replace: true });
  },
  component: LandingPage,
});

function LandingPage() {
  const { data: me } = useMe();
  const navigate = useNavigate();

  // Redirect authenticated users to the main app
  useEffect(() => {
    if (me) {
      navigate({ to: "/home" });
    }
  }, [me, navigate]);

  return (
    <div className="relative min-h-screen bg-background">
      <div className="absolute right-4 top-4">
        <ThemeToggle />
      </div>
      <HeroBanner />
      <LegalFooter className="absolute inset-x-0 bottom-0" />
    </div>
  );
}
