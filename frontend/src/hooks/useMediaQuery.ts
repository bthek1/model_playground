import { useEffect, useState } from "react";

/**
 * Subscribe to a CSS media query from JS.
 *
 * Tailwind's breakpoints handle presentation, so this is only for the cases
 * where the *component tree* has to differ rather than its styling — the system
 * panel is a docked `aside` on a wide screen and a Sheet on a narrow one, and
 * rendering both (hiding one with CSS) would mount two sampling loops.
 *
 * Returns `false` where `matchMedia` is missing rather than guessing.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const media = window.matchMedia(query);
    setMatches(media.matches);
    const onChange = (event: MediaQueryListEvent) => setMatches(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}
