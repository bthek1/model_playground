/**
 * The Model Playground wordmark and mark.
 *
 * One component, three callers (Navbar, Sidebar, HeroBanner). Before this the
 * product name was a string literal hand-typed in each of them, which is how the
 * collapsed sidebar ended up rendering "MP" where an icon belongs.
 *
 * The mark is three nodes joined into a right-pointing triangle — a graph and a
 * play button at once. Its geometry is duplicated in `public/favicon.svg`, which
 * is a standalone asset (browser tab, manifest, OG image) and cannot import from
 * `src/`; change one and change the other.
 */

import { cn } from "@/lib/utils";

/** The product name. Import it rather than retyping the string. */
export const APP_NAME = "Model Playground";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      aria-hidden="true"
      focusable="false"
      className={cn("h-7 w-7 shrink-0", className)}
    >
      <rect width="64" height="64" rx="14" className="fill-primary" />
      <g
        className="fill-primary-foreground stroke-primary-foreground"
        strokeWidth="4.2"
        strokeLinecap="round"
      >
        <path d="M21 16 L21 48 M21 16 L47 32 M21 48 L47 32" />
        <circle cx="21" cy="16" r="5.6" />
        <circle cx="21" cy="48" r="5.6" />
        <circle cx="47" cy="32" r="5.6" />
      </g>
    </svg>
  );
}

/**
 * Mark plus wordmark. The name is real text, not part of the SVG, so it stays
 * selectable and reachable by `getByText`.
 */
export function Logo({
  className,
  markClassName,
  textClassName,
}: {
  className?: string;
  markClassName?: string;
  textClassName?: string;
}) {
  return (
    <span className={cn("flex items-center gap-2", className)}>
      <LogoMark className={markClassName} />
      <span className={cn("font-semibold tracking-tight", textClassName)}>
        {APP_NAME}
      </span>
    </span>
  );
}
