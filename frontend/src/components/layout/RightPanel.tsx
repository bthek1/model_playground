// The system panel's shell, and the keyboard shortcut that opens it.
//
// Three things are deliberate:
//
//  * It is a sibling of `<main>`, never inside it. The model-page grid
//    (model-page-pattern.md §4) is laid out on viewport breakpoints, and
//    wrapping it in another grid cell would leave its DOM order intact but its
//    columns computed against a width that no longer exists.
//
//  * Docked at `lg` and up, a Sheet below it. Squeezing a ~20rem panel out of a
//    tablet-width workbench is how RUN and OUTPUT end up stacked and the result
//    ends up below the fold, which is the exact failure the horizontal layout
//    exists to prevent.
//
//  * One of the two is rendered, never both. `useMediaQuery` decides in JS
//    rather than CSS because a hidden copy would still mount `SystemPanel` and
//    run a second sampling loop.

import { useEffect } from "react";
import { X } from "lucide-react";

import { SystemPanel } from "@/components/telemetry/SystemPanel";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useUIStore } from "@/store/ui";

/** Where the panel stops squeezing the workbench and becomes an overlay. */
const DOCKED_QUERY = "(min-width: 1024px)";

export const PANEL_SHORTCUT = "Alt+Shift+M";

/** Typing in a field must never toggle a panel, whatever the modifiers do. */
function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}

export function RightPanel() {
  const open = useUIStore((s) => s.rightPanelOpen);
  const setOpen = useUIStore((s) => s.setRightPanelOpen);
  const toggle = useUIStore((s) => s.toggleRightPanel);
  const docked = useMediaQuery(DOCKED_QUERY);

  // Registered here, the one component that is mounted whether the panel is
  // open or shut, so there is a single listener and a single teardown.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // `code`, not `key`: Alt+Shift+M produces a different character on
      // several layouts, and the physical key is what the shortcut names.
      if (!event.altKey || !event.shiftKey || event.code !== "KeyM") return;
      if (isEditable(event.target)) return;
      event.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [toggle]);

  if (!open) return null;

  if (!docked) {
    return (
      <Sheet open onOpenChange={setOpen}>
        <SheetContent
          side="right"
          data-testid="right-panel"
          className="w-full overflow-y-auto p-0 sm:max-w-sm"
        >
          <SheetHeader className="border-b">
            <SheetTitle>System</SheetTitle>
          </SheetHeader>
          <SystemPanel active />
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <aside
      data-testid="right-panel"
      aria-label="System panel"
      className="flex w-80 shrink-0 flex-col overflow-y-auto border-l bg-background"
    >
      <div className="flex h-10 shrink-0 items-center justify-between border-b px-3">
        <span className="text-sm font-medium">System</span>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setOpen(false)}
          aria-label="Close system panel"
        >
          <X className="size-4" />
        </Button>
      </div>
      <SystemPanel active />
    </aside>
  );
}
