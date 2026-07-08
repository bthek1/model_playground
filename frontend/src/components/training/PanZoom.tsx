// An interactive canvas surface: drag to pan, wheel to scroll, ⌘/Ctrl-wheel (or
// the buttons) to zoom, and "fit" to recenter. Used as the Training-page
// background so the architecture schematic behaves like a zoomable diagram
// instead of a plain scroll box.
//
// The content is wrapped in a single transformed layer (translate + scale,
// top-left origin); all interactions just mutate {scale, x, y}. Zoom is anchored
// to the cursor (or the viewport centre for the buttons) so the point under the
// pointer stays put. Purely presentational — it transforms whatever children it's
// given and never inspects them.

import { Maximize, Minus, Plus } from "lucide-react";
import {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import { Button } from "@/components/ui/button";

const MIN_SCALE = 0.2;
const MAX_SCALE = 5;

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

interface Transform {
  scale: number;
  x: number;
  y: number;
}

export function PanZoom({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const [t, setT] = useState<Transform>({ scale: 1, x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(
    null,
  );

  // Scale + centre the content so the whole diagram is visible.
  const fit = useCallback(() => {
    const c = containerRef.current;
    const el = contentRef.current;
    if (!c || !el) return;
    const cw = c.clientWidth;
    const ch = c.clientHeight;
    const ew = el.offsetWidth;
    const eh = el.offsetHeight;
    if (!ew || !eh || !cw || !ch) return;
    const scale = clamp(Math.min(cw / ew, ch / eh) * 0.92, MIN_SCALE, MAX_SCALE);
    setT({ scale, x: (cw - ew * scale) / 2, y: (ch - eh * scale) / 2 });
  }, []);

  useLayoutEffect(() => {
    fit();
  }, [fit]);

  // Zoom anchored to a point (in container-local coords).
  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setT((p) => {
      const scale = clamp(p.scale * factor, MIN_SCALE, MAX_SCALE);
      const k = scale / p.scale;
      return { scale, x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k };
    });
  }, []);

  // Native, non-passive wheel: scroll to pan, ⌘/Ctrl-wheel to zoom.
  useEffect(() => {
    const c = containerRef.current;
    if (!c) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = c.getBoundingClientRect();
        zoomAt(
          e.deltaY < 0 ? 1.1 : 1 / 1.1,
          e.clientX - rect.left,
          e.clientY - rect.top,
        );
      } else {
        setT((p) => ({ ...p, x: p.x - e.deltaX, y: p.y - e.deltaY }));
      }
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    drag.current = { px: e.clientX, py: e.clientY, x: t.x, y: t.y };
    setDragging(true);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    setT((p) => ({
      ...p,
      x: d.x + (e.clientX - d.px),
      y: d.y + (e.clientY - d.py),
    }));
  };
  const endDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    drag.current = null;
    setDragging(false);
    try {
      e.currentTarget.releasePointerCapture?.(e.pointerId);
    } catch {
      /* pointer already released */
    }
  };

  const zoomButton = (factor: number) => {
    const c = containerRef.current;
    if (!c) return;
    zoomAt(factor, c.clientWidth / 2, c.clientHeight / 2);
  };

  return (
    <div
      ref={containerRef}
      className={`overflow-hidden touch-none ${className ?? ""}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
      onDoubleClick={fit}
      style={{ cursor: dragging ? "grabbing" : "grab" }}
    >
      <div
        ref={contentRef}
        className="absolute top-0 left-0 origin-top-left select-none will-change-transform"
        style={{ transform: `translate(${t.x}px, ${t.y}px) scale(${t.scale})` }}
      >
        {children}
      </div>

      {/* Zoom controls — don't let clicks/drags on them pan the canvas. */}
      <div
        className="absolute bottom-4 left-4 flex flex-col gap-1"
        onPointerDown={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
      >
        <Button
          variant="outline"
          size="icon-sm"
          className="bg-card/70 backdrop-blur-md"
          aria-label="Zoom in"
          onClick={() => zoomButton(1.2)}
        >
          <Plus className="size-4" />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          className="bg-card/70 backdrop-blur-md"
          aria-label="Zoom out"
          onClick={() => zoomButton(1 / 1.2)}
        >
          <Minus className="size-4" />
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          className="bg-card/70 backdrop-blur-md"
          aria-label="Fit to view"
          onClick={fit}
        >
          <Maximize className="size-4" />
        </Button>
        <span className="mt-0.5 text-center font-mono text-[10px] text-muted-foreground">
          {Math.round(t.scale * 100)}%
        </span>
      </div>
    </div>
  );
}
