// The Q-table drawn over the grid: an arrow per cell at the argmax action,
// the cell shaded by its value. §3.1 of the roadmap's whole argument — the
// arrows turn around on screen while the agent learns, which a notebook can
// only show as a plot after the fact.
//
// Repainted from the worker's render state, which arrives at most at 60 Hz
// however fast the loop runs. The geometry is pulled out as pure functions
// because happy-dom gives a canvas no 2D context: the painting cannot be
// asserted in a unit test, but where each cell lands can
// (`GraphCanvas.layoutToPixels` is the precedent, and doing it there found a
// real centring bug).

import { useEffect, useRef, useState } from "react";

import { MAPS } from "@/rl/envs/gridWorld";

import { ACTION_VECTORS, cellArrow, cellOrigin, gridGeometry } from "./gridGeometry";
import type { GridMapId } from "@/rl/types";

export function GridCanvas({
  map,
  q,
  agent,
}: {
  map: GridMapId;
  q: Float32Array;
  agent: number;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry.contentRect;
      setSize({ width: Math.round(box.width), height: Math.round(box.height) });
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const { width, height } = size;
    if (!canvas || width === 0 || height === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const desc = MAPS[map].join("");
    const g = gridGeometry(MAPS[map].length, MAPS[map][0].length, width, height);
    const styles = getComputedStyle(canvas);
    const ink = styles.color || "#888";

    // Shade relative to the best cell, so the picture is readable on the
    // slippery maps too, where no value comes near 1.
    let best = 0;
    for (let s = 0; s < g.rows * g.cols; s++) best = Math.max(best, cellArrow(q, s).value);

    for (let s = 0; s < g.rows * g.cols; s++) {
      const { x, y } = cellOrigin(s, g);
      const letter = desc[s];
      const inset = Math.max(1, g.cell * 0.04);

      if (letter === "H") {
        ctx.fillStyle = "rgba(100,116,139,0.55)"; // a hole: slate, value-free
      } else if (letter === "G") {
        ctx.fillStyle = "rgba(16,185,129,0.75)"; // the goal: emerald
      } else {
        const v = best > 0 ? cellArrow(q, s).value / best : 0;
        // Fixed blue at alpha = value: a sequential encoding with a meaningful
        // zero (model-visualization.md §3), readable in both themes.
        ctx.fillStyle = `rgba(59,130,246,${(0.06 + 0.7 * Math.max(0, v)).toFixed(3)})`;
      }
      ctx.fillRect(x + inset, y + inset, g.cell - 2 * inset, g.cell - 2 * inset);

      const cx = x + g.cell / 2;
      const cy = y + g.cell / 2;
      if (letter === "H" || letter === "G") {
        ctx.fillStyle = ink;
        ctx.font = `${Math.round(g.cell * 0.16)}px ui-sans-serif, system-ui`;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(letter === "H" ? "hole" : "goal", x + inset * 3, y + inset * 3);
        continue;
      }

      // The greedy arrow. A cell whose actions all tie gets a faint one: that
      // arrow is np.argmax's tie-break, not something the agent learned.
      const arrow = cellArrow(q, s);
      const [dx, dy] = ACTION_VECTORS[arrow.action];
      const len = g.cell * 0.3;
      ctx.strokeStyle = ink;
      ctx.fillStyle = ink;
      ctx.globalAlpha = arrow.tie ? 0.18 : 0.9;
      ctx.lineWidth = Math.max(1.2, g.cell * 0.045);
      ctx.beginPath();
      ctx.moveTo(cx - dx * len * 0.6, cy - dy * len * 0.6);
      ctx.lineTo(cx + dx * len * 0.6, cy + dy * len * 0.6);
      ctx.stroke();
      const hx = cx + dx * len;
      const hy = cy + dy * len;
      const head = g.cell * 0.13;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(hx - dx * head + dy * head * 0.7, hy - dy * head + dx * head * 0.7);
      ctx.lineTo(hx - dx * head - dy * head * 0.7, hy - dy * head - dx * head * 0.7);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      if (letter === "S") {
        ctx.font = `${Math.round(g.cell * 0.16)}px ui-sans-serif, system-ui`;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("start", x + inset * 3, y + inset * 3);
      }
    }

    // The agent, last, so it sits over the arrow of the cell it is in.
    const { x, y } = cellOrigin(agent, g);
    ctx.fillStyle = "rgba(244,63,94,0.95)"; // rose — nothing else on the grid is
    ctx.beginPath();
    ctx.arc(x + g.cell / 2, y + g.cell / 2, g.cell * 0.14, 0, Math.PI * 2);
    ctx.fill();
  }, [size, map, q, agent]);

  return (
    <div
      ref={hostRef}
      className="relative aspect-square w-full max-w-[20rem] text-foreground"
      data-testid="grid-canvas"
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 size-full text-foreground"
        role="img"
        aria-label={`The ${map} grid, with the learned greedy action drawn as an arrow in every cell`}
      />
    </div>
  );
}
