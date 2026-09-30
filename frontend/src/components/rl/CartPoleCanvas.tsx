// CartPole, drawn: a track, a cart, a pole and its angle. Repainted from the
// worker's render state at most 60 times a second; the geometry is
// `cartPoleGeometry.ts`, where a test can see it.

import { useEffect, useRef, useState } from "react";

import { THETA_THRESHOLD } from "@/rl/envs/cartPole";

import { cartPoleGeometry } from "./cartPoleGeometry";

export function CartPoleCanvas({ state }: { state: Float32Array }) {
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

  const [x, , theta] = state;

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

    const g = cartPoleGeometry(x, theta, width, height);
    const ink = getComputedStyle(canvas).color || "#888";
    const fallen = Math.abs(theta) > THETA_THRESHOLD;

    // The track, with its ±2.4 m ends marked: running off it is a termination too.
    ctx.strokeStyle = ink;
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, g.trackY);
    ctx.lineTo(width, g.trackY);
    for (const end of [g.trackLeft, g.trackRight]) {
      ctx.moveTo(end, g.trackY - 8);
      ctx.lineTo(end, g.trackY + 8);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.fillStyle = "rgba(59,130,246,0.85)"; // the cart: blue
    ctx.fillRect(g.cartX - g.cartWidth / 2, g.trackY - g.cartHeight, g.cartWidth, g.cartHeight);

    // The pole: amber while up, rose once past 12°.
    ctx.strokeStyle = fallen ? "rgba(244,63,94,0.95)" : "rgba(245,158,11,0.95)";
    ctx.lineWidth = Math.max(4, g.scale * 0.05);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(g.pivot.x, g.pivot.y);
    ctx.lineTo(g.tip.x, g.tip.y);
    ctx.stroke();

    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(g.pivot.x, g.pivot.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }, [size, x, theta]);

  const degrees = (theta * 180) / Math.PI;
  return (
    <div className="space-y-1">
      <div ref={hostRef} className="relative h-40 w-full text-foreground" data-testid="cartpole-canvas">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 size-full text-foreground"
          role="img"
          aria-label={`A cart at ${x.toFixed(2)} m with its pole at ${degrees.toFixed(1)} degrees`}
        />
      </div>
      <p className="font-mono text-xs text-muted-foreground tabular-nums">
        x {x.toFixed(2)} m · θ {degrees.toFixed(1)}° — the episode ends past ±12° or ±2.4 m
      </p>
    </div>
  );
}
