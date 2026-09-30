// The reaching task drawn: the obstacle, start and target, the demonstrations
// coloured by which side they went, and — once a policy exists — its rollout and
// its action field. The field is the point: it is where the two modes visibly
// cancel into an arrow aimed at the obstacle.

import { useEffect, useRef, useState } from "react";

import type { FieldArrow, Rollout } from "@/rl/behaviourCloning";
import type { Demonstration } from "@/rl/demos";
import { forwardKinematics, type ReacherScene } from "@/rl/envs/reacher2d";

import { SIDE_COLORS } from "./constants";
import { workspaceMap } from "./workspaceGeometry";

export function Workspace({
  scene,
  demos,
  rollout,
  field,
  testId = "workspace",
}: {
  scene: ReacherScene;
  demos: readonly Demonstration[];
  rollout?: Rollout | null;
  field?: readonly FieldArrow[] | null;
  testId?: string;
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
    const ink = getComputedStyle(canvas).color || "#888";
    const m = workspaceMap(width, height);
    const px = m.toPixels;

    // The obstacle.
    const o = px(scene.obstacle);
    ctx.fillStyle = "rgba(100,116,139,0.45)";
    ctx.beginPath();
    ctx.arc(o.x, o.y, scene.radius * m.scale, 0, Math.PI * 2);
    ctx.fill();

    // The demonstrations, one thin line each, coloured by side.
    ctx.lineWidth = 1;
    for (const d of demos) {
      ctx.strokeStyle = SIDE_COLORS[d.side];
      ctx.beginPath();
      d.path.forEach((p, i) => {
        const q = px(p);
        if (i === 0) ctx.moveTo(q.x, q.y);
        else ctx.lineTo(q.x, q.y);
      });
      ctx.stroke();
    }

    // The action field: where the policy would move the hand from each point.
    if (field) {
      ctx.strokeStyle = ink;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1;
      const len = 0.09 * m.scale;
      for (const f of field) {
        const a = px(f.at);
        const speed = Math.hypot(f.v.x, f.v.y) || 1;
        const dx = (f.v.x / speed) * len;
        const dy = (-f.v.y / speed) * len; // world y up, canvas y down
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(a.x + dx, a.y + dy);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(a.x + dx, a.y + dy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // The policy's own rollout, and the arm in its final pose.
    if (rollout && rollout.path.length > 0) {
      ctx.strokeStyle =
        rollout.outcome === "reached" ? "rgba(16,185,129,0.95)" : "rgba(244,63,94,0.95)";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      rollout.path.forEach((p, i) => {
        const q = px(p);
        if (i === 0) ctx.moveTo(q.x, q.y);
        else ctx.lineTo(q.x, q.y);
      });
      ctx.stroke();
    }

    // Start and target.
    for (const [p, label] of [
      [scene.start, "start"],
      [scene.target, "target"],
    ] as const) {
      const q = px(p);
      ctx.fillStyle = ink;
      ctx.beginPath();
      ctx.arc(q.x, q.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "11px ui-sans-serif, system-ui";
      ctx.textAlign = "center";
      ctx.fillText(label, q.x, q.y + 16);
    }

    // The base and a faint arm at the start pose, so "a two-link arm" reads.
    const base = px({ x: 0, y: 0 });
    ctx.fillStyle = ink;
    ctx.fillRect(base.x - 6, base.y - 3, 12, 6);
    if (demos[0]) {
      const q0 = demos[0].obs;
      const { elbow, hand } = forwardKinematics(q0[0], q0[1]);
      ctx.strokeStyle = ink;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(base.x, base.y);
      ctx.lineTo(px(elbow).x, px(elbow).y);
      ctx.lineTo(px(hand).x, px(hand).y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }, [size, scene, demos, rollout, field]);

  return (
    <div ref={hostRef} className="relative aspect-[6/5] w-full max-w-[34rem] text-foreground" data-testid={testId}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 size-full text-foreground"
        role="img"
        aria-label={`${demos.length} demonstrations round an obstacle${rollout ? `, and the policy's rollout, which ${rollout.outcome}` : ""}`}
      />
    </div>
  );
}
