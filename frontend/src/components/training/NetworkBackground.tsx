// Full-bleed, animated visualization of the linear model's *structure* — its
// pooled input field, its 10 output neurons, and the weighted connections
// between them — rendered as the background of the Training stage.
//
// The model is 784 → 10. Drawing 784 input nodes (and 7,840 edges) is noise, so
// the 28×28 input field is pooled into a coarse 7×7 grid: 49 input nodes and
// 49 × 10 = 490 legible edges. Each edge's colour encodes the sign of the pooled
// weight (red = for, blue = against) and its opacity/width encode magnitude.
// Output neurons glow by their bias. The whole thing tweens toward every new
// weight snapshot, so the user watches connections strengthen as training runs.
//
// Purely presentational: it reads a Float32Array of weights (+ optional bias)
// and never touches the trainer. Its own rAF loop is decoupled from React state.

import { useEffect, useRef } from "react";

const SIDE = 28; // 28×28 input image
const POOL = 7; // 7×7 pooled grid
const BLOCK = SIDE / POOL; // 4×4 pixels per pool
const POOLS = POOL * POOL; // 49 input nodes

// Diverging endpoints, matched to ModelWeights (Tailwind red-500 / blue-500).
const POS: [number, number, number] = [239, 68, 68];
const NEG: [number, number, number] = [59, 130, 246];
const NEUTRAL: [number, number, number] = [148, 163, 184]; // slate-400, theme-safe

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true
  );
}

/** Pool a `784 × C` weight matrix down to `49 × C` (mean over each 4×4 block). */
function poolWeights(
  weights: Float32Array,
  numClasses: number,
): Float32Array {
  const out = new Float32Array(POOLS * numClasses);
  if (weights.length < SIDE * SIDE * numClasses) return out;
  for (let y = 0; y < SIDE; y++) {
    const py = Math.floor(y / BLOCK);
    for (let x = 0; x < SIDE; x++) {
      const px = Math.floor(x / BLOCK);
      const pool = py * POOL + px;
      const d = y * SIDE + x;
      for (let c = 0; c < numClasses; c++) {
        out[pool * numClasses + c] += weights[d * numClasses + c];
      }
    }
  }
  const inv = 1 / (BLOCK * BLOCK);
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

export function NetworkBackground({
  weights,
  bias,
  numClasses = 10,
  active = false,
  className,
}: {
  weights: Float32Array;
  bias?: Float32Array;
  numClasses?: number;
  active?: boolean;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Target pooled weights (updated on prop change) and the displayed values that
  // ease toward them each frame. Kept in refs so the rAF loop never re-subscribes.
  const targetRef = useRef<Float32Array>(new Float32Array(POOLS * numClasses));
  const displayRef = useRef<Float32Array>(new Float32Array(POOLS * numClasses));
  const biasRef = useRef<Float32Array | undefined>(bias);
  const activeRef = useRef(active);

  activeRef.current = active;
  biasRef.current = bias;

  // Recompute the pooled target whenever the weights buffer changes.
  useEffect(() => {
    targetRef.current = poolWeights(weights, numClasses);
    if (displayRef.current.length !== targetRef.current.length) {
      displayRef.current = new Float32Array(targetRef.current.length);
    }
    if (prefersReducedMotion()) {
      displayRef.current.set(targetRef.current);
    }
  }, [weights, numClasses]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let dash = 0;
    let cssW = 0;
    let cssH = 0;
    let textColor = "rgba(148,163,184,0.9)";

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      cssW = rect.width;
      cssH = rect.height;
      canvas.width = Math.max(1, Math.round(cssW * dpr));
      canvas.height = Math.max(1, Math.round(cssH * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      textColor = getComputedStyle(canvas).color || textColor;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const reduce = prefersReducedMotion();

    const draw = () => {
      const target = targetRef.current;
      const display = displayRef.current;
      const C = numClasses;

      // Ease displayed weights toward the target (skip when reduced-motion).
      if (!reduce) {
        for (let i = 0; i < display.length; i++) {
          display[i] += (target[i] - display[i]) * 0.08;
        }
      }

      // Normalisation constant over the currently displayed pooled weights.
      let maxAbs = 1e-6;
      for (let i = 0; i < display.length; i++) {
        const a = Math.abs(display[i]);
        if (a > maxAbs) maxAbs = a;
      }

      ctx.clearRect(0, 0, cssW, cssH);
      if (cssW < 4 || cssH < 4) {
        raf = requestAnimationFrame(draw);
        return;
      }

      // ---- Layout -------------------------------------------------------
      // Input pool grid on the left, output neuron column on the right.
      const inX0 = cssW * 0.14;
      const inSpan = Math.min(cssW * 0.28, cssH * 0.72);
      const inStep = inSpan / (POOL - 1);
      const inY0 = (cssH - inSpan) / 2;
      const poolPos = (p: number): [number, number] => [
        inX0 + (p % POOL) * inStep,
        inY0 + Math.floor(p / POOL) * inStep,
      ];

      const outX = cssW * 0.84;
      const outSpan = Math.min(cssH * 0.82, C * 56);
      const outStep = outSpan / (C - 1 || 1);
      const outY0 = (cssH - outSpan) / 2;
      const outPos = (c: number): [number, number] => [
        outX,
        outY0 + c * outStep,
      ];
      const outR = Math.min(16, outStep * 0.32);

      // ---- Edges --------------------------------------------------------
      if (activeRef.current && !reduce) dash = (dash + 0.6) % 12;
      ctx.lineCap = "round";
      for (let p = 0; p < POOLS; p++) {
        const [x0, y0] = poolPos(p);
        for (let c = 0; c < C; c++) {
          const w = display[p * C + c];
          const t = w / maxAbs; // ~[-1, 1]
          const mag = Math.min(1, Math.abs(t));
          if (mag < 0.05) continue;
          const [x1, y1] = outPos(c);
          const [r, g, b] = t >= 0 ? POS : NEG;
          ctx.strokeStyle = `rgba(${r},${g},${b},${(mag * 0.5).toFixed(3)})`;
          ctx.lineWidth = 0.4 + mag * 2.2;
          if (activeRef.current && !reduce && mag > 0.4) {
            ctx.setLineDash([2, 10]);
            ctx.lineDashOffset = -dash;
          } else {
            ctx.setLineDash([]);
          }
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // ---- Input pool nodes --------------------------------------------
      // Fill intensity = aggregate importance (Σ|w| across classes), so the
      // grid faintly echoes where the model is "looking".
      const [nr, ng, nb] = NEUTRAL;
      for (let p = 0; p < POOLS; p++) {
        let agg = 0;
        for (let c = 0; c < C; c++) agg += Math.abs(display[p * C + c]);
        const a = Math.min(0.9, (agg / (maxAbs * C)) * 3 + 0.08);
        const [x, y] = poolPos(p);
        const s = inStep * 0.34;
        ctx.fillStyle = `rgba(${nr},${ng},${nb},${a.toFixed(3)})`;
        ctx.beginPath();
        ctx.roundRect(x - s, y - s, s * 2, s * 2, 2);
        ctx.fill();
      }

      // ---- Output neurons ----------------------------------------------
      const biasArr = biasRef.current;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = `600 ${Math.round(outR * 0.95)}px ui-monospace, monospace`;
      for (let c = 0; c < C; c++) {
        const [x, y] = outPos(c);
        const bv = biasArr ? biasArr[c] : 0;
        const glow = Math.min(1, Math.abs(bv));
        const [gr, gg, gb] = bv >= 0 ? POS : NEG;

        if (glow > 0.02) {
          ctx.shadowColor = `rgba(${gr},${gg},${gb},0.9)`;
          ctx.shadowBlur = 6 + glow * 14;
        }
        ctx.fillStyle = `rgba(${nr},${ng},${nb},0.16)`;
        ctx.beginPath();
        ctx.arc(x, y, outR, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = `rgba(${gr},${gg},${gb},${(0.3 + glow * 0.6).toFixed(3)})`;
        ctx.stroke();

        ctx.fillStyle = textColor;
        ctx.fillText(String(c), x, y + 0.5);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [numClasses]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={className}
    />
  );
}
