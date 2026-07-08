// Visualises a trained linear (784 → 10) classifier as its 10 weight
// "templates". Each output class has 784 weights that reshape to a 28×28 grid —
// literally a picture of what that digit detector responds to. Colour encodes
// the signed weight value: red = positive (evidence *for* the class at that
// pixel), blue = negative (evidence *against*), transparent ≈ zero. Alpha scales
// with magnitude, so near-zero weights fade into the card background and the
// visualisation reads correctly in both light and dark themes.

import { useEffect, useMemo, useRef } from "react";

const SIDE = 28; // 28×28 input image
const CELL = SIDE * SIDE; // 784 weights per class

// Diverging endpoints (Tailwind red-500 / blue-500).
const POS: [number, number, number] = [239, 68, 68];
const NEG: [number, number, number] = [59, 130, 246];

/**
 * @param weights row-major `784 × numClasses` (weight for feature d, class c is
 *   `weights[d * numClasses + c]`).
 * @param bias one value per class.
 */
export function ModelWeights({
  weights,
  bias,
  numClasses = 10,
  epoch,
  tileSize = "size-16",
}: {
  weights: Float32Array;
  bias?: Float32Array;
  numClasses?: number;
  epoch?: number;
  /** Tailwind size class for each template tile (e.g. "size-16", "size-32"). */
  tileSize?: string;
}) {
  // Normalise colour intensity by the largest-magnitude weight so classes are
  // directly comparable.
  const maxAbs = useMemo(() => {
    let m = 0;
    for (let i = 0; i < weights.length; i++) {
      const a = Math.abs(weights[i]);
      if (a > m) m = a;
    }
    return m || 1;
  }, [weights]);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {Array.from({ length: numClasses }, (_, c) => (
          <div key={c} className="flex flex-none flex-col items-center gap-1">
            <ClassTemplate
              weights={weights}
              cls={c}
              numClasses={numClasses}
              maxAbs={maxAbs}
              tileSize={tileSize}
            />
            <span className="font-mono text-xs tabular-nums text-muted-foreground">
              {c}
            </span>
            {bias && (
              <span className="font-mono text-[10px] tabular-nums text-muted-foreground/70">
                b {bias[c] >= 0 ? "+" : ""}
                {bias[c].toFixed(2)}
              </span>
            )}
          </div>
        ))}
      </div>

      <Legend maxAbs={maxAbs} epoch={epoch} />
    </div>
  );
}

function ClassTemplate({
  weights,
  cls,
  numClasses,
  maxAbs,
  tileSize,
}: {
  weights: Float32Array;
  cls: number;
  numClasses: number;
  maxAbs: number;
  tileSize: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(SIDE, SIDE);
    for (let d = 0; d < CELL; d++) {
      const w = weights[d * numClasses + cls];
      const t = w / maxAbs; // roughly [-1, 1]
      const [r, g, b] = t >= 0 ? POS : NEG;
      const o = d * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = Math.round(Math.min(1, Math.abs(t)) * 255);
    }
    ctx.putImageData(img, 0, 0);
  }, [weights, cls, numClasses, maxAbs]);

  return (
    <canvas
      ref={canvasRef}
      width={SIDE}
      height={SIDE}
      className={`${tileSize} rounded border bg-muted/30 [image-rendering:pixelated]`}
      aria-label={`Weight template for digit ${cls}`}
    />
  );
}

function Legend({ maxAbs, epoch }: { maxAbs: number; epoch?: number }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <div className="flex items-center gap-2">
        <span className="font-mono tabular-nums">−{maxAbs.toFixed(2)}</span>
        <span
          className="h-2.5 w-28 rounded"
          style={{
            background: `linear-gradient(to right, rgb(${NEG.join()}), rgba(148,163,184,0.15), rgb(${POS.join()}))`,
          }}
        />
        <span className="font-mono tabular-nums">+{maxAbs.toFixed(2)}</span>
      </div>
      <span>
        <span className="text-[rgb(59,130,246)]">blue</span> = against ·{" "}
        <span className="text-[rgb(239,68,68)]">red</span> = for
      </span>
      {epoch !== undefined && epoch >= 0 && (
        <span className="ml-auto font-mono tabular-nums">
          after epoch {epoch + 1}
        </span>
      )}
    </div>
  );
}
