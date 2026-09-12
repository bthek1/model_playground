// A force-directed layout for the /graph route, and the most expensive thing on
// that page — more expensive than the GNN it draws. It runs once in the worker
// and the coordinates are cached for the session: changing the architecture, the
// depth or any hyperparameter re-trains the model but must never re-lay-out the
// graph, or the user loses the mental map they were reading the result against.
//
// Fruchterman–Reingold, with two departures from the textbook version, both
// forced by 2708 nodes:
//
//   - Repulsion is **grid-binned**. All-pairs is 7.3 M distance computations per
//     iteration, and 300 iterations of that is not something to do in a tab.
//     Nodes are bucketed into cells two ideal-edge-lengths wide and only repel
//     within the 3x3 neighbourhood, which is where the force is worth computing.
//   - A weak **gravity** term pulls everything toward the centroid. Purely local
//     repulsion cannot bound the drawing, so a disconnected component — Cora has
//     78 of them — would otherwise wander off the canvas for ever.
//
// Deterministic by construction: same graph, same seed, same picture.

import { mulberry32 } from "./random";

export interface GraphLayout {
  /** Node coordinates, normalised into the unit square. */
  x: Float32Array;
  y: Float32Array;
}

export interface LayoutOptions {
  iterations?: number;
  seed?: number;
  /** Pull toward the centroid, as a fraction of the ideal edge length. */
  gravity?: number;
}

/**
 * Lay out a CSR graph. `rowPtr`/`colIdx` hold each undirected edge twice, which
 * is what the aggregation kernel wants; the attraction loop therefore visits
 * each edge from both endpoints and halves its strength.
 */
export function forceLayout(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  nNodes: number,
  options: LayoutOptions = {},
): GraphLayout {
  const { iterations = 300, seed = 1, gravity = 0.02 } = options;
  const x = new Float32Array(nNodes);
  const y = new Float32Array(nNodes);
  if (nNodes === 0) return { x, y };

  const rand = mulberry32(seed);
  for (let i = 0; i < nNodes; i++) {
    x[i] = rand();
    y[i] = rand();
  }
  if (nNodes === 1) return { x, y };

  // The ideal edge length: the side of the square each node would get to itself.
  const k = Math.sqrt(1 / nNodes);
  const k2 = k * k;
  const cell = 2 * k;
  const cutoff2 = cell * cell;

  const dx = new Float32Array(nNodes);
  const dy = new Float32Array(nNodes);

  // Reusable grid buckets, rebuilt each iteration by counting sort.
  const cols = Math.max(1, Math.ceil(1 / cell));
  const cellCount = cols * cols;
  const cellStart = new Uint32Array(cellCount + 1);
  const cellItems = new Uint32Array(nNodes);
  const cellOf = new Uint32Array(nNodes);
  const cursor = new Uint32Array(cellCount);

  for (let iter = 0; iter < iterations; iter++) {
    // Cooling: displacement is capped at `temp`, shrinking to zero, so the
    // layout settles instead of oscillating between two mirror images.
    const temp = 0.1 * (1 - iter / iterations) + 1e-4;

    dx.fill(0);
    dy.fill(0);

    // --- bucket -----------------------------------------------------------
    cellStart.fill(0);
    for (let i = 0; i < nNodes; i++) {
      const cx = clampIndex(Math.floor(x[i] / cell), cols);
      const cy = clampIndex(Math.floor(y[i] / cell), cols);
      const c = cy * cols + cx;
      cellOf[i] = c;
      cellStart[c + 1]++;
    }
    for (let c = 0; c < cellCount; c++) cellStart[c + 1] += cellStart[c];
    cursor.set(cellStart.subarray(0, cellCount));
    for (let i = 0; i < nNodes; i++) cellItems[cursor[cellOf[i]]++] = i;

    // --- repulsion, within the 3x3 cell neighbourhood ----------------------
    for (let i = 0; i < nNodes; i++) {
      const c = cellOf[i];
      const cx = c % cols;
      const cy = (c - cx) / cols;
      for (let gy = Math.max(0, cy - 1); gy <= Math.min(cols - 1, cy + 1); gy++) {
        for (let gx = Math.max(0, cx - 1); gx <= Math.min(cols - 1, cx + 1); gx++) {
          const g = gy * cols + gx;
          for (let s = cellStart[g]; s < cellStart[g + 1]; s++) {
            const j = cellItems[s];
            if (j === i) continue;
            let ex = x[i] - x[j];
            let ey = y[i] - y[j];
            let d2 = ex * ex + ey * ey;
            if (d2 > cutoff2) continue;
            if (d2 < 1e-12) {
              // Coincident nodes have no direction to separate along; nudge
              // them apart deterministically by index so the layout stays
              // reproducible.
              ex = (i - j) * 1e-6;
              ey = 1e-6;
              d2 = ex * ex + ey * ey;
            }
            const d = Math.sqrt(d2);
            const force = k2 / d;
            dx[i] += (ex / d) * force;
            dy[i] += (ey / d) * force;
          }
        }
      }
    }

    // --- attraction along edges -------------------------------------------
    for (let u = 0; u < nNodes; u++) {
      for (let e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
        const v = colIdx[e];
        const ex = x[u] - x[v];
        const ey = y[u] - y[v];
        const d = Math.hypot(ex, ey);
        if (d < 1e-12) continue;
        // Halved: the loop sees every undirected edge from both ends.
        const force = (d * d) / k / 2;
        dx[u] -= (ex / d) * force;
        dy[u] -= (ey / d) * force;
      }
    }

    // --- gravity, and the move --------------------------------------------
    let cxSum = 0;
    let cySum = 0;
    for (let i = 0; i < nNodes; i++) {
      cxSum += x[i];
      cySum += y[i];
    }
    const centreX = cxSum / nNodes;
    const centreY = cySum / nNodes;

    for (let i = 0; i < nNodes; i++) {
      dx[i] += (centreX - x[i]) * gravity;
      dy[i] += (centreY - y[i]) * gravity;

      const d = Math.hypot(dx[i], dy[i]);
      if (d < 1e-12) continue;
      const step = Math.min(d, temp);
      x[i] += (dx[i] / d) * step;
      y[i] += (dy[i] / d) * step;
    }
  }

  return normalise(x, y);
}

function clampIndex(v: number, cols: number): number {
  if (!Number.isFinite(v) || v < 0) return 0;
  return v >= cols ? cols - 1 : v;
}

/**
 * Rescale into the unit square, preserving aspect ratio so the drawing is not
 * stretched. A degenerate axis (every node on one line) collapses to 0.5 rather
 * than dividing by zero.
 */
function normalise(x: Float32Array, y: Float32Array): GraphLayout {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < x.length; i++) {
    if (x[i] < minX) minX = x[i];
    if (x[i] > maxX) maxX = x[i];
    if (y[i] < minY) minY = y[i];
    if (y[i] > maxY) maxY = y[i];
  }
  const span = Math.max(maxX - minX, maxY - minY);
  if (!Number.isFinite(span) || span < 1e-9) {
    x.fill(0.5);
    y.fill(0.5);
    return { x, y };
  }
  // Centre the shorter axis in the square.
  const padX = (span - (maxX - minX)) / 2;
  const padY = (span - (maxY - minY)) / 2;
  for (let i = 0; i < x.length; i++) {
    x[i] = (x[i] - minX + padX) / span;
    y[i] = (y[i] - minY + padY) / span;
  }
  return { x, y };
}
