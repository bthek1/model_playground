// Parse and format the row-major matrices used by the tensor-arithmetic page.
// Rows are separated by newlines (or semicolons); values by whitespace or commas.

export interface ParsedMatrix {
  data: Float32Array;
  rows: number;
  cols: number;
}

/**
 * Parse free-form text into a rectangular matrix. Throws with a human-readable
 * message on empty input, ragged rows, or non-numeric cells.
 */
export function parseMatrix(text: string): ParsedMatrix {
  const rows = text
    .split(/[\n;]/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.split(/[\s,]+/).filter((cell) => cell.length > 0));

  if (rows.length === 0) {
    throw new Error("Enter at least one row of numbers.");
  }

  const cols = rows[0].length;
  const values: number[] = [];
  for (let r = 0; r < rows.length; r++) {
    if (rows[r].length !== cols) {
      throw new Error(
        `Row ${r + 1} has ${rows[r].length} values but row 1 has ${cols}. Every row must be the same length.`,
      );
    }
    for (const cell of rows[r]) {
      const value = Number(cell);
      if (!Number.isFinite(value)) {
        throw new Error(`"${cell}" is not a number.`);
      }
      values.push(value);
    }
  }

  return { data: new Float32Array(values), rows: rows.length, cols };
}

/** Format a flat row-major buffer as a 2-D array of trimmed number strings. */
export function formatMatrix(
  data: Float32Array,
  rows: number,
  cols: number,
  precision = 4,
): string[][] {
  const out: string[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: string[] = [];
    for (let c = 0; c < cols; c++) {
      row.push(trimNumber(data[r * cols + c], precision));
    }
    out.push(row);
  }
  return out;
}

// Round to `precision` decimals, then drop trailing zeros ("1.5000" -> "1.5").
function trimNumber(value: number, precision: number): string {
  if (!Number.isFinite(value)) return String(value);
  return String(Number(value.toFixed(precision)));
}
