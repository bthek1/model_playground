// A CSV parser that writes straight into typed arrays.
//
// There is no `papaparse` in this repo and adding one would be the wrong call,
// which is worth stating because it looks like a gap. A general-purpose parser
// returns an array of row objects, and an array of row objects *is* the memory
// problem this whole module exists to avoid: the point is one `Float32Array`
// per column, allocated once, and nothing else. So the parse is ours.
//
// Pure — no DOM, no worker, no I/O. `fit.worker.ts` is what keeps it off the
// main thread; that is a scheduling decision and not this file's business.

import type { Column, Dataset, ParseIssue, ParseResult } from "./types";

export interface ParseOptions {
  /** What the dataset is called on screen. */
  name?: string;
  /**
   * Keep at most this many rows, sampled evenly across the file so the tail is
   * represented. The UI must say the number *and* the source count: silently
   * modelling the first 50 000 rows of a file sorted by date is a different
   * dataset from the one the user handed over.
   */
  maxRows?: number;
  /**
   * A column with more than this many distinct strings is kept as text but is
   * not offered as a feature — see `design.ts`. Free-text and id columns are
   * the common case, and one-hot encoding an id column is how a model reaches
   * 100% on the training half and chance on the held-out one.
   */
  maxLevels?: number;
}

export const DEFAULT_MAX_LEVELS = 24;

/**
 * Split one CSV line into fields, honouring RFC 4180 quoting.
 *
 * Exported for the tests, which is the only reason it is not a closure: the
 * quoting rules are where a hand-written parser is wrong, and they deserve
 * assertions that do not have to build a whole file to reach them.
 */
export function splitLine(line: string): string[] {
  const out: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        // A doubled quote inside a quoted field is one literal quote.
        if (line[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"' && field.length === 0) {
      quoted = true;
    } else if (ch === ",") {
      out.push(field);
      field = "";
    } else {
      field += ch;
    }
  }
  out.push(field);
  return out;
}

/**
 * Cut the text into logical records.
 *
 * Not `text.split("\n")`: a quoted field may contain a newline, and a file with
 * an address column routinely does. Splitting first and reassembling later is
 * the bug that turns one row into three ragged ones, which this parser would
 * then dutifully reject — a correct-looking failure on a correct file.
 */
function toRecords(text: string): string[] {
  const records: string[] = [];
  let current = "";
  let quoted = false;
  // A quote only opens a field at the *start* of one. Without that rule a stray
  // apostrophe-as-quote in the middle of a free-text cell puts the whole rest of
  // the file into "inside a quoted field", and every later row merges into one.
  let fieldStart = true;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      current += ch;
      if (ch === '"') {
        if (text[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          quoted = false;
          fieldStart = false;
        }
      }
      continue;
    }
    if (ch === '"' && fieldStart) {
      quoted = true;
      current += ch;
      fieldStart = false;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      // CRLF is one break, not two.
      if (ch === "\r" && text[i + 1] === "\n") i++;
      records.push(current);
      current = "";
      fieldStart = true;
      continue;
    }
    current += ch;
    fieldStart = ch === ",";
  }
  if (current.length > 0) records.push(current);
  return records;
}

/** Everything a string may be while still meaning "no value". */
function isBlank(value: string): boolean {
  const t = value.trim();
  return t === "" || t === "NA" || t === "N/A" || t === "null" || t === "NaN";
}

/**
 * `Number(value)` with the empty string excluded.
 *
 * `Number("")` is 0, which is the single most expensive coercion in JavaScript
 * for a file parser: an empty cell in a numeric column becomes a real zero and
 * shifts every mean, every split threshold and every coefficient that column
 * touches, with nothing failing anywhere.
 */
function parseNumber(value: string): number | null {
  const t = value.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** Row indices to keep, spread evenly so the file's tail survives the cap. */
function sampleIndices(total: number, cap: number): number[] | null {
  if (total <= cap) return null;
  const keep: number[] = [];
  for (let i = 0; i < cap; i++) keep.push(Math.floor((i * total) / cap));
  return keep;
}

/**
 * Parse CSV text into one typed array per column.
 *
 * Two passes: the first records the raw strings per column so type inference
 * can see the whole column (a column of integers is still numeric when row 900
 * turns out to have a decimal, and *not* numeric when row 900 says "unknown"),
 * the second writes the typed arrays. The intermediate is one `string[]` per
 * column rather than per row — the same total characters, without an object
 * header per cell.
 */
export function parseCsv(text: string, options: ParseOptions = {}): ParseResult {
  const maxLevels = options.maxLevels ?? DEFAULT_MAX_LEVELS;
  // A BOM at the head of the file becomes part of the first header name, which
  // then never matches anything the user typed.
  const body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;

  const records = toRecords(body);
  if (records.length === 0 || records[0].trim().length === 0) {
    throw new Error("The file is empty.");
  }

  const header = splitLine(records[0]).map((h) => h.trim());
  // Duplicate header names are legal in a CSV and fatal in a column picker, so
  // they are disambiguated rather than rejected: the file is still usable and
  // renaming someone's export for them is not this parser's call.
  const seen = new Map<string, number>();
  const names = header.map((raw, i) => {
    const base = raw.length > 0 ? raw : `column_${i + 1}`;
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base} (${count + 1})`;
  });

  const width = names.length;

  // Blank lines are dropped — but only on a file with more than one column.
  // On a single-column file an empty line is not a blank line, it is a missing
  // value, and dropping it silently shortens the column instead of recording a
  // gap. The two are indistinguishable from the text alone, so the width is
  // what decides.
  const dataRecords: { line: number; text: string }[] = [];
  for (let r = 1; r < records.length; r++) {
    if (width > 1 && records[r].trim().length === 0) continue;
    dataRecords.push({ line: r + 1, text: records[r] });
  }

  const issues: ParseIssue[] = [];
  const raw: string[][] = names.map(() => []);
  let kept = 0;

  const dataLines = dataRecords.length;
  const keepList = sampleIndices(dataLines, options.maxRows ?? Infinity);
  const keepSet = keepList ? new Set(keepList) : null;

  for (let r = 0; r < dataLines; r++) {
    if (keepSet && !keepSet.has(r)) continue;
    const record = dataRecords[r];
    const fields = splitLine(record.text);
    if (fields.length !== width) {
      // Rejected with its line number, not padded or truncated. A best-effort
      // recovery here produces a model fitted on shifted columns.
      issues.push({
        line: record.line,
        message: `${fields.length} fields, expected ${width}`,
      });
      continue;
    }
    for (let c = 0; c < width; c++) raw[c].push(fields[c]);
    kept++;
  }

  if (kept === 0) {
    throw new Error(
      issues.length > 0
        ? `No usable rows: every row had the wrong number of fields (first at line ${issues[0].line}).`
        : "No data rows after the header.",
    );
  }

  const columns: Column[] = names.map((name, c) =>
    buildColumn(name, raw[c], kept, maxLevels),
  );

  return {
    dataset: {
      name: options.name ?? "dataset",
      columns,
      rowCount: kept,
      sourceRowCount: dataLines,
      sampled: keepList != null,
    },
    issues,
  };
}

/** Infer one column's kind from its strings, then write its typed arrays. */
function buildColumn(
  name: string,
  cells: string[],
  rowCount: number,
  maxLevels: number,
): Column {
  const values = new Float32Array(rowCount);
  const missing = new Uint8Array(rowCount);
  let missingCount = 0;

  // Inference looks at every present cell, not a prefix. A prefix is wrong on
  // exactly the files where it matters: an id column that is numeric for 900
  // rows and then contains "N/A", or a postcode column that loses its leading
  // zeros the moment it is called numeric.
  let numeric = true;
  for (const cell of cells) {
    if (isBlank(cell)) continue;
    if (parseNumber(cell) == null) {
      numeric = false;
      break;
    }
  }

  if (numeric) {
    for (let i = 0; i < rowCount; i++) {
      const cell = cells[i];
      if (isBlank(cell)) {
        missing[i] = 1;
        missingCount++;
        continue;
      }
      values[i] = parseNumber(cell) as number;
    }
    return { name, kind: "numeric", values, missing, missingCount };
  }

  const levels: string[] = [];
  const index = new Map<string, number>();
  for (let i = 0; i < rowCount; i++) {
    const cell = cells[i];
    if (isBlank(cell)) {
      missing[i] = 1;
      missingCount++;
      continue;
    }
    const key = cell.trim();
    let code = index.get(key);
    if (code == null) {
      code = levels.length;
      levels.push(key);
      index.set(key, code);
    }
    values[i] = code;
  }
  void maxLevels; // The cap is enforced in `design.ts`, where it can be explained.
  return { name, kind: "categorical", values, levels, missing, missingCount };
}

/** Rebuild the file's own text for one cell — for the preview table only. */
export function cellText(column: Column, row: number): string {
  if (column.missing[row]) return "";
  if (column.kind === "categorical") {
    return column.levels?.[column.values[row]] ?? "";
  }
  const v = column.values[row];
  return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(6)));
}

/** Build a `Dataset` straight from columns — the bundled samples' path. */
export function datasetFrom(name: string, columns: Column[]): Dataset {
  const rowCount = columns[0]?.values.length ?? 0;
  return { name, columns, rowCount, sourceRowCount: rowCount, sampled: false };
}
