import { useMemo, useState, type ReactNode, type UIEvent } from "react";
import { toast } from "../../stores/ui";
import { toCsv, type CsvCell } from "./csv";
import { copyText, downloadText, fileSlug } from "./download";
import { Icon } from "./Icon";
import { visibleRange } from "./VirtualList";

export type Column<T> = {
  key: string;
  label: string;
  unit?: string;
  align?: "left" | "right";
  /** Raw value for sorting and CSV (numbers stay unformatted in CSV). */
  value?: (row: T) => CsvCell;
  /** Cell content; defaults to the formatted value. */
  render?: (row: T) => ReactNode;
  sortable?: boolean;
  width?: string;
};

export type TableProps<T> = {
  columns: Column<T>[];
  rows: readonly T[];
  rowKey?: (row: T, i: number) => string | number;
  caption?: ReactNode;
  /** Highlighted rows (`.hl`). */
  highlight?: (row: T) => boolean;
  onRowClick?: (row: T) => void;
  initialSort?: { key: string; dir: "asc" | "desc" };
  /** Rows above this count are virtualised (default 500). */
  virtualizeOver?: number;
  rowHeight?: number;
  maxHeight?: number;
  /** Show the CSV copy/download buttons; the value names the file. */
  csvName?: string | null;
  empty?: ReactNode;
};

/** Missing for sorting: null, undefined, NaN and ±Infinity (always listed last). */
const isMissing = (v: CsvCell) => v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v));

function cmp(a: CsvCell, b: CsvCell): number {
  const na = isMissing(a);
  const nb = isMissing(b);
  if (na || nb) return na === nb ? 0 : na ? 1 : -1; // missing values last
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

/** The table's rows as CSV (column labels with units; raw values). */
export function tableCsv<T>(columns: Column<T>[], rows: readonly T[]): string {
  const header = columns.map((c) => (c.unit ? `${c.label} (${c.unit})` : c.label));
  return toCsv(
    header,
    rows.map((r) => columns.map((c) => (c.value ? c.value(r) : (r as Record<string, CsvCell>)[c.key]))),
  );
}

/** Sortable table; virtualised beyond `virtualizeOver` rows; CSV copy and download. */
export function Table<T>({
  columns,
  rows,
  rowKey,
  caption,
  highlight,
  onRowClick,
  initialSort,
  virtualizeOver = 500,
  rowHeight = 33,
  maxHeight = 480,
  csvName = null,
  empty,
}: TableProps<T>) {
  const [sort, setSort] = useState(initialSort ?? null);
  const [scrollTop, setScrollTop] = useState(0);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    const get = (r: T) => (col.value ? col.value(r) : (r as Record<string, CsvCell>)[col.key]);
    const out = [...rows].sort((a, b) => cmp(get(a), get(b)));
    if (sort.dir === "desc") {
      // keep missing values last when descending
      const present = out.filter((r) => !isMissing(get(r)));
      const missing = out.filter((r) => isMissing(get(r)));
      return [...present.reverse(), ...missing];
    }
    return out;
  }, [rows, columns, sort]);

  const virtual = sorted.length > virtualizeOver;
  const [first, last] = virtual ? visibleRange(scrollTop, maxHeight, rowHeight, sorted.length, 10) : [0, sorted.length];

  const toggle = (key: string) =>
    setSort((s) => (!s || s.key !== key ? { key, dir: "asc" } : s.dir === "asc" ? { key, dir: "desc" } : null));

  const cell = (c: Column<T>, r: T): ReactNode => {
    if (c.render) return c.render(r);
    const v = c.value ? c.value(r) : (r as Record<string, CsvCell>)[c.key];
    return v === null || v === undefined ? "—" : String(v);
  };

  const csv = () => tableCsv(columns, sorted);

  return (
    <div className="stack" style={{ gap: 6 }}>
      {csvName !== null ? (
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button
            type="button"
            className="btn small ghost"
            onClick={async () => {
              const ok = await copyText(csv());
              toast(ok ? "success" : "warning", ok ? "Table copied as CSV" : "Copy failed: use Download CSV");
            }}
          >
            <Icon name="copy" /> Copy CSV
          </button>
          <button type="button" className="btn small ghost" onClick={() => downloadText(csv(), `${fileSlug(csvName || "table")}.csv`, "text/csv;charset=utf-8")}>
            <Icon name="download" /> Download CSV
          </button>
        </div>
      ) : null}
      <div className="tablewrap" style={virtual ? { maxHeight, overflowY: "auto" } : undefined} onScroll={virtual ? (e: UIEvent<HTMLDivElement>) => setScrollTop(e.currentTarget.scrollTop) : undefined}>
        <table className="tbl">
          {caption ? <caption className="sr-only">{caption}</caption> : null}
          <thead>
            <tr>
              {columns.map((c) => {
                const active = sort?.key === c.key;
                const label = c.unit ? `${c.label} (${c.unit})` : c.label;
                return (
                  <th
                    key={c.key}
                    className={c.align === "right" ? "r" : undefined}
                    style={c.width ? { width: c.width } : undefined}
                    aria-sort={active ? (sort!.dir === "asc" ? "ascending" : "descending") : undefined}
                    scope="col"
                  >
                    {c.sortable === false ? (
                      label
                    ) : (
                      <button type="button" onClick={() => toggle(c.key)}>
                        {label}
                        {active ? <Icon name={sort!.dir === "asc" ? "chevronUp" : "chevronDown"} size={12} /> : null}
                      </button>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="muted">
                  {empty ?? "No rows"}
                </td>
              </tr>
            ) : null}
            {virtual && first > 0 ? (
              <tr className="spacer-row" aria-hidden="true">
                <td colSpan={columns.length} style={{ height: first * rowHeight }} />
              </tr>
            ) : null}
            {sorted.slice(first, last).map((r, k) => {
              const i = first + k;
              return (
                <tr
                  key={rowKey ? rowKey(r, i) : i}
                  className={highlight?.(r) ? "hl" : undefined}
                  onClick={onRowClick ? () => onRowClick(r) : undefined}
                  style={virtual ? { height: rowHeight } : onRowClick ? { cursor: "pointer" } : undefined}
                >
                  {columns.map((c) => (
                    <td key={c.key} className={c.align === "right" ? "r" : undefined}>
                      {cell(c, r)}
                    </td>
                  ))}
                </tr>
              );
            })}
            {virtual && last < sorted.length ? (
              <tr className="spacer-row" aria-hidden="true">
                <td colSpan={columns.length} style={{ height: (sorted.length - last) * rowHeight }} />
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
