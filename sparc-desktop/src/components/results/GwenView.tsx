import { useState, useEffect, useMemo } from "react";
import { getGwenData } from "@/lib/api";

export default function GwenView() {
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortCol, setSortCol] = useState<string>("");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    getGwenData()
      .then((d) => {
        const r = d?.rows ?? [];
        setRows(r);
        // Default sort by first importance-like column descending
        if (r.length > 0) {
          const cols = Object.keys(r[0]);
          const importCol = cols.find(
            (c) =>
              c.toLowerCase().includes("importance") ||
              c.toLowerCase().includes("score") ||
              c.toLowerCase().includes("frequency"),
          );
          if (importCol) setSortCol(importCol);
        }
      })
      .catch((e) => setError(e.message));
  }, []);

  const columns = useMemo(() => {
    if (!rows || rows.length === 0) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  const numericCols = useMemo(() => {
    if (!rows || rows.length === 0) return new Set<string>();
    return new Set(
      columns.filter((c) => typeof rows[0][c] === "number"),
    );
  }, [rows, columns]);

  const sorted = useMemo(() => {
    if (!rows) return [];
    if (!sortCol) return rows;
    return [...rows].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
  }, [rows, sortCol, sortAsc]);

  // Find max value per numeric column for bar widths
  const maxValues = useMemo(() => {
    if (!rows) return {};
    const result: Record<string, number> = {};
    for (const col of columns) {
      if (!numericCols.has(col)) continue;
      result[col] = Math.max(
        ...rows.map((r) => Math.abs(Number(r[col]) || 0)),
        0.001,
      );
    }
    return result;
  }, [rows, columns, numericCols]);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(false);
    }
  };

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">{error}</p>
      </div>
    );
  }

  if (!rows) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-500">Loading GWEN data…</p>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-sm text-sparc-gray-600">No GWEN results available.</p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-sparc-gray-100 px-4 py-3">
        <h3 className="text-sm font-semibold">
          GWEN Variable Selection — {rows.length} variables
        </h3>
        <p className="text-[10px] text-sparc-gray-400">
          Click column headers to sort. Higher importance = stronger selection signal.
        </p>
      </div>

      <div className="flex-1 overflow-auto p-4">
        <div className="overflow-auto rounded border border-sparc-gray-200">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-sparc-gray-200 bg-sparc-gray-100">
                {columns.map((col) => (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    className="cursor-pointer px-3 py-2 text-xs font-medium hover:bg-sparc-gray-200 select-none"
                  >
                    <div className="flex items-center gap-1">
                      {col}
                      {sortCol === col && (
                        <span className="text-sparc-purple">
                          {sortAsc ? "↑" : "↓"}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => {
                // Determine if this row is a "selected" feature
                const isSelected =
                  (typeof row["selected"] === "boolean" && row["selected"]) ||
                  (typeof row["selected"] === "number" && row["selected"] === 1) ||
                  (typeof row["selection_frequency"] === "number" &&
                    (row["selection_frequency"] as number) > 0.5);

                return (
                  <tr
                    key={i}
                    className={`border-b border-sparc-gray-100 ${
                      isSelected
                        ? "bg-sparc-purple/5"
                        : i % 2 === 0
                          ? "bg-white"
                          : "bg-sparc-gray-50/50"
                    }`}
                  >
                    {columns.map((col) => {
                      const val = row[col];
                      const isNumeric = numericCols.has(col);
                      const numVal = Number(val) || 0;
                      const barWidth =
                        isNumeric && maxValues[col]
                          ? (Math.abs(numVal) / maxValues[col]) * 100
                          : 0;

                      return (
                        <td key={col} className="px-3 py-1.5 text-xs tabular-nums">
                          {isNumeric ? (
                            <div className="flex items-center gap-2">
                              <span className={isSelected ? "font-medium" : ""}>
                                {numVal.toFixed(4)}
                              </span>
                              {col.toLowerCase().includes("importance") ||
                              col.toLowerCase().includes("frequency") ||
                              col.toLowerCase().includes("score") ? (
                                <div className="h-1.5 flex-1 rounded bg-sparc-gray-100">
                                  <div
                                    className="h-full rounded bg-sparc-purple/40"
                                    style={{ width: `${barWidth}%` }}
                                  />
                                </div>
                              ) : null}
                            </div>
                          ) : (
                            <span className={isSelected ? "font-medium text-sparc-purple" : ""}>
                              {val == null ? "—" : String(val)}
                            </span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
