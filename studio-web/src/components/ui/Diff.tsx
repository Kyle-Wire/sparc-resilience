import { useMemo } from "react";

export type DiffLine = { op: "ctx" | "add" | "del"; text: string; a?: number; b?: number };

/**
 * Line diff (longest common subsequence). Inputs above 4,000 lines fall back to a
 * prefix/suffix trim plus a block replace, which keeps it fast on big files.
 */
export function diffLines(a: string, b: string): DiffLine[] {
  const A = a === "" ? [] : a.split("\n");
  const B = b === "" ? [] : b.split("\n");
  let pre = 0;
  while (pre < A.length && pre < B.length && A[pre] === B[pre]) pre++;
  let suf = 0;
  while (suf < A.length - pre && suf < B.length - pre && A[A.length - 1 - suf] === B[B.length - 1 - suf]) suf++;
  const out: DiffLine[] = [];
  for (let i = 0; i < pre; i++) out.push({ op: "ctx", text: A[i], a: i + 1, b: i + 1 });
  const a0 = A.slice(pre, A.length - suf);
  const b0 = B.slice(pre, B.length - suf);
  if (a0.length * b0.length > 16_000_000) {
    a0.forEach((t, i) => out.push({ op: "del", text: t, a: pre + i + 1 }));
    b0.forEach((t, i) => out.push({ op: "add", text: t, b: pre + i + 1 }));
  } else {
    const n = a0.length;
    const m = b0.length;
    const L = new Uint32Array((n + 1) * (m + 1));
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--) L[i * (m + 1) + j] = a0[i] === b0[j] ? L[(i + 1) * (m + 1) + j + 1] + 1 : Math.max(L[(i + 1) * (m + 1) + j], L[i * (m + 1) + j + 1]);
    let i = 0;
    let j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && a0[i] === b0[j]) {
        out.push({ op: "ctx", text: a0[i], a: pre + i + 1, b: pre + j + 1 });
        i++;
        j++;
      } else if (j < m && (i >= n || L[i * (m + 1) + j + 1] >= L[(i + 1) * (m + 1) + j])) {
        out.push({ op: "add", text: b0[j], b: pre + j + 1 });
        j++;
      } else {
        out.push({ op: "del", text: a0[i], a: pre + i + 1 });
        i++;
      }
    }
  }
  for (let k = 0; k < suf; k++) out.push({ op: "ctx", text: A[A.length - suf + k], a: A.length - suf + k + 1, b: B.length - suf + k + 1 });
  return out;
}

type Row = DiffLine | { op: "skip"; count: number };

/** Collapse unchanged runs longer than 2·context into "… N unchanged lines". */
export function collapseContext(lines: DiffLine[], context: number): Row[] {
  const rows: Row[] = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].op !== "ctx") {
      rows.push(lines[i++]);
      continue;
    }
    let j = i;
    while (j < lines.length && lines[j].op === "ctx") j++;
    const run = lines.slice(i, j);
    const head = i === 0 ? 0 : context;
    const tail = j === lines.length ? 0 : context;
    if (run.length > head + tail + 1) {
      rows.push(...run.slice(0, head));
      rows.push({ op: "skip", count: run.length - head - tail });
      rows.push(...run.slice(run.length - tail));
    } else rows.push(...run);
    i = j;
  }
  return rows;
}

/** Unified diff view; changes are marked with +/− as well as colour. */
export function Diff({ a, b, context = 3, label = "Changes" }: { a: string; b: string; context?: number; label?: string }) {
  const rows = useMemo(() => collapseContext(diffLines(a, b), context), [a, b, context]);
  const changed = rows.some((r) => r.op === "add" || r.op === "del");
  return (
    <div className="diff" role="region" aria-label={label}>
      {!changed ? <div className="skip">No changes</div> : null}
      {changed
        ? rows.map((r, k) =>
            r.op === "skip" ? (
              <div key={k} className="skip">
                <span />
                <span>… {r.count} unchanged lines</span>
              </div>
            ) : (
              <div key={k} className={r.op}>
                <span aria-hidden="true">{r.op === "add" ? "+" : r.op === "del" ? "−" : " "}</span>
                <span>
                  <span className="sr-only">{r.op === "add" ? "added: " : r.op === "del" ? "removed: " : ""}</span>
                  {r.text || " "}
                </span>
              </div>
            ),
          )
        : null}
    </div>
  );
}
