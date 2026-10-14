// Pipeline Status Board (SPEC §5.12): one row per run (mode, label, created, status) and one
// column per stage, post-run action and study. Each cell is a status chip (icon + text): done
// (seconds), cached, running (%), failed, skipped (reason), stale, not run (+ launch). Clicking a
// chip opens the tracker span or the analysis view. While a cell runs the board is refetched
// every 10 s, so its percentage keeps moving between job status changes.
import { useEffect, useRef, useState } from "react";
import { useResource } from "../../api/resource";
import { getStatusBoard, type StatusBoard as Board } from "../../api/tracking";
import { ModeBadge } from "../../layouts/tips";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { Link } from "../../router";
import { fmtDate } from "../../theme/format";
import { boardCellChip, boardCellHref, boardGroups } from "./model";

/** True while a horizontal scroller has columns hidden past its right edge (drives the edge fade). */
function useMoreRight() {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(update) : null;
    if (ro) {
      ro.observe(el);
      if (el.firstElementChild) ro.observe(el.firstElementChild);
    }
    return () => {
      el.removeEventListener("scroll", update);
      ro?.disconnect();
    };
  }, []);
  return [ref, more] as const;
}

export function StatusBoardTable({ board }: { board: Board }) {
  const groups = boardGroups(board);
  const [scroller, more] = useMoreRight();
  const starts = new Set(groups.map((g) => g.columns[0]?.id));
  if (!board.rows.length) return <EmptyState title="No runs yet" body="The board fills in as runs are launched or imported." />;
  return (
    <div className="board-frame" data-more={more || undefined}>
      <div className="board-wrap" ref={scroller}>
        <table className="board" aria-label="Status of every run">
          <thead>
            <tr className="groups">
              <th scope="col">
                <span className="sr-only">Run</span>
              </th>
              {groups.map((g) => (
                <th key={g.group} scope="colgroup" colSpan={g.columns.length} className="group-start">
                  {g.label}
                </th>
              ))}
            </tr>
            <tr>
              <th scope="col">Run</th>
              {board.columns.map((c) => (
                <th key={c.id} scope="col" className={starts.has(c.id) ? "group-start" : undefined}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {board.rows.map(({ run, cells }) => (
              <tr key={run.id} data-run={run.id}>
                <th scope="row">
                  <span className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                    <ModeBadge mode={run.mode} coarse_m={run.coarse_m} />
                    <Link to={`/r/${encodeURIComponent(run.id)}`}>{run.label || run.id}</Link>
                    <span className="cap">{fmtDate(run.created_utc)}</span>
                    <StatusChip status={run.status} />
                  </span>
                </th>
                {board.columns.map((c) => {
                  const cell = cells[c.id];
                  const chip = boardCellChip(cell);
                  const href = boardCellHref(run.id, c.id, c.group, cell);
                  const title = `${run.label || run.id} · ${c.label}: ${chip.text}${chip.meta ? ` (${chip.meta})` : ""}`;
                  // the cell's row and column headers say which run and step it is (the link's name says it too)
                  const chipEl = <StatusChip status={chip.status} text={chip.text} meta={chip.meta} />;
                  return (
                    <td key={c.id} className={starts.has(c.id) ? "group-start" : undefined} data-col={c.id} data-state={cell?.state ?? "not_run"}>
                      <span className="cell-actions">
                        {href ? (
                          <Link to={href} className="cell-btn" aria-label={title}>
                            {chipEl}
                          </Link>
                        ) : (
                          chipEl
                        )}
                        {cell?.state === "not_run" && cell.action ? <ActionButton action={cell.action} size="small" variant="default" /> : null}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Running cells show a stage's progress, which no global event carries: the board refreshes this often while any cell runs. */
export const BOARD_REFRESH_MS = 10_000;

/** The board of a project (`GET /api/projects/{pid}/status-board`). */
export function StatusBoard({ pid }: { pid: string }) {
  const res = useResource(`project:${pid}:status-board`, (s) => getStatusBoard(pid, s), { tags: [`project:${pid}`, "runs", "jobs", "studies"], keepPrevious: true });
  const running = !!res.data?.rows.some((r) => Object.values(r.cells).some((c) => c.state === "running"));
  const reload = res.reload;
  useEffect(() => {
    if (!running) return;
    const h = setInterval(() => void reload(), BOARD_REFRESH_MS);
    return () => clearInterval(h);
  }, [running, reload]);
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  if (!res.data) return <p className="cap">Loading the status board…</p>;
  return <StatusBoardTable board={res.data} />;
}
