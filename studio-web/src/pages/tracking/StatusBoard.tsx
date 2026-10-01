// Pipeline Status Board (SPEC §5.12): one row per run (mode, label, created, status) and one
// column per stage, post-run action and study. Each cell is a status chip (icon + text): done
// (seconds), cached, running (%), failed, skipped (reason), stale, not run (+ launch). Clicking a
// chip opens the tracker span or the analysis view.
import { useResource } from "../../api/resource";
import { getStatusBoard, type StatusBoard as Board } from "../../api/tracking";
import { Badge, modeLabel } from "../../components/ui/Badge";
import { ActionButton, EmptyState } from "../../components/ui/EmptyState";
import { StatusChip } from "../../components/ui/StatusChip";
import { Link } from "../../router";
import { fmtDate } from "../../theme/format";
import { boardCellChip, boardCellHref, boardGroups } from "./model";

export function StatusBoardTable({ board }: { board: Board }) {
  const groups = boardGroups(board);
  const starts = new Set(groups.map((g) => g.columns[0]?.id));
  if (!board.rows.length) return <EmptyState title="No runs yet" body="The board fills in as runs are launched or imported." />;
  return (
    <div className="board-wrap">
      <table className="board" aria-label="Pipeline status board">
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
                  <Badge tone="accent">{modeLabel(run.mode, run.coarse_m)}</Badge>
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
                const chipEl = <StatusChip status={chip.status} text={chip.text} meta={chip.meta} title={title} />;
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
  );
}

/** The board of a project (`GET /api/projects/{pid}/status-board`). */
export function StatusBoard({ pid }: { pid: string }) {
  const res = useResource(`project:${pid}:status-board`, (s) => getStatusBoard(pid, s), { tags: [`project:${pid}`, "runs", "jobs", "studies"], keepPrevious: true });
  if (res.error && !res.data) return <EmptyState error={res.error} />;
  if (!res.data) return <p className="cap">Loading the status board…</p>;
  return <StatusBoardTable board={res.data} />;
}
