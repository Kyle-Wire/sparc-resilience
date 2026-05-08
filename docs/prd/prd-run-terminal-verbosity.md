# PRD: Run Page Terminal Verbosity & Messaging Cleanup

## Problem Statement

The Run page terminal output has three verbosity tiers (Summary, Normal, Debug) that don't meaningfully differ. Debug looks identical to Normal because raw `log` events are always silently dropped. Message strings are inconsistent — mixing emoji, ASCII art separators, bare key-value pairs, and informal prose — which looks unprofessional.

## Goals

- Make each tier visually and informationally distinct
- Debug: true full terminal output including raw unmatched pipeline stdout
- Normal: structured updates without low-level noise
- Summary: high-level pass/fail only
- Rewrite all message strings to a consistent, professional format

## Non-Goals

- No backend changes — `SPARC_VERBOSITY` env var and `console.py` are untouched
- No changes to the WebSocket event stream — backend always streams everything
- No new event types or backend API changes
- No changes to other pages

## Design Decisions

### Tier definitions

| Tier    | Shows                                                                 |
|---------|-----------------------------------------------------------------------|
| Summary | `milestone`, `error`, `warn`, `success`                              |
| Normal  | All structured events except `epoch_update`, `capacity_result`, `progress`, `log` |
| Debug   | All structured events + raw `log` type events                        |

The key change: `log` events (type === "log") are currently dropped in `eventToLogLine` with `return null`. In Debug mode they must be rendered, tagged with level `"log"` and shown in a dim color.

### Messaging format

Replace ad-hoc symbols with a consistent prefix + sentence-case prose scheme:

- Stage lifecycle: `[STAGE] Starting <Name> — <description>` / `✓ <Name> completed (12.3s)` / `✕ <Name> failed: <reason>`
- Fold boundaries: `[FOLD] Fold 2 / 5  —  1,200 train  /  300 test`
- Fold completion: `✓ Fold 2 complete  (45s)`
- Model training: `[MODEL] Training OLS...` / `✓ OLS complete  R²=0.7234`
- Metrics: `[METRIC] OLS  fold 2  —  metric = 0.1234`
- Epoch updates: `[EPOCH] 3 / 50  loss=0.4231  eta=120s` (Debug / Normal only — already debug level)
- Convergence: `[INFO] Convergence: good`
- Curriculum: `[INFO] Curriculum phase → <label>`
- Capacity: `[INFO] Capacity check  dim=128  R²=0.7234`
- Errors: `[ERROR] <message>`
- DAG gate: `[WARN] DAG approval required — review on the DAG page`
- Health: `[WARN] <warning>`
- Pipeline complete: `✓ Pipeline complete`
- Checkpoints: keep existing message text, ensure level maps correctly

### Color map additions

Add `"log"` entry to `LEVEL_COLORS` at `#607d8b` (dimmer than existing `debug` at `#78909c`) so raw log lines visually recede behind structured output.

## Technical Approach

Single file change: `sparc-desktop/src/components/pages/RunPage.tsx`

1. Add `"log": "#607d8b"` to `LEVEL_COLORS`
2. Rewrite `eventToLogLine`:
   - Change all message strings to new format
   - Change final `return null` to `return { text: (evt as any).message ?? "", level: "log" as const, ts }` for raw log events
3. Update verbosity filter:
   - `debug`: `return true` (already correct)
   - `normal`: exclude `level === "debug" || level === "log"` (add `log`)
   - `summary`: keep as-is (`["milestone","error","warn","success"].includes(line.level)`)

## Acceptance Criteria

- [ ] Summary shows only stage pass/fail, errors, warnings — nothing else
- [ ] Normal shows all structured events; fold detail, model metrics visible; no raw log lines
- [ ] Debug shows everything including raw stdout lines in dim color
- [ ] All message text uses bracketed prefix + sentence-case prose
- [ ] `✓`/`✕` used for pass/fail outcomes only
- [ ] No TypeScript errors introduced
- [ ] Backend (`console.py`, `stream.py`) unchanged

## Open Questions

None — all decisions resolved.
