# Tasks: Run Page Terminal Verbosity & Messaging Cleanup

Related PRD: docs/prd/prd-run-terminal-verbosity.md

## Tasks

- [x] Write PRD
- [~] Implement changes in `sparc-desktop/src/components/pages/RunPage.tsx`
  - Add `"log"` color to `LEVEL_COLORS`
  - Rewrite all `eventToLogLine` message strings to consistent prefix+prose format
  - Change final `return null` to emit raw `log` events at `"log"` level
  - Update verbosity filter: `normal` excludes `log`; `debug` passes all
- [ ] Validate: no TypeScript errors, tiers visually distinct
