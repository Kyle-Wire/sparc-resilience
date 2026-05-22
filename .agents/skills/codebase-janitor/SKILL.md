---
name: codebase-janitor
description: Audits a codebase to safely remove dead code, unused imports, and orphaned files, then reorganizes scaffolding for logical clarity. Guarantees no functional code is deleted and that all import paths remain valid after moves. Use when user wants to remove dead code, clean up unused imports, reorganize folder structure, prune scaffolding, or says "clean up the codebase", "remove dead code", "delete unused files", "reorganize modules", "tidy up", or invokes /janitor.
---

# Codebase Janitor

Safely removes dead code and reorganizes scaffolding. The two cardinal rules:

1. **Never delete functional code.** When in doubt, keep it.
2. **Never break an import.** Every path change must be accompanied by a reference update.

## Workflows

### Workflow A — Dead Code Removal

1. **Map the entry points** — identify all executables, public APIs, test runners, and CLI scripts. These are the roots; everything reachable from them is live.

2. **Detect candidates** using static analysis (grep/AST):
   - Unused imports (`import X` but `X` never referenced in the file)
   - Unreferenced functions/classes/variables (defined but never called)
   - Orphaned files (not imported by anything, not an entry point)
   - Commented-out blocks older than one logical feature

3. **Verify each candidate** before removal:
   - Run `grep -r "symbol_name"` across the whole repo — a hit anywhere means it's live
   - Check dynamic usage: `getattr`, `importlib`, string-based lookups, plugin registries
   - Check test files — a symbol tested but not used in production is still functional code
   - Check config/YAML files that may reference module paths by string

4. **Batch removals by risk tier**:
   - **Safe** — unused imports, unreachable branches confirmed by static analysis
   - **Review** — functions with no callers but plausible future use; present to user before deleting
   - **Skip** — anything dynamically loaded, exported as a public API, or referenced by string

5. **Run the test suite** after each batch. If anything breaks, revert the batch and triage individually.

---

### Workflow B — Scaffolding Reorganization

1. **Understand the intended structure** — ask the user for the target layout, or infer it from existing conventions (`docs/`, `scripts/`, `tests/`, module namespaces).

2. **Produce a move plan** as a table before touching anything:

   | Current path | New path | Reason |
   |---|---|---|
   | `src/utils/foo.py` | `src/io/foo.py` | belongs with I/O helpers |

   Present the plan and get approval before moving files.

3. **Update import references** before or immediately after each move:
   - Search for all `import` and `from … import` statements referencing the old path
   - Search for string-based references (`"src.utils.foo"`, `"utils/foo"`)
   - Update `__init__.py` re-exports
   - Update any `pyproject.toml`, `setup.cfg`, `tsconfig.json`, `package.json` path mappings

4. **Verify with a dry-run** — run the test suite (or at minimum `python -c "import <module>"` / `tsc --noEmit`) to confirm nothing is broken.

5. **Commit atomically** — move + reference updates in a single logical change so git history stays coherent.

---

### Workflow C — Full Audit Report

When asked for a full audit rather than targeted cleanup, produce a self-contained HTML report in the OS temp dir:

- Resolve temp dir from `$TMPDIR` → `/tmp` → `%TEMP%` (Windows)
- Write to `<tmpdir>/janitor-audit-<timestamp>.html`
- Open it: `xdg-open` (Linux), `open` (macOS), `Start-Process` (Windows)

Report sections:
- **Dead imports** — file, line, symbol
- **Unreferenced symbols** — name, file, confidence (`certain` / `likely` / `possible`)
- **Orphaned files** — path, last-modified, why it looks orphaned
- **Structural suggestions** — files that belong elsewhere, with rationale
- **Risk summary** — counts by tier (Safe / Review / Skip)

Each finding has a **Keep / Remove / Move** recommendation with one-line justification.

---

## Safety Checklist

Before finalizing any removal or move:

- [ ] Grep confirms zero references to deleted symbol outside its own file
- [ ] No string-based dynamic load paths match the symbol/path
- [ ] Test suite passes after change
- [ ] `__init__.py` and re-export files updated
- [ ] Config files (pyproject, tsconfig, package.json) checked for path references
- [ ] Git diff reviewed — no accidental collateral deletions

## Anti-Patterns to Avoid

- **Bulk deletes without verification** — always verify each symbol individually
- **Moving files without updating imports first** — broken imports are harder to debug than messy structure
- **Deleting "obviously unused" test helpers** — they may be fixtures or shared utilities
- **Assuming a symbol is dead because grep returns one result** — that one result might be a re-export
