"""
Fast structural verification for C1 + C2 using ast.parse (no imports, no DLL scan).
Checks that all expected method signatures are present in the source files.
"""
import ast, sys, pathlib

ROOT = pathlib.Path(__file__).parent.parent

def parse(rel: str) -> ast.Module:
    src = (ROOT / rel).read_text(encoding="utf-8")
    return ast.parse(src)

def methods_in(tree: ast.Module, class_name: str | None = None) -> set[str]:
    """Return set of function/method names at module level or in a given class."""
    if class_name is None:
        return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    return set()

def public_methods(tree: ast.Module, class_name: str) -> set[str]:
    return {n for n in methods_in(tree, class_name) if not n.startswith("_")}

# ── C1: RunRegistry typed seam ────────────────────────────────────────────
rr_tree = parse("sparc/registry/run_registry.py")
rr_methods = methods_in(rr_tree, "RunRegistry")

for m in ("write_df", "read_df", "write_struct_payload", "read_struct_payload",
          "write_blob_payload", "read_blob_payload", "list_blob_shas"):
    assert m in rr_methods, f"RunRegistry.{m} not found"
print("C1: 7 typed methods present in RunRegistry: OK")

# _sqlite_connection must exist and sqlite_connection must NOT exist as public
assert "_sqlite_connection" in rr_methods, "_sqlite_connection not found"
assert "sqlite_connection" not in public_methods(rr_tree, "RunRegistry"), \
    "sqlite_connection still public!"
print("C1: sqlite_connection is private: OK")

all_funcs = {n.name for n in ast.walk(rr_tree) if isinstance(n, ast.FunctionDef)}
assert "_utcnow" in all_funcs, "_utcnow not defined in run_registry.py"
print("C1: _utcnow helper present: OK")

# store.py must NOT import sqlite3 and must NOT define _utcnow
store_tree = parse("sparc/registry/store.py")
store_funcs = {n.name for n in ast.walk(store_tree) if isinstance(n, ast.FunctionDef)}
store_imports = {n.names[0].name if isinstance(n, ast.Import) else n.module
                 for n in ast.walk(store_tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
assert "sqlite3" not in store_imports, "store.py still imports sqlite3"
assert "_utcnow" not in store_funcs, "store.py still defines _utcnow (should have been removed)"
print("C1: store.py has no sqlite3 import and no _utcnow: OK")

# render.py must NOT have sqlite_connection calls
render_src = (ROOT / "sparc/report/render.py").read_text(encoding="utf-8")
assert "sqlite_connection" not in render_src, "render.py still calls sqlite_connection"
print("C1: render.py clean of sqlite_connection: OK")

# ── C2: ServerState split ──────────────────────────────────────────────────
ps_tree = parse("sparc/server/project_session.py")
ps_methods = methods_in(ps_tree, "ProjectSession")
for m in ("load", "clear", "get_store"):
    assert m in ps_methods, f"ProjectSession.{m} not found"
print("C2: ProjectSession has load/clear/get_store: OK")

ec_tree = parse("sparc/server/execution_context.py")
ec_methods = methods_in(ec_tree, "ExecutionContext")
for m in ("set_running", "set_idle", "buffer_event", "get_buffered_events",
          "set_dag_approval", "wait_for_dag_approval", "clear"):
    assert m in ec_methods, f"ExecutionContext.{m} not found"
print("C2: ExecutionContext has 7 methods: OK")

deps_src = (ROOT / "sparc/server/deps.py").read_text(encoding="utf-8")
assert "ProjectSession" in deps_src, "deps.py missing ProjectSession"
assert "ExecutionContext" in deps_src, "deps.py missing ExecutionContext"
assert "session: ProjectSession" in deps_src or "session = ProjectSession" in deps_src, \
    "deps.py missing session instance"
assert "execution: ExecutionContext" in deps_src or "execution = ExecutionContext" in deps_src, \
    "deps.py missing execution instance"
print("C2: deps.py wired correctly: OK")

print("\n=== ALL STRUCTURAL VERIFICATIONS PASSED ===")
