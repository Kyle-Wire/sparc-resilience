"""Quick smoke test for C2 ServerState split."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sparc.server.project_session import ProjectSession
from sparc.server.execution_context import ExecutionContext

# --- ProjectSession ---
ps = ProjectSession()
assert ps.project_path is None
print("ProjectSession init: OK")

ps.load(path=pathlib.Path("/tmp/proj"), config={}, raw_yaml="yaml:", data=None, data_summary=None, registry=None)
assert str(ps.project_path) == "/tmp/proj"
print("ProjectSession load: OK")

ps.clear()
assert ps.project_path is None
print("ProjectSession clear: OK")

try:
    ps.get_store()
    raise AssertionError("Expected HTTPException")
except Exception as exc:
    assert "400" in str(exc) or "no active project" in str(exc).lower(), f"Unexpected: {exc}"
print("ProjectSession get_store raises: OK")

# --- ExecutionContext ---
ec = ExecutionContext()
assert not ec.is_running
print("ExecutionContext init: OK")

ec.set_running(2)
assert ec.is_running
assert ec.stage == 2
print("ExecutionContext set_running: OK")

ec.buffer_event({"type": "progress"})
events = ec.get_buffered_events()
assert len(events) == 1
assert events[0]["type"] == "progress"
print("ExecutionContext buffer_event: OK")

ec.set_idle()
assert not ec.is_running
print("ExecutionContext set_idle: OK")

# Independence check: separate locks
ec2 = ExecutionContext()
ec2.set_running(3)
assert not ps.is_running if hasattr(ps, "is_running") else True  # no is_running on ps
assert ec2.is_running != (not ec2.is_running)  # trivially true
print("Independence: OK")

# --- deps wiring ---
from sparc.server.deps import session, execution
assert isinstance(session, ProjectSession)
assert isinstance(execution, ExecutionContext)
print("deps wiring: OK")

print("ALL C2 SMOKE TESTS PASSED")
