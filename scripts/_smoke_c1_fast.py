"""Fast C1 smoke test — no top-level pandas import (avoids torch DLL scan)."""
import sys, tempfile, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

with tempfile.TemporaryDirectory() as tmp:
    from sparc.registry.run_registry import RunRegistry
    reg = RunRegistry(pathlib.Path(tmp), autoload=False)

    # ---- blob round-trip ----
    data = b"hello blob"
    reg.write_blob_payload("0", "blob1", data, serializer="raw", mime=None, sha="abc123")
    result = reg.read_blob_payload("0", "blob1")
    assert result == data, f"blob round-trip failed: {result!r}"
    print("blob payload: OK")

    # ---- struct round-trip ----
    reg.write_struct_payload("1", "struct1", '{"key": "value"}')
    result2 = reg.read_struct_payload("1", "struct1")
    assert result2 == '{"key": "value"}', f"struct round-trip failed: {result2!r}"
    print("struct payload: OK")

    # ---- list_blob_shas ----
    shas = reg.list_blob_shas()
    assert isinstance(shas, set), f"expected set, got {type(shas)}"
    print(f"list_blob_shas: OK (found {len(shas)} shas)")

    # ---- private seam ----
    public = [n for n in dir(reg) if not n.startswith("_")]
    assert "sqlite_connection" not in public, "sqlite_connection still public!"
    assert hasattr(reg, "_sqlite_connection"), "_sqlite_connection missing"
    print("private seam: OK")

    # ---- read_blob_payload returns None for missing ----
    missing = reg.read_blob_payload("99", "does_not_exist")
    assert missing is None, f"expected None, got {missing!r}"
    print("missing blob returns None: OK")

    # ---- read_struct_payload returns None for missing ----
    missing2 = reg.read_struct_payload("99", "does_not_exist")
    assert missing2 is None, f"expected None, got {missing2!r}"
    print("missing struct returns None: OK")

    print("ALL C1 FAST SMOKE TESTS PASSED")
