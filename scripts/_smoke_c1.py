"""Quick smoke test for C1 RunRegistry typed query seam."""
import sys, tempfile, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pandas as pd

with tempfile.TemporaryDirectory() as tmp:
    from sparc.registry.run_registry import RunRegistry
    reg = RunRegistry(pathlib.Path(tmp), autoload=False)

    # write_df / read_df
    df = pd.DataFrame({"a": [1, 2, 3]})
    reg.write_df("test_table", df)
    result = reg.read_df("test_table")
    assert list(result["a"]) == [1, 2, 3], "write_df/read_df failed"
    print("write_df/read_df: OK")

    # write_blob_payload / read_blob_payload
    data = b"hello blob"
    reg.write_blob_payload("0", "blob1", data, serializer="raw", mime=None, sha="abc")
    result2 = reg.read_blob_payload("0", "blob1")
    assert result2 == data, "blob round-trip failed"
    print("blob payload: OK")

    # write_struct_payload / read_struct_payload
    reg.write_struct_payload("1", "struct1", "test_json")
    result3 = reg.read_struct_payload("1", "struct1")
    assert result3 == "test_json", "struct round-trip failed"
    print("struct payload: OK")

    # sqlite_connection is private
    public = [n for n in dir(reg) if not n.startswith("_")]
    assert "sqlite_connection" not in public, f"sqlite_connection still public"
    assert hasattr(reg, "_sqlite_connection"), "_sqlite_connection missing"
    print("private seam: OK")

    # list_blob_shas
    shas = reg.list_blob_shas()
    assert isinstance(shas, set)
    print("list_blob_shas: OK")

    # ArtifactStore regression
    from sparc.registry.store import ArtifactStore
    store = ArtifactStore(reg)
    df2 = pd.DataFrame({"score": [0.1, 0.2], "label": ["a", "b"]})
    store.write_table("0", "regression_table", df2)
    out = store.read_table("0", "regression_table")
    assert list(out["label"]) == ["a", "b"]
    print("ArtifactStore regression: OK")

    print("ALL C1 SMOKE TESTS PASSED")
