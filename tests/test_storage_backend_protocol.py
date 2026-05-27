"""TDD — C1: StorageBackend protocol as a formal seam.

Verifies that:
1. ``StorageBackend`` protocol is importable from ``sparc.registry``
2. ``MemoryStore`` satisfies the protocol (runtime_checkable isinstance)
3. Any object with the required methods is a valid StorageBackend
4. The protocol is exported from the registry package
"""

from __future__ import annotations

import pytest


class TestStorageBackendProtocol:
    def test_protocol_importable_from_backend_module(self):
        from sparc.registry.backend import StorageBackend
        assert StorageBackend is not None

    def test_protocol_importable_from_registry_package(self):
        from sparc.registry import StorageBackend
        assert StorageBackend is not None

    def test_memory_store_satisfies_protocol(self):
        from sparc.registry.backend import StorageBackend
        from sparc.registry.memory_store import MemoryStore
        store = MemoryStore()
        assert isinstance(store, StorageBackend)

    def test_memory_store_exported_from_registry_package(self):
        from sparc.registry import MemoryStore
        assert MemoryStore is not None

    def test_protocol_has_required_write_methods(self):
        from sparc.registry.backend import StorageBackend
        required = {"write_struct", "write_blob", "write_table"}
        for method in required:
            assert hasattr(StorageBackend, method), f"StorageBackend missing {method}"

    def test_protocol_has_required_read_methods(self):
        from sparc.registry.backend import StorageBackend
        required = {"read_struct", "read_blob", "read_table", "read_any"}
        for method in required:
            assert hasattr(StorageBackend, method), f"StorageBackend missing {method}"

    def test_non_conforming_object_is_not_storage_backend(self):
        from sparc.registry.backend import StorageBackend
        class NotAStore:
            pass
        assert not isinstance(NotAStore(), StorageBackend)

    def test_memory_store_roundtrip_through_protocol(self):
        """Any caller that accepts StorageBackend can use MemoryStore."""
        from sparc.registry.backend import StorageBackend
        from sparc.registry.memory_store import MemoryStore

        def write_and_read(store: StorageBackend) -> dict:
            store.write_struct("stage1", "metrics", {"rmse": 0.5})
            return store.read_struct("stage1", "metrics")

        result = write_and_read(MemoryStore())
        assert result["rmse"] == 0.5

    def test_conforming_custom_class_satisfies_protocol(self):
        """Any class with the correct methods satisfies StorageBackend."""
        from sparc.registry.backend import StorageBackend

        class MinimalStore:
            def write_struct(self, stage, artifact_id, payload, **_): pass
            def read_struct(self, stage, artifact_id): return {}
            def write_blob(self, stage, artifact_id, obj, **_): pass
            def read_blob(self, stage, artifact_id, **_): return None
            def write_table(self, stage, artifact_id, df, **_): return ""
            def read_table(self, stage, artifact_id): return None
            def read_any(self, stage, artifact_id, **_): return None
            def list_artifacts(self, stage=None): return []

        assert isinstance(MinimalStore(), StorageBackend)
