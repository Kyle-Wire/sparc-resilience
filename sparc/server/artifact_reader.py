"""Artifact batch reader — the single deep module behind /results/batch.

All logic for resolving, reading, and error-shaping artifact lookups lives
here.  The 20+ individual ``/results/*`` endpoints are thin wrappers that
call this function with a known (stage, artifact_id) pair.
"""

from __future__ import annotations

from typing import Any, TypedDict

from sparc.registry.manifest import RunManifest
from sparc.registry.store import ArtifactStore


class MissingDetail(TypedDict):
    id: str
    stage: str
    hint: str


class BatchResult(TypedDict):
    results: dict[str, Any]
    missing: list[MissingDetail]


def read_batch(
    store: ArtifactStore,
    requests: list[tuple[str, str]],
    *,
    hint_map: dict[tuple[str, str], str] | None = None,
) -> BatchResult:
    """Read a batch of artifacts from *store*.

    Parameters
    ----------
    store:
        The active :class:`~sparc.registry.store.ArtifactStore`.
    requests:
        Ordered list of ``(stage, artifact_id)`` pairs to resolve.
    hint_map:
        Optional mapping of ``(stage, artifact_id)`` to a human-readable hint
        string included in the missing-artifact detail.  When absent the hint
        defaults to a generic message.

    Returns
    -------
    BatchResult
        ``results`` maps ``"stage:artifact_id"`` to the deserialized payload.
        ``missing`` lists every artifact that could not be resolved.
    """
    results: dict[str, Any] = {}
    missing: list[MissingDetail] = []

    for stage, artifact_id in requests:
        key = f"{stage}:{artifact_id}"

        # Check the registry entry directly so we can inspect partial flag.
        entry = store.registry.lookup(stage, artifact_id)
        if entry is None or entry.partial:
            hint = (hint_map or {}).get(
                (stage, artifact_id),
                f"Stage {stage} has not produced {artifact_id!r}.",
            )
            missing.append(MissingDetail(id=artifact_id, stage=str(stage), hint=hint))
            continue

        results[key] = store.read_any(stage, artifact_id)

    return BatchResult(results=results, missing=missing)


def prewarm_ids(manifest: RunManifest) -> list[tuple[str, str]]:
    """Return all ``(stage, artifact_id)`` pairs eligible for pre-warming.

    Includes every complete (non-partial) artifact in the manifest regardless
    of its ``consumers`` tags.  Extracted from the pre-warm thread so it can
    be tested without FastAPI module globals.
    """
    return [
        (entry.stage, entry.id)
        for entry in manifest.all_artifacts()
        if not entry.partial
    ]
