"""SPARC Studio: the local web app for the ``sparc/core`` pipeline.

``sparc studio`` (or ``python -m sparc.studio``) starts one FastAPI process
that indexes projects and runs in SQLite, runs every slow operation as a
tracked job in its own process group, and serves the React SPA built from
``studio-web/``.  See ``docs/studio/SPEC.md`` (behaviour) and
``docs/studio/api.md`` (wire contract).

Importing this package is cheap: the server modules (FastAPI, uvicorn) are
imported by :mod:`sparc.studio.app` and :mod:`sparc.studio.cli` only, and the
server process never imports torch.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
