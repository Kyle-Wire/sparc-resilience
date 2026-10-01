"""Router registry (SPEC §10.2).

``create_app`` imports each ``sparc.studio.routes.<name>`` of
:data:`ROUTER_MODULES` and mounts its ``router`` under ``/api`` (and its
optional ``root_router`` at ``/``).  A module that does not exist yet
(``ModuleNotFoundError`` whose ``.name`` is exactly that module) is skipped
with a debug log; any other import error is raised.  Feature items add
routes by creating their module - they never edit this file.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

log = logging.getLogger("sparc.studio")

__all__ = ["ROUTER_MODULES", "load_router_modules"]

ROUTER_MODULES = ["system", "jobs", "stream", "projects", "inputs", "runs", "layers", "compare", "lab",
                  "plans", "studies", "exports", "findings"]


def load_router_modules(names: list[str] | None = None) -> list[ModuleType]:
    """Import the route modules that exist, in registry order."""
    out = []
    for name in names or ROUTER_MODULES:
        mod_name = f"{__name__}.{name}"
        try:
            out.append(importlib.import_module(mod_name))
        except ModuleNotFoundError as exc:
            if exc.name != mod_name:
                raise
            log.debug("route module %s not present; skipped", mod_name)
    return out
