"""Plain words for SPARC's technical terms (``glossary.json``), for copy the server writes.

The same file feeds SPARC Studio (``studio-web/src/content/glossary.ts``) and the results page, so a label
reads the same wherever it appears.  ``term("r2")`` → ``{label, tip, technical?, more?}``;
``label("r2")`` → ``"Accuracy score"``; ``label("r2", lower=True)`` → ``"accuracy score"`` for mid-sentence use.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

__all__ = ["glossary", "term", "label", "section"]


@lru_cache(maxsize=1)
def glossary() -> dict:
    """The whole glossary document (read once)."""
    return json.loads(resources.files("sparc.core").joinpath("glossary.json").read_text("utf-8"))


def term(tid: str) -> dict:
    """The term ``tid``: ``{label, tip, technical?, more?}``.  ``KeyError`` for an unknown id."""
    return glossary()["terms"][tid]


def label(tid: str, *, lower: bool = False) -> str:
    """The plain label of term ``tid``; ``lower`` lower-cases its first letter (for use inside a sentence)."""
    s = term(tid)["label"]
    return s[:1].lower() + s[1:] if lower else s


def section(name: str) -> dict:
    """A fixed section (``emissions``, ``periods``, ``heat``, ``trust`` …); ``{}`` when absent."""
    v = glossary().get(name)
    return v if isinstance(v, dict) else {}
