"""SPARC Streamlit UI package.

.. deprecated:: 2.0.0
    The Streamlit UI is deprecated in favor of the SPARC Desktop App
    (Tauri + React). Use ``sparc server`` to start the FastAPI backend
    and open the desktop application, or run ``sparc desktop`` to launch
    the native app directly.
"""
import warnings as _warnings

_warnings.warn(
    "sparc.ui (Streamlit) is deprecated. Use the SPARC Desktop App instead: "
    "run 'sparc server' + the Tauri desktop app, or 'sparc desktop'.",
    DeprecationWarning,
    stacklevel=2,
)
