"""Runs: the registry of run directories, reading their outputs, grids, layers, selections, analysis tools,
run comparison, launch/resume and the ``run.core`` / ``run.external`` job kinds (SPEC §4.3, §5.9, §5.12,
§5.14, §6; api.md §6).

Modules:

* :mod:`.registry` - index runs (workspace, imports, watch roots), status, import in place, external runs;
* :mod:`.reader` - :class:`~.reader.RunContext`: merged manifest, sections, data, grid, folds, live metrics;
* :mod:`.grid` - the run grid, ``grid.npz`` cache, ``grid.bin`` / ``ids.bin`` and ``GridMeta``;
* :mod:`.layers` - the generated layer catalog and per-layer arrays;
* :mod:`.outputs` - output states, run-tab availability, docs, files, conversions, the data dictionary;
* :mod:`.views` - ViewModel builders of the run-hub tabs;
* :mod:`.selection` - ``resolve(ctx, spec)`` of every SelectionSpec kind, bitsets, regions and blobs;
* :mod:`.stats` - region stats (fold jackknife), breakdown, hexbin, correlogram, hex aggregates;
* :mod:`.launch` - plan, preflight, launch, resume and rerun;
* :mod:`.kinds` - the ``run.core`` and ``run.external`` job kinds;
* :mod:`.statusboard` - the Pipeline Status Board and the run timeline;
* :mod:`.compare` - run comparison and priority agreement;
* :mod:`.export` - single-layer and hex exports;
* :mod:`.caveats` - the generated caveats of the results page;
* :mod:`.schemas` - the request and response models (api.md §6); :mod:`.common` - small shared helpers.

Only :mod:`.kinds` is imported by job workers; everything here imports heavy libraries lazily enough for
the server (which never imports torch).
"""
