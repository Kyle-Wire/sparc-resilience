"""Exports, the project report and the findings notebook (SPEC §6.7, §6.9, §6.10, api.md §10–11).

* :mod:`.store` - the ``exports`` rows, ``POST /api/exports`` (row first, then ``export.<kind>`` by name),
  their ``export.json`` record and the reindex hook;
* :mod:`.bundle` / :mod:`.gis` - the whole-run ZIP and the GIS pack;
* :mod:`.narrative` / :mod:`.report` - sentences written from the numbers and the self-contained report;
* :mod:`.findings` - findings rows, their mirror files and the Markdown / HTML export;
* :mod:`.kinds` - the ``export.bundle|gis|page|report|findings`` job kinds.

Every export writes under ``projects/<slug>/exports/<export_id>/``.
"""
