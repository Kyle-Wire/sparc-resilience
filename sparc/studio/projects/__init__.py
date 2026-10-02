"""Projects: templates, setup, config editing and validation, data checks and open-data input jobs.

Owned by the backend-projects work item (SPEC §9, api.md §5).  Modules:

* :mod:`.schemas` - wire models of api.md §5;
* :mod:`.config_schema` - ``CoreConfigModel`` (``GET /api/config/schema``);
* :mod:`.config_service` - YAML round trip, versions, ``If-Match`` concurrency, impact preview;
* :mod:`.validate` - ``validate_deep`` (SPEC §9.4);
* :mod:`.service` - project CRUD, the ``project.json`` mirror and the reindex hook;
* :mod:`.templates` - blank, synthetic demo, Providence example and config import;
* :mod:`.readiness` - the readiness spine (SPEC §9.3);
* :mod:`.files` - uploads, inspect, column suggestions;
* :mod:`.datacheck` - S0 inline and the preview binaries;
* :mod:`.inputs` - input state, chart views, link-into-config, the ISD station lookup;
* :mod:`.kinds` - the ``input.*`` job kinds (imported by the job registry).
"""
