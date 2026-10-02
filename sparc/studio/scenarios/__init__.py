"""Scenario Lab services (SPEC §7, api.md §7): the scenario library, templates, budget plans, climate ×
adaptation, sweeps, impacts, comparisons and the decision / plan / compare packs.

Modules:

* :mod:`.schemas` - request and response models of api.md §7;
* :mod:`.library` - scenarios (CRUD, revisions and forks, ladders, promote to config, design CSVs), the
  project mirror files and the ``scenarios`` reindex;
* :mod:`.templates` - the seven scenario templates;
* :mod:`.plans` - budget plans (planned mode, field kit, plan → scenario);
* :mod:`.climate` - CMIP6 factors and the climate × adaptation explorer;
* :mod:`.sweeps` - dose sweeps;
* :mod:`.impacts` - exposure, equity, hot days, zones, hexagons and climate offset of a result;
* :mod:`.compare` - comparisons with paired standard errors;
* :mod:`.packs` - the ``export.*_pack`` builders.

Everything here runs in the server (or a job worker); nothing imports torch.
"""
