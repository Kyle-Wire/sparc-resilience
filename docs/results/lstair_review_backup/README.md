# LST pilot: city review backup (2026-10-20)

Working files from the 31-city data review, kept so the fix round can resume after the session's container is reclaimed:

- `qa_all.json`: per-city QA facts from `qa_city.py`.
- `qa_result.json`, `qa_details.txt`: the review workflow's findings (`city_review_workflow.js`), with three reviewers, a skeptic per reviewer and a completeness critic.
- `osf_survey.json`, `clock_survey.json`: the OSF file listing per catalog city and the clock check per downloaded city.
- `qa_fixes_workflow.js`: the four-stream fix workflow (data, Landsat, evaluation, NLCD land cover), interrupted before its reviews.

These are not final results. `docs/results/lst_air_pilot.md` (pre-registered) is the record.
