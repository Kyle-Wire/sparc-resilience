// ViewModel fixtures for the run-hub tests. views.json holds one ViewModel per view (api.md
// §6.1 envelope; section shapes of src/api/runs.ts) whose numbers were derived from the
// synthetic run fixture tests/studio/fixtures/synth_run (manifest.json, influence.json,
// causal.json, climate.json, optimize.json, baselines.json, response_curves.json,
// scenarios.json, predictions.parquet). Sections the synthetic run cannot fill (planner pack,
// uncertainty) carry small hand-made values of the same shape.
import type { ViewModelOf, ViewName } from "../../../api/runs";
import raw from "./views.json";

const VIEWS = raw as unknown as { [V in ViewName]: ViewModelOf<V> };

/** A fresh copy of one view's fixture (tests may mutate it). */
export function viewFixture<V extends ViewName>(view: V): ViewModelOf<V> {
  return structuredClone(VIEWS[view]);
}

/** The view as an older-code run would return it: every section key present and null. */
export function olderCodeFixture<V extends ViewName>(view: V): ViewModelOf<V> {
  const vm = viewFixture(view);
  const sections = Object.fromEntries(Object.keys(vm.sections).map((k) => [k, null]));
  return { ...vm, sections: sections as ViewModelOf<V>["sections"] };
}

/** Section keys of a view (api.md §6.1 table). */
export function sectionKeys(view: ViewName): string[] {
  return Object.keys(VIEWS[view].sections);
}
