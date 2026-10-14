// Reports (`/p/:pid/exports`, phase 2): the project's one place for output, as two tabs under
// one page title:
// - "Reports & exports" (default): the Exports page without its title (`ExportsPanel`): report
//   builder, run bundle, GIS pack, standalone results page, the run's GIS & data files, history.
// - "Notes" (`?tab=notes`; `/p/:pid/findings` redirects here): the project's Findings notebook
//   without its title (`FindingsPanel`). `/findings` keeps the workspace-wide notebook page.
// Switching tabs drops `run`, which the two panels read differently (the run to export vs. the
// notes filter).
import { Tabs } from "../../components/ui/Tabs";
import { codecs, setQuery, useUrlState } from "../../router";
import { ExportsPanel } from "./Exports";
import { FindingsPanel } from "./Findings";
import "./studies.css";

export const REPORT_TABS = ["exports", "notes"] as const;
export type ReportTab = (typeof REPORT_TABS)[number];

export const REPORT_TAB_LABELS: Record<ReportTab, string> = { exports: "Reports & exports", notes: "Notes" };

export default function Reports() {
  const [tab] = useUrlState("tab", codecs.enum(REPORT_TABS, "exports"));
  return (
    <section className="sx-page" aria-labelledby="reports-title">
      <header className="page-head">
        <h1 id="reports-title">Reports</h1>
      </header>
      <Tabs
        label="Reports"
        value={tab}
        onChange={(t) => setQuery({ tab: t === "exports" ? null : t, run: null }, { push: true })}
        items={REPORT_TABS.map((id) => ({ id, label: REPORT_TAB_LABELS[id] }))}
      >
        {tab === "notes" ? <FindingsPanel /> : <ExportsPanel />}
      </Tabs>
    </section>
  );
}
