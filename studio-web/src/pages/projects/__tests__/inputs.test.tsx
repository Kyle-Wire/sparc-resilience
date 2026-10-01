// Inputs step (SPEC §9.3, J2 step 6) with fetch mocked against the api.md fixtures: host
// pre-check, the station picker that offers "Fetch station list" when the ISD index is not
// cached (never downloading inline), launching a tracked forcing job, and "Link into config"
// showing the YAML diff before applying it.
import { afterEach, describe, expect, it } from "vitest";
import { useJobs } from "../../../stores/jobs";
import { byText, click, flush, mockFetch, typeInto, waitFor, type MockHandler } from "../../../test/render";
import { InputsStep } from "../setup/InputsStep";
import { demoRaw, job, PID } from "../__fixtures__/api";
import { projectRoutes, renderStep, resetAll, seedDraft } from "./helpers";

afterEach(() => resetAll());

const base = `/api/projects/${PID}`;
const META = {
  modes: [],
  job_kinds: [
    { kind: "input.forcing", network_hosts: ["cds.climate.copernicus.eu", "www.ncei.noaa.gov"] },
    { kind: "input.cmip6", network_hosts: ["storage.googleapis.com"] },
  ],
};

function routes(extra: Record<string, MockHandler> = {}) {
  return projectRoutes({
    "GET /api/meta": { body: META },
    "POST /api/system/netcheck": (_u, init) => {
      const hosts = (JSON.parse(String(init.body)) as { hosts: string[] }).hosts;
      return { body: { results: hosts.map((h) => ({ host: h, ok: h !== "www.ncei.noaa.gov", ms: h === "www.ncei.noaa.gov" ? null : 120, error: h === "www.ncei.noaa.gov" ? "timeout" : null })) } };
    },
    ...extra,
  });
}

function forcingCard(): HTMLElement {
  return document.querySelector<HTMLElement>('[data-input="forcing"]')!;
}

describe("inputs step", () => {
  it("pre-checks the hosts of each job and shows which answer", async () => {
    const m = mockFetch(routes());
    seedDraft();
    renderStep(<InputsStep />, `/p/${PID}/setup/inputs`);
    await waitFor(() => forcingCard()?.textContent?.includes("120 ms"), 5000, "netcheck");
    const card = forcingCard();
    expect(card.textContent).toContain("cds.climate.copernicus.eu");
    expect(card.textContent).toContain("www.ncei.noaa.gov unreachable");
    expect(card.textContent).toContain("Some hosts did not answer");
    expect(document.querySelectorAll(".input-card")).toHaveLength(4);
    m.restore();
  });

  it("offers Fetch station list when the ISD index is not cached, and launches the forcing job", async () => {
    const posted: { url: string; body: unknown }[] = [];
    const record = (status: number, body: unknown): MockHandler => (u, init) => {
      posted.push({ url: u.pathname, body: JSON.parse(String(init.body ?? "{}")) });
      return { status, body };
    };
    const m = mockFetch(
      routes({
        [`GET ${base}/forcing/stations`]: {
          status: 404,
          body: {
            error: {
              code: "not_found",
              message: "isd-history.csv is not cached",
              detail: { missing: "isd-history.csv" },
              action: { kind: "fetch_input", label: "Fetch station list", method: "POST", path: `${base}/inputs/stations` },
            },
          },
        },
        [`POST ${base}/inputs/stations`]: record(202, job({ id: "j_st", kind: "input.stations", label: "Station list" })),
        [`POST ${base}/inputs/forcing`]: record(202, job({ id: "j_fo", kind: "input.forcing", label: "Campaign forcing" })),
      }),
    );
    seedDraft({ raw: { ...demoRaw(), climate: { enabled: true, source: "table", table: "x.csv", site: [41.826, -71.403] } } });
    renderStep(<InputsStep />, `/p/${PID}/setup/inputs`);
    await waitFor(() => forcingCard(), 5000, "card");
    const card = forcingCard();
    // the location defaults to climate.site
    expect(card.querySelector<HTMLInputElement>('input[aria-label="Latitude"]')!.value).toBe("41.826");
    click(byText(card, "button", "Find nearby stations"));
    const fetchBtn = await waitFor(() => byText(forcingCard(), "button", "Fetch station list"), 5000, "fetch action");
    expect(forcingCard().textContent).toContain("never downloads inside a page request");
    click(fetchBtn);
    await waitFor(() => posted.some((p) => p.url.endsWith("/inputs/stations")), 5000, "stations job");
    // the start button waits for a date
    const start = byText(forcingCard(), "button", "Fetch forcing") as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    typeInto(forcingCard().querySelector('input[aria-label="Campaign date"]'), "2020-07-29");
    typeInto(forcingCard().querySelector('input[aria-label="Time zone"]'), "America/New_York");
    expect(start.disabled).toBe(false);
    click(start);
    await waitFor(() => useJobs.getState().jobs.j_fo, 5000, "forcing job");
    expect(posted.find((p) => p.url.endsWith("/inputs/forcing"))!.body).toEqual({
      date: "2020-07-29",
      hours: [15, 16],
      tz: "America/New_York",
      lat: 41.826,
      lon: -71.403,
      wind_source: "auto",
      link: false,
    });
    expect(useJobs.getState().jobs.j_fo?.kind).toBe("input.forcing");
    // the running job shows as a JobStrip in the card
    await waitFor(() => forcingCard().querySelector(".jobstrip"), 5000, "job strip");
    m.restore();
  });

  it("Link into config previews the YAML diff, then applies it", async () => {
    const links: unknown[] = [];
    const m = mockFetch(
      routes({
        [`GET ${base}/inputs`]: {
          body: {
            forcing: { path: "inputs/forcing/forcing_2020-07-29.json", date: "2020-07-29", physics: { sw_down: 636 }, checks: ["station 42 km away"], linked: false },
            climate: null,
            layers: null,
            features: null,
            ghcn: null,
          },
        },
        [`GET ${base}/inputs/forcing/view`]: {
          body: { era5: {}, station: null, compare: [{ name: "air temperature (°F)", era5: 87.8, station: 87.5 }], checks: ["station 42 km away"] },
        },
        [`POST ${base}/link`]: (_u, init) => {
          const b = JSON.parse(String(init.body)) as { apply: boolean };
          links.push(b);
          return { body: b.apply ? { yaml_diff: "", applied: true, version: 4 } : { yaml_diff: "--- a\n+++ b\n@@ -10,2 +10,3 @@\n   physics:\n-    forcing: null\n+    forcing: inputs/forcing/forcing_2020-07-29.json\n", applied: false } };
        },
      }),
    );
    seedDraft();
    renderStep(<InputsStep />, `/p/${PID}/setup/inputs`);
    const btn = await waitFor(() => byText(forcingCard(), "button", "Link into config"), 5000, "link button");
    expect(forcingCard().textContent).toContain("station 42 km away");
    expect(forcingCard().querySelector('[aria-label="ERA5 vs station"]')!.textContent).toContain("87.8");
    click(btn);
    await waitFor(() => document.querySelector('.dialog [aria-label="Config changes"] .add'), 5000, "diff");
    const diff = document.querySelector('.dialog [aria-label="Config changes"]')!;
    expect(diff.querySelector(".del")!.textContent).toContain("forcing: null");
    expect(diff.querySelector(".add")!.textContent).toContain("forcing: inputs/forcing/forcing_2020-07-29.json");
    expect(links).toEqual([{ kind: "forcing", path: "inputs/forcing/forcing_2020-07-29.json", apply: false }]);
    click(byText(document.body, ".dialog button", "Apply to config"));
    await waitFor(() => links.length === 2 && !document.querySelector(".dialog"), 5000, "applied");
    expect(links[1]).toEqual({ kind: "forcing", path: "inputs/forcing/forcing_2020-07-29.json", apply: true });
    await flush(2);
    m.restore();
  });
});
