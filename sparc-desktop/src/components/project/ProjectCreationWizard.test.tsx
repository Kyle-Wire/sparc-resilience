/**
 * ProjectCreationWizard — template flash tests.
 *
 * T-wizard-flash: Wizard must never show an empty template grid on first
 * render, even when the parent passes templates=[] and the server hasn't
 * responded yet.  The wizard should seed the picker from its built-in
 * TEMPLATE_BLURBS catalog immediately, then replace with the server list
 * once the fetch resolves.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import ProjectCreationWizard from "./ProjectCreationWizard";

// Tauri APIs used indirectly by the api layer — stub at import time
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));
vi.mock("@tauri-apps/plugin-shell", () => ({ Command: { create: vi.fn() } }));

const noop = () => {};

describe("ProjectCreationWizard template seeding", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("renders template cards immediately when no templates are pre-loaded (no flash)", () => {
    // No fetch stub — server is not running. Component must still show templates
    // from its built-in fallback catalog on the very first render.
    render(
      <ProjectCreationWizard templates={[]} onCreated={noop} onCancel={noop} />,
    );

    // Step 0 is the template picker. At least one template name must be
    // visible without any async work.
    const cards = screen.getAllByRole("button", { name: /uhi|blank|coastal|drought|wildfire/i });
    expect(cards.length).toBeGreaterThan(0);
  });

  it("does not show a loading-only state (no templates visible) at any point when server is slow", async () => {
    // Freeze fetch so the server never responds during this test
    vi.stubGlobal("fetch", () => new Promise(() => {})); // never resolves

    render(
      <ProjectCreationWizard templates={[]} onCreated={noop} onCancel={noop} />,
    );

    // Even while fetch is pending, templates should be visible
    const cards = screen.getAllByRole("button", { name: /uhi|blank|coastal|drought|wildfire/i });
    expect(cards.length).toBeGreaterThan(0);
  });

  it("uses pre-loaded templates from parent when provided", () => {
    const templates = [
      { name: "my-template", has_project_yml: true },
    ];
    render(
      <ProjectCreationWizard templates={templates} onCreated={noop} onCancel={noop} />,
    );
    expect(screen.getByRole("button", { name: /my-template/i })).toBeInTheDocument();
  });

  it("keeps fallback templates when server returns an empty list", async () => {
    // Server is reachable but has no templates (e.g. wheel install with missing
    // templates dir). The wizard must NOT clear the fallback catalog.
    //
    // Use a controllable promise so the test can verify the state BEFORE
    // and AFTER the fetch resolves, avoiding React 18 batching races.
    let resolveFetch!: (v: unknown) => void;
    const fetchPending = new Promise<unknown>(res => { resolveFetch = res; });
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(fetchPending));

    render(
      <ProjectCreationWizard templates={[]} onCreated={noop} onCancel={noop} />,
    );

    // Before fetch resolves: initial fallback catalog must be present.
    const before = screen.getAllByRole("button", { name: /uhi|blank|coastal|drought|wildfire/i });
    expect(before.length).toBeGreaterThan(0);

    // Resolve the fetch with an empty templates list and flush all async work.
    await act(async () => {
      resolveFetch({
        ok: true,
        json: () => Promise.resolve({ templates: [] }),
        text: () => Promise.resolve(""),
      });
      // Drain the microtask queue: fetch → res.json() → listTemplates .then() → React state
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });

    // After fetch resolved with [], fallback catalog must still be shown.
    const after = screen.getAllByRole("button", { name: /uhi|blank|coastal|drought|wildfire/i });
    expect(after.length).toBeGreaterThan(0);
  });

  it("updates template list when server responds with more templates", async () => {
    const serverTemplates = [{ name: "server-only-template", has_project_yml: true }];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ templates: serverTemplates }),
        text: () => Promise.resolve(""),
      }),
    );

    render(
      <ProjectCreationWizard templates={[]} onCreated={noop} onCancel={noop} />,
    );

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /server-only-template/i })).toBeInTheDocument(),
    );
  });
});
