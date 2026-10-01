import { afterEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { api } from "../api/client";
import { CodeArea } from "../components/ui/CodeArea";
import { Chips } from "../components/ui/Chips";
import { ConfirmDialog, Dialog } from "../components/ui/Dialog";
import { Drawer } from "../components/ui/Drawer";
import { FileDrop } from "../components/ui/FileDrop";
import { Slider } from "../components/ui/Slider";
import { Tabs } from "../components/ui/Tabs";
import { Toasts } from "../components/ui/Toast";
import { VirtualList, visibleRange } from "../components/ui/VirtualList";
import { useUi } from "../stores/ui";
import { byText, click, flush, key, mockFetch, render, typeInto } from "./render";

afterEach(() => {
  vi.unstubAllEnvs();
  useUi.setState({ toasts: [] });
});

describe("Dialog", () => {
  it("is modal, labelled, focuses its content and closes on Escape", async () => {
    let open = true;
    const onClose = () => (open = false);
    render(
      <Dialog open={open} onClose={onClose} title="Delete checkpoint?" footer={<button type="button">Delete</button>}>
        <input aria-label="Confirm name" data-autofocus />
      </Dialog>,
    );
    await flush(2);
    const dlg = document.querySelector('[role="dialog"]')!;
    expect(dlg.getAttribute("aria-modal")).toBe("true");
    expect(document.getElementById(dlg.getAttribute("aria-labelledby")!)!.textContent).toBe("Delete checkpoint?");
    expect(document.activeElement?.getAttribute("aria-label")).toBe("Confirm name");
    key(dlg, "Escape");
    expect(open).toBe(false);
  });
  it("ConfirmDialog guards the action", () => {
    let confirmed = 0;
    render(<ConfirmDialog open onClose={() => {}} onConfirm={() => confirmed++} title="Force stop?" confirmLabel="Force stop" danger />);
    click(byText(document.body, "button", "Force stop"));
    expect(confirmed).toBe(1);
  });
  it("Drawer renders an inline complementary panel", () => {
    let closed = false;
    const { container } = render(
      <Drawer open mode="inline" title="Analysis tools" onClose={() => (closed = true)}>
        body
      </Drawer>,
    );
    expect(container.querySelector('aside[role="complementary"]')).not.toBeNull();
    click(container.querySelector('button[aria-label="Close panel"]'));
    expect(closed).toBe(true);
  });
});

describe("Tabs, Chips, Slider", () => {
  it("Tabs use roving focus and arrow keys", () => {
    function T() {
      const [v, setV] = useState<"logs" | "warnings" | "outputs">("logs");
      return <Tabs label="Bottom" value={v} onChange={setV} items={[{ id: "logs", label: "Logs" }, { id: "warnings", label: "Warnings" }, { id: "outputs", label: "Outputs" }]}>{`panel ${v}`}</Tabs>;
    }
    const { container } = render(<T />);
    const list = container.querySelector('[role="tablist"]')!;
    key(list, "ArrowRight");
    expect(container.querySelector('[aria-selected="true"]')!.textContent).toBe("Warnings");
    key(list, "End");
    expect(container.querySelector('[role="tabpanel"]')!.textContent).toBe("panel outputs");
    key(list, "ArrowRight");
    expect(container.querySelector('[aria-selected="true"]')!.textContent).toBe("Logs");
  });
  it("Chips toggle with aria-pressed", () => {
    const toggled: [string, boolean][] = [];
    const { container } = render(<Chips label="SSP" items={[{ value: "ssp245", label: "SSP2-4.5" }, { value: "ssp585", label: "SSP5-8.5" }]} selected={["ssp245"]} onToggle={(v, on) => toggled.push([v, on])} />);
    expect(container.querySelector('[aria-pressed="true"]')!.textContent).toBe("SSP2-4.5");
    click(byText(container, "button", "SSP5-8.5"));
    expect(toggled).toEqual([["ssp585", true]]);
  });
  it("Slider maps a log scale", () => {
    let v = 1000;
    const { container } = render(<Slider label="Budget" value={v} min={100} max={100000} log step={100} onChange={(x) => (v = x)} format={(x) => `${x} pp·cells`} />);
    const input = container.querySelector('input[type="range"]')!;
    expect(input.getAttribute("aria-valuetext")).toBe("1000 pp·cells");
    expect(Number((input as HTMLInputElement).value)).toBeCloseTo(333.33, 1);
    typeInto(input, "1000");
    expect(v).toBe(100000);
  });
});

describe("CodeArea, VirtualList, Toasts", () => {
  it("CodeArea numbers lines and marks issues in the gutter", () => {
    const { container } = render(<CodeArea label="config.yml" value={"data:\n  path: x.csv\nmodels: [mgwr]"} issues={[{ line: 2, level: "error", message: "file not found" }]} />);
    const rows = container.querySelectorAll(".gutter div");
    expect(rows).toHaveLength(3);
    expect(rows[1].getAttribute("data-level")).toBe("error");
    expect(rows[1].getAttribute("title")).toBe("file not found");
    expect(container.querySelector("textarea")!.getAttribute("aria-label")).toBe("config.yml");
    expect(container.textContent).toContain("Line 2: error file not found");
  });
  it("VirtualList renders only the visible window", () => {
    expect(visibleRange(0, 200, 20, 10000, 2)).toEqual([0, 12]);
    expect(visibleRange(1000, 200, 20, 10000, 2)).toEqual([48, 62]);
    const { container } = render(<VirtualList count={10000} rowHeight={20} height={200} renderRow={(i) => `row ${i}`} label="Log" />);
    expect(container.querySelectorAll('[role="listitem"]').length).toBeLessThan(30);
    expect((container.querySelector(".vlist-inner") as HTMLElement).style.height).toBe("200000px");
  });
  it("Toasts show title, body and a link, and dismiss", () => {
    useUi.getState().pushToast({ kind: "error", title: "Run failed", body: "MemoryError", href: "/jobs/j_1", linkLabel: "Open" });
    const { container } = render(<Toasts />);
    expect(container.querySelector('[role="alert"]')!.textContent).toContain("Run failed");
    expect(byText(container, "a", "Open")!.getAttribute("href")).toBe("/jobs/j_1");
    click(container.querySelector('button[aria-label="Dismiss notification"]'));
    expect(useUi.getState().toasts).toHaveLength(0);
  });
});

describe("FileDrop", () => {
  it("streams a raw PUT and shows upload progress", async () => {
    const sent: { url: string; method: string; body: unknown; headers: Record<string, string> }[] = [];
    class FakeXHR {
      upload: { onprogress: ((e: { loaded: number; total: number; lengthComputable: boolean }) => void) | null } = { onprogress: null };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onabort: (() => void) | null = null;
      status = 0;
      statusText = "";
      responseText = "";
      withCredentials = false;
      private h: Record<string, string> = {};
      private m = "";
      private u = "";
      open(m: string, u: string) {
        this.m = m;
        this.u = u;
      }
      setRequestHeader(k: string, v: string) {
        this.h[k] = v;
      }
      getResponseHeader() {
        return "application/json";
      }
      abort() {}
      send(body: unknown) {
        sent.push({ url: this.u, method: this.m, body, headers: this.h });
        this.upload.onprogress?.({ loaded: 4, total: 8, lengthComputable: true });
        setTimeout(() => {
          this.status = 201;
          this.responseText = JSON.stringify({ name: "brown4.csv", bytes: 10 });
          this.onload?.();
        }, 0);
      }
    }
    vi.stubGlobal("XMLHttpRequest", FakeXHR);
    const done: unknown[] = [];
    const { container } = render(<FileDrop label="Upload CSV" accept=".csv" upload={{ url: (f) => `/api/projects/p_1/files/data/${f.name}`, contentType: () => "text/csv", onDone: (r) => done.push(r) }} />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["a,b\n1,2\n"], "brown4.csv", { type: "text/csv" });
    Object.defineProperty(input, "files", { value: [file] });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flush(4);
    vi.unstubAllGlobals();
    expect(sent[0]).toMatchObject({ method: "PUT", url: "/api/projects/p_1/files/data/brown4.csv", headers: { "Content-Type": "text/csv" } });
    expect(sent[0].body).toBe(file);
    expect(done).toEqual([{ name: "brown4.csv", bytes: 10 }]);
    expect(container.textContent).toContain("Uploaded 8 B");
    expect(container.querySelector('[role="progressbar"]')!.getAttribute("aria-valuenow")).toBe("100");
  });

  it("clears the picker after a choice, so the same file can be chosen again", () => {
    const got: string[] = [];
    const { container } = render(<FileDrop label="Upload CSV" onFiles={(fs) => got.push(...fs.map((f) => f.name))} />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    let value = "C:\\fakepath\\brown4.csv";
    Object.defineProperty(input, "value", { get: () => value, set: (v: string) => void (value = v), configurable: true });
    Object.defineProperty(input, "files", { value: [new File(["x"], "brown4.csv")] });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    expect(got).toEqual(["brown4.csv"]);
    expect(value).toBe("");
  });
});

describe("dev token bootstrap", () => {
  it("exchanges VITE_STUDIO_TOKEN at /auth on a 401 and retries once", async () => {
    vi.stubEnv("VITE_STUDIO_TOKEN", "dev");
    let authed = false;
    const m = mockFetch({
      "GET /auth": () => {
        authed = true;
        return { status: 200, body: { ok: true } };
      },
      "GET /api/health": () => (authed ? { body: { ok: true } } : { status: 401, body: { error: { code: "unauthorized", message: "Missing or invalid cookie/token" } } }),
    });
    const r = await api.get<{ ok: boolean }>("/api/health");
    m.restore();
    expect(r.ok).toBe(true);
    expect(m.calls.map((c) => c.url)).toEqual(["/api/health", "/auth?t=dev&next=%2Fapi%2Fhealth", "/api/health"]);
  });
});
