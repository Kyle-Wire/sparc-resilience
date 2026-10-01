import { afterEach, describe, expect, it } from "vitest";
import { clearResources, invalidate, invalidateAll, mutate, peekResource, useResource } from "../api/resource";
import { flush, render } from "./render";

afterEach(() => clearResources());

function Probe({ k, fetcher, tags, immutable, label }: { k: string | null; fetcher: () => Promise<string>; tags?: string[]; immutable?: boolean; label: string }) {
  const r = useResource(k, fetcher, { tags, immutable, keepPrevious: true });
  return (
    <span data-testid={label}>
      {r.status}:{r.data ?? "-"}
    </span>
  );
}

describe("useResource", () => {
  it("shares one request between components with the same key", async () => {
    let n = 0;
    const f = async () => `v${++n}`;
    const { container } = render(
      <>
        <Probe k="run:r1:outputs" fetcher={f} label="a" />
        <Probe k="run:r1:outputs" fetcher={f} label="b" />
      </>,
    );
    await flush();
    expect(n).toBe(1);
    expect(container.textContent).toBe("ready:v1ready:v1");
  });

  it("invalidates by tag with a ':' boundary and refetches mounted entries", async () => {
    let a = 0;
    let b = 0;
    let c = 0;
    const { container } = render(
      <>
        <Probe k="run:r1:outputs" tags={["run:r1"]} fetcher={async () => `a${++a}`} label="a" />
        <Probe k="run:r10:outputs" tags={["run:r10"]} fetcher={async () => `b${++b}`} label="b" />
        <Probe k="run:r1:grid" tags={["run:r1"]} immutable fetcher={async () => `c${++c}`} label="c" />
      </>,
    );
    await flush();
    invalidate("run:r1");
    await flush();
    expect([a, b, c]).toEqual([2, 1, 1]); // r10 untouched (boundary), immutable grid not refetched
    expect(container.textContent).toBe("ready:a2ready:b1ready:c1");
    invalidate("run:r1:outputs"); // exact sub-tag via the entry's own key
    await flush();
    expect(a).toBe(3);
    invalidateAll();
    await flush();
    expect([a, b, c]).toEqual([4, 2, 1]);
  });

  it("drops unmounted entries on invalidation so the next mount refetches", async () => {
    let n = 0;
    const f = async () => `v${++n}`;
    const r = render(<Probe k="project:p_1" fetcher={f} label="a" />);
    await flush();
    r.unmount();
    invalidate("project:p_1");
    expect(peekResource("project:p_1")).toBeUndefined();
    render(<Probe k="project:p_1" fetcher={f} label="a" />);
    await flush();
    expect(n).toBe(2);
  });

  it("surfaces errors and supports local mutation", async () => {
    const { container } = render(<Probe k="bad" fetcher={async () => Promise.reject(new Error("boom"))} label="a" />);
    await flush();
    expect(container.textContent).toBe("error:-");
    mutate("bad", "fixed");
    await flush();
    expect(container.textContent).toBe("ready:fixed");
  });

  it("skips fetching for a null key and keeps previous data while a new key loads", async () => {
    let resolve: (v: string) => void = () => {};
    const slow = () => new Promise<string>((r) => (resolve = r));
    const r = render(<Probe k={null} fetcher={async () => "x"} label="a" />);
    expect(r.container.textContent).toBe("idle:-");
    r.rerender(<Probe k="k1" fetcher={async () => "one"} label="a" />);
    await flush();
    r.rerender(<Probe k="k2" fetcher={slow} label="a" />);
    await flush();
    expect(r.container.textContent).toBe("loading:one");
    resolve("two");
    await flush();
    expect(r.container.textContent).toBe("ready:two");
  });
});
