"""The emulator preview (SPEC §7.5, api.md §7.3): core parity, latest-wins, one thread, missing emulator."""

from __future__ import annotations

import base64
import json
import threading

import numpy as np
import pytest

EDITS = [{"lever": "canopy", "mode": "add", "amount": 10,
          "where": {"kind": "top", "column": "pred:target", "frac": 0.2, "direction": "highest"}},
         {"lever": "albedo", "mode": "set", "amount": 0.3,
          "where": {"kind": "filter", "column": "predictor:impervious", "op": ">=", "value": 50}}]


def _unpack(r) -> dict:
    offs = json.loads(r.headers["X-SPARC-Offsets"])
    out = {}
    for o in offs:
        dt = {"float32": "<f4", "uint8": "u1"}[o["dtype"]]
        size = np.dtype(dt).itemsize
        out[o["name"]] = np.frombuffer(r.content[o["offset"]:o["offset"] + size * o["length"]], dtype=dt)
    return out


def _expected(ctx, run_ctx, edits, brush=None):
    from sparc.core.emulator import emulate
    from sparc.studio.engine.compile import compile_scenario
    from sparc.studio.engine.preview import load_emulator

    em = load_emulator(run_ctx)
    comp = compile_scenario(run_ctx, {"name": "p", "edits": edits}, db=ctx.db, brush=brush)
    want = np.zeros(comp.n)
    for var, dx in comp.dx().items():
        if np.any(dx):
            want = want + emulate(em.levers[var], run_ctx.data.grid, dx)
    return comp, want


def test_preview_equals_core_emulate(client, ctx, run_ctx, synth_run, fake_emulator):
    rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    idx = np.array([1, 2, 3], dtype="<i4")
    val = np.array([5.0, 5.0, 5.0], dtype="<f4")
    brush = {"canopy": {"idx": base64.b64encode(idx.tobytes()).decode(), "val": base64.b64encode(val.tobytes()).decode()}}
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": EDITS, "brush": brush, "request_seq": 1})
    assert r.status_code == 200, r.text
    got = _unpack(r)
    comp, want = _expected(ctx, run_ctx, EDITS, brush)
    assert got["delta"].size == run_ctx.grid.n
    np.testing.assert_allclose(got["delta"], want, atol=1e-6, rtol=0)
    edited = np.unpackbits(got["edited"], bitorder="little")[:comp.n].astype(bool)
    np.testing.assert_array_equal(edited, comp.edited)
    s = json.loads(r.headers["X-SPARC-Summary"])
    assert s["request_seq"] == 1 and s["n_edited"] == int(comp.edited.sum())
    assert s["mean"] == pytest.approx(float(np.mean(want)), abs=1e-9)
    assert s["edited_mean"] == pytest.approx(float(np.mean(want[comp.edited])), abs=1e-9)
    assert s["trust"] == "rough"                    # albedo's validation in the fake emulator is rough
    assert set(s) == {"mean", "edited_mean", "n_edited", "outside_share", "trust", "hatched", "reasons",
                      "request_seq"}
    # a city-wide edit of a lever whose uniform error is 246 % is hatched with the SPEC's banner
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": [{"lever": "albedo", "mode": "add", "amount": 0.05}],
                                                      "request_seq": 2})
    s = json.loads(r.headers["X-SPARC-Summary"])
    assert s["hatched"] and any("uniform albedo rel. error 246%" in x for x in s["reasons"])


def test_newer_request_supersedes_older(client, synth_run, run_ctx, fake_emulator):
    """An older request_seq is superseded while a newer one for the run is in flight; once the run's previews
    are idle, a restarted sequence (a reloaded page) is served again."""
    from sparc.studio.engine import preview as P

    rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    gate, started = threading.Event(), threading.Event()

    def busy():
        started.set()
        gate.wait(10)

    t = threading.Thread(target=P.FLIGHTS.run, args=(rid, 5, busy))
    t.start()
    try:
        assert started.wait(5)
        r = client.post(f"/api/runs/{rid}/preview", json={"edits": EDITS, "request_seq": 3})
        assert r.status_code == 409 and r.json()["error"]["code"] == "superseded"
        assert r.json()["error"]["detail"] == {"request_seq": 3, "latest_seq": 5}
    finally:
        gate.set()
        t.join(5)
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": EDITS, "request_seq": 1})
    assert r.status_code == 200 and json.loads(r.headers["X-SPARC-Summary"])["request_seq"] == 1


def test_queued_request_is_superseded_by_a_newer_one():
    from sparc.studio.engine.preview import PreviewFlights
    from sparc.studio.errors import ApiError

    flights = PreviewFlights()
    gate, started = threading.Event(), threading.Event()
    results: dict = {}

    def slow():
        started.set()
        gate.wait(5)
        return "first"

    def call(seq, fn):
        try:
            results[seq] = flights.run("r", seq, fn)
        except ApiError as exc:
            results[seq] = exc.code

    t1 = threading.Thread(target=call, args=(1, slow))
    t1.start()
    started.wait(5)
    t2 = threading.Thread(target=call, args=(2, lambda: "second"))
    t2.start()
    t3 = threading.Thread(target=call, args=(3, lambda: "third"))
    t3.start()
    import time

    time.sleep(0.2)
    gate.set()
    for t in (t1, t2, t3):
        t.join(5)
    assert results[1] == "first"                    # in flight: completes
    assert results[3] == "third"
    assert results[2] == "superseded"               # queued behind 1 while 3 arrived


def test_preview_runs_with_one_thread(ctx, run_ctx, synth_run, fake_emulator, monkeypatch):
    from threadpoolctl import threadpool_info, threadpool_limits

    from sparc.core import emulator as emulator_mod
    from sparc.studio.engine import preview as P
    from sparc.studio.scenarios.schemas import PreviewRequest

    _rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    seen = []
    real = emulator_mod.emulate

    def spy(em, grid, dx):
        seen.append([lib["num_threads"] for lib in threadpool_info()])
        return real(em, grid, dx)

    monkeypatch.setattr(emulator_mod, "emulate", spy)
    with threadpool_limits(limits=3):
        before = [lib["num_threads"] for lib in threadpool_info()]
        P.preview(run_ctx, PreviewRequest(edits=EDITS, request_seq=1), db=ctx.db)
        after = [lib["num_threads"] for lib in threadpool_info()]
    assert seen and all(n == 1 for row in seen for n in row)
    assert before == after


def test_missing_emulator_is_404_with_build_action(client, synth_run):
    rid, _ = synth_run
    r = client.post(f"/api/runs/{rid}/preview", json={"edits": EDITS, "request_seq": 1})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "no_emulator"
    assert err["action"]["kind"] == "build_emulator" and err["action"]["path"] == f"/api/runs/{rid}/actions/emulator"
    info = client.get(f"/api/runs/{rid}/emulator").json()
    assert info["present"] is False and info["action"]["kind"] == "build_emulator"


def test_levers_and_emulator_info(client, synth_run, run_ctx, fake_emulator):
    rid, rd = synth_run
    fake_emulator(rd, run_ctx)
    lv = {x["var"]: x for x in client.get(f"/api/runs/{rid}/levers").json()}
    assert set(lv) == {"canopy", "impervious", "albedo"}
    assert lv["canopy"]["emulator"]["trust"] == "good" and lv["albedo"]["emulator"]["trust"] == "rough"
    assert lv["impervious"]["direction"] == "decrease" and lv["canopy"]["role"] == "canopy"
    assert lv["canopy"]["mediator_children"] == ["ndvi"] and lv["canopy"]["headroom_available"] is True
    assert lv["canopy"]["sd"] == pytest.approx(18.405, rel=1e-3)
    info = client.get(f"/api/runs/{rid}/emulator").json()
    assert info["present"] and info["levers"]["canopy"]["trust"] == "good" and info["action"] is None


def test_older_request_arriving_during_a_newer_one_is_superseded_at_once():
    """Out-of-order arrival: an older request that arrives while a newer one is in flight never waits."""
    from sparc.studio.engine.preview import PreviewFlights
    from sparc.studio.errors import ApiError

    flights = PreviewFlights()
    gate, started = threading.Event(), threading.Event()
    calls = []

    def slow():
        started.set()
        gate.wait(5)
        return "newer"

    t = threading.Thread(target=flights.run, args=("r", 7, slow))
    t.start()
    started.wait(5)
    with pytest.raises(ApiError) as exc:
        flights.run("r", 6, lambda: calls.append(6))
    assert exc.value.code == "superseded" and calls == []
    gate.set()
    t.join(5)
    assert flights.run("r", 2, lambda: "restarted") == "restarted"     # idle: a new page's sequence
    assert flights.run("other-run", 1, lambda: "independent") == "independent"
