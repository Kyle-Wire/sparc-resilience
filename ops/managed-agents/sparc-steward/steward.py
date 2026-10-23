"""Run the sparc-steward Managed Agent.

Setup (agent, environment, memory store) is done once with `ant apply`; see
README.md. This script is the runtime side:

    python steward.py burst --total 95 --per-run 48   # back-to-back runs until $95 is spent
    python steward.py run --budget 24                  # one run
    python steward.py schedule                         # create the recurring deployment (left paused)
    python steward.py sweep                            # deny calls left waiting in scheduled runs

Needs: `pip install anthropic`, an Anthropic credential (`ant auth login` or
ANTHROPIC_API_KEY), and GITHUB_TOKEN (a fine-grained token with Contents
read/write on kyle-wire/sparc-resilience only). IDs come from AGENT_ID / ENV_ID
/ STORE_ID, or from claude-lock.json at the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
REPO_URL = "https://github.com/kyle-wire/sparc-resilience"
BRANCH = "sparc-steward"
SCHEDULE = {"type": "cron", "expression": "13 22 1,8,15,22 * *", "timezone": "America/New_York"}
SCHEDULED_BUDGET_CENTS = 2400  # 4 runs a month x $24 = $96

DESCRIPTION = (
    "Advance the SPARC roadmap. Read docs/agent/STATUS.md, docs/agent/ROADMAP.md "
    "and your memory notes, then complete the next unfinished step of the current "
    "phase, test it, write it up, record it, and push it to the sparc-steward branch."
)


def _lock_id(suffix: str) -> str | None:
    lock = REPO_ROOT / "claude-lock.json"
    if not lock.exists():
        return None
    for path, entry in json.loads(lock.read_text())["resources"].items():
        if path.endswith(f"sparc-steward/{suffix}"):
            return entry["id"]
    return None


def _ids() -> tuple[str, str, str]:
    ids = (
        os.getenv("AGENT_ID") or _lock_id("agent.md"),
        os.getenv("ENV_ID") or _lock_id("environment.yaml"),
        os.getenv("STORE_ID") or _lock_id("memory_store.yaml"),
    )
    if not all(ids):
        sys.exit("Missing AGENT_ID / ENV_ID / STORE_ID; run `ant apply` first (README.md).")
    return ids


def _resources(store_id: str) -> list[dict]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN (fine-grained, Contents read/write on this repo only).")
    return [
        {
            "type": "github_repository",
            "url": REPO_URL,
            "authorization_token": token,
            "checkout": {"type": "branch", "name": BRANCH},
        },
        {
            "type": "memory_store",
            "memory_store_id": store_id,
            "access": "read_write",
            "instructions": "Your notes from earlier runs. Read them before starting; add this run's notes before finishing.",
        },
    ]


def _outcome() -> dict:
    return {
        "type": "user.define_outcome",
        "description": DESCRIPTION,
        "rubric": {"type": "text", "content": (HERE / "rubric.md").read_text()},
        "max_iterations": 5,
    }


def _budget(cents: int) -> dict:
    return {"type": "limit", "max_list_cost": {"amount": str(cents), "currency": "USD"}}


def _deny(client: anthropic.Anthropic, session_id: str, event_id: str) -> None:
    client.beta.sessions.events.send(
        session_id=session_id,
        events=[{
            "type": "user.tool_confirmation",
            "tool_use_id": event_id,
            "result": "deny",
            "deny_message": "No one is watching this run to approve this call. Find another way, or record it in STATUS.md as a blocker.",
        }],
    )


def run_once(client: anthropic.Anthropic, budget_cents: int) -> tuple[int, str]:
    """One session to completion. Returns (cents spent, how it ended)."""
    agent_id, env_id, store_id = _ids()
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        resources=_resources(store_id),
        budget=_budget(budget_cents),
        title="sparc-steward run",
    )
    print(f"Session {session.id}: https://platform.claude.com/workspaces/default/sessions/{session.id}", flush=True)

    ended = "unknown"
    with client.beta.sessions.events.stream(session_id=session.id) as stream:
        client.beta.sessions.events.send(session_id=session.id, events=[_outcome()])
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, flush=True)
            elif event.type in ("agent.tool_use", "agent.mcp_tool_use") and event.evaluated_permission == "ask":
                print(f"[denied paused call: {getattr(event, 'name', '?')}]", flush=True)
                _deny(client, session.id, event.id)
            elif event.type == "span.outcome_evaluation_end":
                print(f"[grader, iteration {event.iteration}: {event.result}] {event.explanation}", flush=True)
            elif event.type == "session.status_terminated":
                ended = "terminated"
                break
            elif event.type == "session.status_idle" and event.stop_reason.type != "requires_action":
                ended = event.stop_reason.type
                break

    spent = int(client.beta.sessions.retrieve(session.id).usage.list_cost.amount)
    print(f"Session {session.id} ended ({ended}), spent ${spent / 100:.2f}", flush=True)
    return spent, ended


def burst(client: anthropic.Anthropic, total_cents: int, per_run_cents: int) -> None:
    spent = 0
    while total_cents - spent >= 500:  # don't start a run with less than $5 left
        cost, ended = run_once(client, min(per_run_cents, total_cents - spent))
        spent += cost
        print(f"=== total spent ${spent / 100:.2f} of ${total_cents / 100:.2f} ===", flush=True)
        if ended == "terminated":
            sys.exit("Session terminated; check it in the Console before starting another.")


def schedule(client: anthropic.Anthropic) -> None:
    if not hasattr(client.beta, "deployments"):
        sys.exit("This anthropic SDK has no deployments API; `pip install -U anthropic`.")
    agent_id, env_id, store_id = _ids()
    deployment = client.beta.deployments.create(
        name="sparc-steward",
        agent=agent_id,
        environment_id=env_id,
        resources=_resources(store_id),
        initial_events=[_outcome()],
        schedule=SCHEDULE,
        budget=_budget(SCHEDULED_BUDGET_CENTS),
    )
    client.beta.deployments.pause(deployment.id)  # live on create; stays paused until you unpause it
    print(f"DEPLOYMENT_ID={deployment.id} (paused). Next fire times: {deployment.schedule.upcoming_runs_at}")


def sweep(client: anthropic.Anthropic, deployment_id: str, last: int) -> None:
    """Deny calls left waiting for approval in the most recent scheduled runs."""
    for i, run in enumerate(client.beta.deployment_runs.list(deployment_id=deployment_id)):
        if i >= last:
            break
        if not run.session_id:
            print(f"{run.id}: failed to start ({run.error.type})")
            continue
        for event in client.beta.sessions.events.list(session_id=run.session_id):
            if event.type in ("agent.tool_use", "agent.mcp_tool_use") and getattr(event, "evaluated_permission", None) == "ask":
                try:
                    _deny(client, run.session_id, event.id)
                    print(f"{run.session_id}: denied {getattr(event, 'name', '?')}")
                except anthropic.BadRequestError:
                    pass  # already answered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("burst")
    b.add_argument("--total", type=float, required=True, help="dollars to spend across runs")
    b.add_argument("--per-run", type=float, default=48.0, help="dollar cap per run")
    r = sub.add_parser("run")
    r.add_argument("--budget", type=float, default=24.0, help="dollar cap for this run")
    sub.add_parser("schedule")
    s = sub.add_parser("sweep")
    s.add_argument("--deployment-id", default=os.getenv("DEPLOYMENT_ID"), required=not os.getenv("DEPLOYMENT_ID"))
    s.add_argument("--last", type=int, default=3)
    args = p.parse_args()

    client = anthropic.Anthropic()
    if args.cmd == "burst":
        burst(client, round(args.total * 100), round(args.per_run * 100))
    elif args.cmd == "run":
        run_once(client, round(args.budget * 100))
    elif args.cmd == "schedule":
        schedule(client)
    else:
        sweep(client, args.deployment_id, args.last)


if __name__ == "__main__":
    main()
