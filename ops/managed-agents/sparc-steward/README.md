# sparc-steward (Claude Managed Agent)

An Anthropic-hosted agent that works through `docs/agent/ROADMAP.md` one
verified step per run and pushes each step to the `sparc-steward` branch. You
steer it by editing `ROADMAP.md` and `STATUS.md` on that branch.

| File | What it is |
|---|---|
| `agent.md` | Agent config (frontmatter) and system prompt (body) |
| `environment.yaml` | Cloud sandbox with network access limited to the data hosts SPARC uses, plus pip |
| `memory_store.yaml` | The agent's notes across runs |
| `rubric.md` | What a "done" run means; a grader checks each run against it |
| `steward.py` | Runtime: today's runs, the recurring schedule, the paused-call sweep |

## Money

- **Hard monthly stop:** in the Claude Console, set the workspace's monthly spend
  limit to $100. The per-run caps below are per session; the workspace limit is
  what holds the month.
- **Schedule:** daily at 22:13 New York time, $24 cap per run. The workspace's
  $100 monthly limit is what stops the month's runs; they start again when it
  resets. Edit `SCHEDULE` / `SCHEDULED_BUDGET_CENTS` in `steward.py`, then run
  `python steward.py schedule` with `DEPLOYMENT_ID` set to update the deployment.
- **A run that hits its cap pauses**; it doesn't lose anything it already pushed.

## One-time setup (run from the repo root)

```sh
# 1. CLI (needs ant >= 1.30.0) and login
ant --version || echo "install: https://github.com/anthropics/anthropic-cli/releases"
ant auth login
pip install -U anthropic

# 2. Agent, environment, memory store -> IDs land in ./claude-lock.json (commit it)
ant apply --dry-run -v ops/managed-agents/sparc-steward/agent.md ops/managed-agents/sparc-steward/environment.yaml ops/managed-agents/sparc-steward/memory_store.yaml
ant apply ops/managed-agents/sparc-steward/agent.md ops/managed-agents/sparc-steward/environment.yaml ops/managed-agents/sparc-steward/memory_store.yaml

# 3. GitHub token for the repo mount. Create a fine-grained token at
#    github.com/settings/personal-access-tokens: repository = kyle-wire/sparc-resilience only,
#    permission Contents: Read and write. It never enters the sandbox.
read -rs GITHUB_TOKEN && export GITHUB_TOKEN
```

On Windows (PowerShell), steps 1 and 3 are:

```powershell
$v = "2.2.0"; $dir = "$env:LOCALAPPDATA\Programs\ant"
New-Item -ItemType Directory -Force $dir | Out-Null
Invoke-WebRequest "https://github.com/anthropics/anthropic-cli/releases/download/v$v/ant_${v}_windows_amd64.zip" -OutFile "$env:TEMP\ant.zip"
Expand-Archive "$env:TEMP\ant.zip" -DestinationPath $dir -Force
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";$dir", "User")
$env:Path += ";$dir"
ant --version
ant auth login
$env:GITHUB_TOKEN = Read-Host "GitHub token" -MaskInput
```

Also protect `main` (and `pi-jepa-dev` if you like) under the repo's
Settings → Branches, so the token can only land work on `sparc-steward`.

## Runs

```sh
cd ops/managed-agents/sparc-steward
python steward.py run --budget 10                # first: a short run to check the sandbox works
python steward.py burst --total 85 --per-run 45  # then the rest of today's money, run after run
python steward.py schedule                       # creates the recurring deployment, paused
ant beta:deployments unpause --deployment-id <DEPLOYMENT_ID>   # when you're happy with the runs
```

Each run prints a Console link where you can watch it live.

**Paused calls.** The agent's tools run on the `auto` policy: the server runs
calls it judges safe, refuses high-risk ones, and pauses any it can't judge.
`steward.py run`/`burst` deny paused calls for you (the agent is told to find
another way or record a blocker). A scheduled run has no one attached, so a
paused call waits; `python steward.py sweep --deployment-id <id>` denies them
for the last few runs. Run it after each scheduled run, or ask for a webhook
handler to do it automatically.
