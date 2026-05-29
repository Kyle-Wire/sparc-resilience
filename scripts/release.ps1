#!/usr/bin/env pwsh
# release.ps1  — SPARC version-bump and release-cleanup CLI
#
# USAGE
#   .\scripts\release.ps1 <version>    Bump all version files, commit, push, tag → triggers CI
#   .\scripts\release.ps1 clear        Delete every release & tag in sparc-resilience and sparc-status
#   .\scripts\release.ps1 help         Show this help

param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$Root  = Resolve-Path "$PSScriptRoot\.."
$Owner = "Kyle-Wire"
$Repos = @("sparc-resilience", "sparc-status")

# ──────────────────────────────────────────────────────────────────────────────
function Show-Help {
    Write-Host ""
    Write-Host "  SPARC Release CLI" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  USAGE:" -ForegroundColor Yellow
    Write-Host "    .\scripts\release.ps1 <version>    Bump version, commit, push + tag"
    Write-Host "    .\scripts\release.ps1 clear        Delete all releases and tags from both repos"
    Write-Host "    .\scripts\release.ps1 help         Show this help"
    Write-Host ""
    Write-Host "  EXAMPLES:" -ForegroundColor Yellow
    Write-Host "    .\scripts\release.ps1 1.0.11"
    Write-Host "    .\scripts\release.ps1 clear"
    Write-Host ""
}

# ──────────────────────────────────────────────────────────────────────────────
function Update-File {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Replacement,
        [string]$Label
    )
    $lines = Get-Content $Path
    $updated = $lines -replace $Pattern, $Replacement
    Set-Content $Path $updated
    Write-Host "  [ok] $Label"
}

# ──────────────────────────────────────────────────────────────────────────────
function Invoke-Bump {
    param([string]$Version)

    if ($Version -notmatch '^\d+\.\d+\.\d+') {
        Write-Error "Version '$Version' must be semver (e.g. 1.2.3). Aborting."
        exit 1
    }

    Set-Location $Root

    # Guard: tag must not already exist
    git rev-parse --verify --quiet "refs/tags/v$Version" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Error "Local tag v$Version already exists. Delete it first:  git tag -d v$Version"
        exit 1
    }
    git ls-remote --exit-code --tags origin "refs/tags/v$Version" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Error "Remote tag v$Version already exists. Delete it first:  git push origin :refs/tags/v$Version"
        exit 1
    }

    $today = Get-Date -Format "yyyy-MM-dd"
    Write-Host ""
    Write-Host "Bumping to v$Version..." -ForegroundColor Cyan

    Update-File "pyproject.toml"       '^version = ".*"'          "version = `"$Version`""          "pyproject.toml"
    Update-File "sparc\__init__.py"    '^__version__ = ".*"'      "__version__ = `"$Version`""      "sparc/__init__.py"
    Update-File "sparc\server\app.py"  '^\s{4}version=".*",'      "    version=`"$Version`","        "sparc/server/app.py"
    Update-File "CITATION.cff"         '^version: ".*"'           "version: `"$Version`""           "CITATION.cff (version)"
    Update-File "CITATION.cff"         '^date-released: ".*"'     "date-released: `"$today`""       "CITATION.cff (date)"

    # sparc-desktop (optional — skip gracefully if files don't exist)
    $tauriConf  = "sparc-desktop\src-tauri\tauri.conf.json"
    $cargoToml  = "sparc-desktop\src-tauri\Cargo.toml"
    $packageJson = "sparc-desktop\package.json"

    if (Test-Path $tauriConf)  { Update-File $tauriConf  '"version": ".*",'    "`"version`": `"$Version`","   "sparc-desktop/tauri.conf.json" }
    if (Test-Path $cargoToml)  { Update-File $cargoToml  '^version = ".*"'     "version = `"$Version`""       "sparc-desktop/Cargo.toml" }
    if (Test-Path $packageJson){ Update-File $packageJson '"version": ".*",'   "`"version`": `"$Version`","   "sparc-desktop/package.json" }

    # Stage everything we touched
    $filesToStage = @("pyproject.toml", "sparc\__init__.py", "sparc\server\app.py", "CITATION.cff")
    if (Test-Path $tauriConf)  { $filesToStage += $tauriConf }
    if (Test-Path $cargoToml)  { $filesToStage += $cargoToml }
    if (Test-Path $packageJson){ $filesToStage += $packageJson }

    git add $filesToStage

    $staged = git diff --cached --stat
    if ($staged) {
        git commit -m "chore: bump version to $Version" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
        Write-Host ""
        Write-Host "  [committed] chore: bump version to $Version" -ForegroundColor Green
    } else {
        Write-Host "  [skip] no changes to commit" -ForegroundColor Yellow
    }

    git tag "v$Version"
    git push
    git push --tags

    Write-Host ""
    Write-Host "  v$Version released — tag pushed, CI should trigger now." -ForegroundColor Green
    Write-Host ""
}

# ──────────────────────────────────────────────────────────────────────────────
function Invoke-Clear {
    Write-Host ""
    Write-Host "Clearing all releases and tags from: $($Repos -join ', ')" -ForegroundColor Yellow
    Write-Host ""

    foreach ($repo in $Repos) {
        Write-Host "[$repo]" -ForegroundColor Cyan

        # Delete all releases (--cleanup-tag removes the associated tag too)
        $releases = gh release list --repo "$Owner/$repo" --limit 200 --json tagName 2>$null | ConvertFrom-Json
        if ($releases -and $releases.Count -gt 0) {
            foreach ($rel in $releases) {
                Write-Host "  Deleting release $($rel.tagName)..."
                gh release delete $rel.tagName --repo "$Owner/$repo" --yes --cleanup-tag 2>$null
            }
            Write-Host "  [ok] deleted $($releases.Count) release(s)" -ForegroundColor Green
        } else {
            Write-Host "  [skip] no releases found"
        }

        # Delete any tags that survived (e.g. tags without a matching release)
        $tags = gh api "repos/$Owner/$repo/tags" --paginate 2>$null | ConvertFrom-Json
        if ($tags -and $tags.Count -gt 0) {
            foreach ($tag in $tags) {
                Write-Host "  Deleting tag $($tag.name)..."
                gh api -X DELETE "repos/$Owner/$repo/git/refs/tags/$($tag.name)" 2>$null
            }
            Write-Host "  [ok] deleted $($tags.Count) orphaned tag(s)" -ForegroundColor Green
        } else {
            Write-Host "  [skip] no orphaned tags"
        }

        Write-Host ""
    }

    Write-Host "Done — all releases and tags cleared." -ForegroundColor Green
    Write-Host ""
}

# ──────────────────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "help"   { Show-Help }
    "-h"     { Show-Help }
    "--help" { Show-Help }
    "clear"  { Invoke-Clear }
    default  { Invoke-Bump -Version $Command }
}
