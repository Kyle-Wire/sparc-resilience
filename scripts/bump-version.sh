#!/usr/bin/env bash
# Usage: ./scripts/bump-version.sh 1.2.3
# Updates version in all places, commits, and pushes a tag that triggers the release CI.
#
# Pass a semver string for the desktop app (e.g. 1.0.0-beta, 1.2.3).
# pyproject.toml / sparc/__init__.py use the PEP-440 equivalent
# (hyphens removed: 1.0.0-beta → 1.0.0b1, 1.2.3 → 1.2.3).

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <new-version>  (e.g. 1.2.3 or 1.0.0-beta)"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Derive PEP-440 version: strip hyphen so 1.0.0-beta → 1.0.0beta → 1.0.0b1
# Simple rule: -beta → b1, -beta.N → bN, -alpha → a1, -rc.N → rcN
PY_VERSION=$(echo "$VERSION" \
  | sed 's/-beta\.\([0-9]*\)/b\1/' \
  | sed 's/-beta/b1/' \
  | sed 's/-alpha\.\([0-9]*\)/a\1/' \
  | sed 's/-alpha/a1/' \
  | sed 's/-rc\.\([0-9]*\)/rc\1/' \
  | sed 's/-rc/rc1/')

echo "Desktop version : $VERSION"
echo "Python version  : $PY_VERSION"

# Refuse to proceed if the tag already exists locally or on the remote.
# Without this guard, `git tag` silently fails (the script's set -e doesn't
# trip on the && short-circuit below) and `git push --tags` is a no-op,
# which means the release CI never runs and you're left wondering why no
# build appeared.
if git rev-parse --verify --quiet "refs/tags/v$VERSION" >/dev/null; then
  echo "Error: local tag v$VERSION already exists. Delete it first:"
  echo "  git tag -d v$VERSION"
  exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/v$VERSION" >/dev/null 2>&1; then
  echo "Error: remote tag v$VERSION already exists on origin. Delete it first:"
  echo "  git push origin :refs/tags/v$VERSION"
  exit 1
fi

# --- tauri.conf.json ---
sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"$VERSION\"/" \
  "$ROOT/sparc-desktop/src-tauri/tauri.conf.json"
rm "$ROOT/sparc-desktop/src-tauri/tauri.conf.json.bak"

# --- Cargo.toml ---
sed -i.bak "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" \
  "$ROOT/sparc-desktop/src-tauri/Cargo.toml"
rm "$ROOT/sparc-desktop/src-tauri/Cargo.toml.bak"

# --- package.json ---
sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"$VERSION\"/" \
  "$ROOT/sparc-desktop/package.json"
rm "$ROOT/sparc-desktop/package.json.bak"

# --- pyproject.toml ---
sed -i.bak "s/^version = \"[^\"]*\"/version = \"$PY_VERSION\"/" \
  "$ROOT/pyproject.toml"
rm "$ROOT/pyproject.toml.bak"

# --- sparc/__init__.py ---
sed -i.bak "s/^__version__ = \"[^\"]*\"/__version__ = \"$PY_VERSION\"/" \
  "$ROOT/sparc/__init__.py"
rm "$ROOT/sparc/__init__.py.bak"

# --- CITATION.cff ---
sed -i.bak "s/^version: \"[^\"]*\"/version: \"$PY_VERSION\"/" \
  "$ROOT/CITATION.cff"
rm "$ROOT/CITATION.cff.bak"

echo "Bumped all versions to $VERSION (Python: $PY_VERSION)"

# Commit + tag (skip commit if nothing changed — e.g. re-tagging same version)
git add \
  sparc-desktop/src-tauri/tauri.conf.json \
  sparc-desktop/src-tauri/Cargo.toml \
  sparc-desktop/package.json \
  pyproject.toml \
  sparc/__init__.py \
  CITATION.cff

git diff --cached --quiet || git commit -m "chore: bump version to $VERSION"
git tag "v$VERSION"

echo ""
echo "Run the following to trigger the release build:"
echo "  git push && git push --tags"
