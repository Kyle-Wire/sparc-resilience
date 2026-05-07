#!/usr/bin/env bash
# Usage: ./scripts/bump-version.sh 1.2.3
# Updates version in all 3 places, commits, and pushes a tag that triggers the release CI.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <new-version>  (e.g. 1.2.3)"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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

echo "Bumped all versions to $VERSION"

# Commit + tag (skip commit if nothing changed — e.g. re-tagging same version)
git add \
  sparc-desktop/src-tauri/tauri.conf.json \
  sparc-desktop/src-tauri/Cargo.toml \
  sparc-desktop/package.json

git diff --cached --quiet || git commit -m "chore: bump version to $VERSION"
git tag "v$VERSION"

echo ""
echo "Run the following to trigger the release build:"
echo "  git push && git push --tags"
