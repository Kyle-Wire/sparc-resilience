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

# Commit + tag
cd "$ROOT"
git add \
  sparc-desktop/src-tauri/tauri.conf.json \
  sparc-desktop/src-tauri/Cargo.toml \
  sparc-desktop/package.json

git commit -m "chore: bump version to $VERSION"
git tag "v$VERSION"

echo ""
echo "Run the following to trigger the release build:"
echo "  git push && git push --tags"
