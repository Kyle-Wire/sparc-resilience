# To Bump Version:

& "C:\Program Files\Git\bin\bash.exe" ./scripts/bump-version.sh 1.x.x
git push && git push --tags

# Delete all v1.* tags locally
git tag --list "v1.*" | ForEach-Object { git tag -d $_ }

# Delete all v1.* tags from remote
git tag --list "v1.*" | ForEach-Object { git push origin ":refs/tags/$_" }

# NUKE Local
git tag | ForEach-Object { git tag -d $_ }

# NUKE Remote
git tag | ForEach-Object { git push origin ":refs/tags/$_" }

# Fresh-start at beta_1.0.0 (current):
#   1. Nuke all old tags (local + remote) using commands above
#   2. Files already updated to 1.0.0-beta (desktop) / 1.0.0b1 (Python)
#   3. Commit + tag + push:
#        git add -A
#        git commit -m "chore: fresh start at v1.0.0-beta"
#        git tag v1.0.0-beta
#        git push && git push --tags