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