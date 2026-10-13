#!/usr/bin/env bash
# Smoke-test a packed NuGet package before it is published.
#
# Restores the freshly packed .nupkg into a throwaway console project from a
# local feed and runs a program that actually calls the library, so a package
# that restores but cannot load its native binary -- the failure mode that
# shipping native assets in `runtimes/{rid}/native/` invites -- fails here
# instead of on a consumer's first call.
#
# Usage: scripts/nuget_package_smoke.sh <nupkg-dir> <package-id> <version> <program.cs>
#
# <program.cs> is the console program to run (top-level statements); it throws
# on anything unexpected. This file is the same in every repository that packs
# a NuGet package; the program is each package's own.
set -euo pipefail

usage="usage: nuget_package_smoke.sh <nupkg-dir> <package-id> <version> <program.cs>"
NUPKG_DIR="${1:?$usage}"
PACKAGE="${2:?$usage}"
VERSION="${3:?$usage}"
PROGRAM="${4:?$usage}"
NUPKG_DIR="$(cd "$NUPKG_DIR" && pwd)"
PROGRAM="$(cd "$(dirname "$PROGRAM")" && pwd)/$(basename "$PROGRAM")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Isolate the global packages folder. Without this, NuGet resolves the version
# from its machine-wide cache whenever that version was restored before, and the
# gate silently validates a *previous* package instead of the one just packed --
# it would pass even for a package with no native binary at all.
export NUGET_PACKAGES="$WORK/packages"

cd "$WORK"
dotnet new console -o consumer >/dev/null
cd consumer

# Consume only the local feed, so the test can never silently succeed against an
# already-published version from nuget.org.
cat > nuget.config <<XML
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="${NUPKG_DIR}" />
  </packageSources>
</configuration>
XML

cp "$PROGRAM" Program.cs
dotnet add package "$PACKAGE" --version "$VERSION" --source "$NUPKG_DIR" >/dev/null
dotnet run --configuration Release
