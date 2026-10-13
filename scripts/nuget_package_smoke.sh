#!/usr/bin/env bash
# Smoke-test a packed UNesting NuGet package before it is published.
#
# Restores the freshly packed .nupkg into a throwaway console project from a
# local feed and actually calls the library, so a package that restores but
# cannot load its native binary — the failure mode that shipping native assets
# in `runtimes/{rid}/native/` invites — fails here instead of on a consumer's
# first call.
#
# Usage: scripts/nuget_package_smoke.sh <nupkg-dir> <version>
set -euo pipefail

NUPKG_DIR="${1:?usage: nuget_package_smoke.sh <nupkg-dir> <version>}"
VERSION="${2:?usage: nuget_package_smoke.sh <nupkg-dir> <version>}"
NUPKG_DIR="$(cd "$NUPKG_DIR" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Isolate the global packages folder. Without this, NuGet resolves the version
# from its machine-wide cache whenever that version was restored before, and the
# gate silently validates a *previous* package instead of the one just packed —
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

cat > Program.cs <<'CSHARP'
using UNesting;
using UNesting.Models;

Console.WriteLine($"native version: {UNestingInfo.Version}");

var request = new NestingRequest
{
    Geometries = { Geometry2D.Rectangle("rect", 100, 50, quantity: 4) },
    Boundary = new Boundary2D { Width = 400, Height = 300 },
    Config = new Config2D { Strategy = Strategy2D.NfpGuided, TimeLimitMs = 5000 },
};

using var nester = new Nester2D();
var result = nester.Solve(request);

Console.WriteLine($"all_placed={result.AllPlaced} placed={result.Placements.Count} " +
                  $"sheets={result.SheetsUsed} utilization={result.Utilization:F4}");

if (!result.AllPlaced)
{
    throw new Exception("not every part was placed");
}

if (result.Placements.Count != 4)
{
    throw new Exception($"expected 4 placements, got {result.Placements.Count}");
}

Console.WriteLine("nuget package smoke test passed");
CSHARP

dotnet add package UNesting --version "$VERSION" --source "$NUPKG_DIR" >/dev/null
dotnet run --configuration Release
