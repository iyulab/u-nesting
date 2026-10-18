#!/usr/bin/env bash
#
# Single source of truth for the release version is `[workspace.package] version`
# in the root Cargo.toml. Every release artifact must agree with it:
#
#   - internal crate deps in [workspace.dependencies] (u-nesting-core, -d2, ...)
#   - the C# binding (UNesting.csproj <Version>)
#   - the Python binding (crates/python/pyproject.toml — must inherit via
#     `dynamic = ["version"]`, or pin the workspace version explicitly)
#   - the engine card (engine-card.json — engine and surface versions)
#
# The npm package version is produced by wasm-pack from crates/wasm/Cargo.toml,
# which uses `version.workspace = true`, so it is always consistent and needs no
# check here.
#
# Run locally before pushing a version bump:
#   bash scripts/check-version-consistency.sh
#
# Enforced in CI (.github/workflows/ci.yml, "Version Consistency" job). Mismatches
# are emitted as GitHub Actions ::error:: annotations so they surface inline.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ws_version="$(grep -m1 -E '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
if [[ -z "${ws_version}" ]]; then
  echo "::error file=Cargo.toml::could not read [workspace.package] version"
  exit 1
fi
echo "workspace version: ${ws_version}"
status=0

# 1) Internal crate deps must pin the workspace version (kept in lockstep so the
#    crates.io dependency-ordered publish references the version being released).
while IFS= read -r line; do
  name="$(sed -E 's/^([A-Za-z0-9_-]+) =.*/\1/' <<<"${line}")"
  ver="$(sed -E 's/.*version = "([^"]+)".*/\1/' <<<"${line}")"
  if [[ "${ver}" != "${ws_version}" ]]; then
    echo "::error file=Cargo.toml::internal dependency '${name}' is ${ver}, expected ${ws_version} (keep [workspace.dependencies] in lockstep with [workspace.package])"
    status=1
  fi
#    The pattern deliberately has no trailing hyphen: an earlier `u-nesting-` form
#    matched every member crate except the `u-nesting` facade itself, and that pin
#    silently sat at 0.3.1 while the workspace was 0.9.0 with this check reporting
#    everything consistent.
done < <(grep -E '^u-nesting[A-Za-z0-9_-]* = \{ version = ' Cargo.toml)

# 2) C# binding.
csproj="bindings/csharp/UNesting/UNesting.csproj"
cs_version="$(grep -oE '<Version>[^<]+</Version>' "${csproj}" | sed -E 's#</?Version>##g')"
if [[ "${cs_version}" != "${ws_version}" ]]; then
  echo "::error file=${csproj}::<Version>${cs_version}</Version>, expected ${ws_version}"
  status=1
fi

# 3) Python binding — version must come from Cargo.toml (dynamic), otherwise the
#    explicit pin must match. A stale static version here would ship a wrong wheel.
pyproject="crates/python/pyproject.toml"
if grep -qE '^[[:space:]]*dynamic[[:space:]]*=.*"version"' "${pyproject}"; then
  : # inherited from Cargo.toml via maturin — always consistent
else
  py_version="$(grep -m1 -E '^version = ' "${pyproject}" | sed -E 's/version = "(.*)"/\1/')"
  if [[ "${py_version}" != "${ws_version}" ]]; then
    echo "::error file=${pyproject}::version = \"${py_version}\", expected ${ws_version} (or declare dynamic = [\"version\"] to inherit from Cargo.toml)"
    status=1
  fi
fi

# 4) The changelog must have gained a heading for the version being released. A
#    bump whose entry is still sitting under `## [Unreleased]` publishes a version
#    whose consumers have no record of what they upgraded into, and none of the
#    checks above would notice: every version string can agree while the changelog
#    says nothing.
changelog="CHANGELOG.md"
if [[ -f "${changelog}" ]]; then
  if grep -qE "^## \[${ws_version//./\\.}\]" "${changelog}"; then
    echo "Changelog has an entry for ${ws_version}"
  else
    echo "::error file=${changelog}::no '## [${ws_version}]' heading — add the entry for this release in the same commit as the version bump (move it out of '## [Unreleased]')"
    status=1
  fi
fi

# 5) The engine card names the engine version and each surface's package and
#    engine versions; every surface here is versioned with the workspace. Its
#    surfaces once sat four releases behind while engine.version was kept up.
card="engine-card.json"
if [[ -f "${card}" ]]; then
  while IFS= read -r field; do
    ver="$(sed -E 's/.*: "([^"]+)"/\1/' <<<"${field}")"
    if [[ "${ver}" != "${ws_version}" ]]; then
      echo "::error file=${card}::${field}, expected ${ws_version}"
      status=1
    fi
  done < <(grep -oE '"(version|package_version|engine_version)": "[^"]+"' "${card}")
fi

if [[ "${status}" -ne 0 ]]; then
  echo "::error::Release artifacts disagree — bring every version string and the changelog to ${ws_version} together before publishing."
  exit 1
fi
echo "All binding versions consistent at ${ws_version}"
