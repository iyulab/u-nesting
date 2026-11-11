"""Smoke-test an installed ``u_nesting`` wheel before it is published.

Run this against a wheel that has just been *installed* (not against the source
tree): it imports the extension module, exercises both solve entry points, and
checks that the strict input contract is in force. A wheel that builds but
cannot be imported — or one whose native extension is missing from the archive —
fails here instead of on a user's first ``pip install``.

Usage:
    pip install --no-index --find-links dist u-nesting
    python scripts/python_wheel_smoke.py
"""

import sys


def main() -> int:
    import u_nesting

    version = u_nesting.version()
    strategies = u_nesting.available_strategies()
    print(f"imported u_nesting {version}; strategies: {strategies}")
    if not version:
        raise AssertionError("version() returned an empty string")
    if not strategies:
        raise AssertionError("available_strategies() returned an empty list")

    # The package re-exports the native module's functions by name; one that is
    # left out is documented but not callable as `u_nesting.<name>`.
    import u_nesting.u_nesting as native

    native_functions = {
        name
        for name in dir(native)
        if callable(getattr(native, name)) and not name.startswith("_")
    }
    missing = native_functions - set(u_nesting.__all__)
    if missing:
        raise AssertionError(
            f"the package does not export its native functions: {sorted(missing)}"
        )

    catalog = u_nesting.codes()
    if not catalog["refusals"] or not catalog["violations"]:
        raise AssertionError(f"codes() answered an empty catalog: {catalog}")
    print(
        f"codes(): {len(catalog['refusals'])} refusals, "
        f"{len(catalog['violations'])} violations"
    )

    result = u_nesting.solve_2d(
        geometries=[
            {
                "id": "rect",
                "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "quantity": 4,
            }
        ],
        boundary={"width": 400, "height": 300},
        config={"strategy": "nfp", "time_limit_ms": 5000},
    )
    if result["error"] is not None:
        raise AssertionError(f"solve_2d reported an error: {result['error']}")
    if not result["all_placed"]:
        raise AssertionError(
            f"solve_2d placed {len(result['placements'])} of {result['total_requested']}"
        )
    print(f"solve_2d placed {len(result['placements'])}/{result['total_requested']}")

    result = u_nesting.solve_3d(
        geometries=[{"id": "box", "dimensions": [100, 50, 30], "quantity": 4}],
        boundary={"dimensions": [400, 300, 200]},
        config={"strategy": "ep"},
    )
    if result["error"] is not None:
        raise AssertionError(f"solve_3d reported an error: {result['error']}")
    if not result["all_placed"]:
        raise AssertionError(
            f"solve_3d placed {len(result['placements'])} of {result['total_requested']}"
        )
    print(f"solve_3d placed {len(result['placements'])}/{result['total_requested']}")

    # Unknown keys must be rejected rather than silently ignored; a wheel built
    # against a stale contract would quietly accept them. A refusal comes back
    # in the result, named by the key's path -- not as an exception.
    result = u_nesting.solve_2d(
        geometries=[
            {
                "id": "rect",
                "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "quantiy": 4,
            }
        ],
        boundary={"width": 400, "height": 300},
        config=None,
    )
    if result["success"] or result["code"] != "malformed_input":
        raise AssertionError(f"a misspelled geometry key was not refused: {result}")
    print(f"unknown key rejected: {result['code']} at {result['details']['parameter']}")

    # A setting the strategy does not read is refused, not ignored.
    result = u_nesting.solve_2d(
        geometries=[{"id": "rect", "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]]}],
        boundary={"width": 400, "height": 300},
        config={"strategy": "blf", "population_size": 10},
    )
    if result["success"] or result["code"] != "invalid_option":
        raise AssertionError(f"a setting blf does not read was not refused: {result}")
    print(f"unread setting rejected: {result['details']['parameter']}")

    print("wheel smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
