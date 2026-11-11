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

import contextlib
import io
import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "crates" / "python" / "README.md"


def run_readme_examples() -> int:
    """Run the Python examples the README teaches, against the installed wheel.

    An example that imports the package is self-contained and runs as written; a
    README whose examples run is one whose examples a reader can paste. A README
    with no such example fails: nothing proves the examples still run.
    """
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\r?\n(.*?)^```", text, flags=re.S | re.M)
    ran = 0
    for number, code in enumerate(blocks, start=1):
        if not re.search(r"^\s*(import u_nesting|from u_nesting)\b", code, re.M):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(code, f"README python block {number}", "exec"), {})
            except Exception as error:  # noqa: BLE001 -- name the example, whatever failed
                raise AssertionError(
                    f"README python block {number} does not run: "
                    f"{type(error).__name__}: {error}\n{code}"
                ) from error
        ran += 1
    if ran == 0:
        raise AssertionError(f"{README} has no self-contained python example to run")
    return ran


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

    problem_2d = dict(
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
    result = u_nesting.solve_2d(**problem_2d)
    if result["error"] is not None:
        raise AssertionError(f"solve_2d reported an error: {result['error']}")
    if not result["all_placed"]:
        raise AssertionError(
            f"solve_2d placed {len(result['placements'])} of {result['total_requested']}"
        )
    print(f"solve_2d placed {len(result['placements'])}/{result['total_requested']}")

    # The result is the canonical response: the keys every binding answers.
    first = result["placements"][0]
    for key in ("id", "x", "y", "rotation", "sheet_index", "flipped"):
        if key not in first:
            raise AssertionError(f"a 2D placement lacks {key!r}: {first}")
    if "sheets_used" not in result or "elapsed_ms" not in result:
        raise AssertionError(f"solve_2d answered a shape of its own: {sorted(result)}")

    # The layout it answered is feasible by the checker; two pieces on one spot are not.
    answer = u_nesting.check_2d(solve_result=result, **problem_2d)
    if not (answer["success"] and answer["feasible"]):
        raise AssertionError(f"check_2d judged its own solve infeasible: {answer}")
    broken = {**result, "placements": [dict(p) for p in result["placements"]]}
    broken["placements"][1].update(x=first["x"], y=first["y"])
    answer = u_nesting.check_2d(solve_result=broken, **problem_2d)
    if not answer["success"] or answer["feasible"] or not answer["violations"]:
        raise AssertionError(f"check_2d missed two pieces on one spot: {answer}")
    print(f"check_2d: feasible, and {answer['violations'][0]['code']} on a broken layout")

    # The cutting path of the layout: one step per contour, nothing skipped.
    cut = u_nesting.optimize_cutting_path(problem_2d["geometries"], result)
    if not cut["success"] or not cut["sequence"] or cut["skipped_contours"]:
        raise AssertionError(f"optimize_cutting_path failed on its own solve: {cut}")
    if not cut["total_cut_distance"] > 0:
        raise AssertionError(f"a cutting path without cut distance: {cut}")
    print(f"optimize_cutting_path: {len(cut['sequence'])} steps, {cut['total_pierces']} pierces")
    # Refused, not raised: a setting out of range is named by its path.
    cut = u_nesting.optimize_cutting_path(
        problem_2d["geometries"], result, {"pierce_candidates": 0}
    )
    if cut["success"] or cut["details"]["parameter"] != "cutting_config.pierce_candidates":
        raise AssertionError(f"a pierce_candidates of 0 was not refused by path: {cut}")

    problem_3d = dict(
        geometries=[{"id": "box", "dimensions": [100, 50, 30], "quantity": 4}],
        boundary={"dimensions": [400, 300, 200]},
        config={"strategy": "ep"},
    )
    result = u_nesting.solve_3d(**problem_3d)
    if result["error"] is not None:
        raise AssertionError(f"solve_3d reported an error: {result['error']}")
    if not result["all_placed"]:
        raise AssertionError(
            f"solve_3d placed {len(result['placements'])} of {result['total_requested']}"
        )
    print(f"solve_3d placed {len(result['placements'])}/{result['total_requested']}")
    answer = u_nesting.check_3d(solve_result=result, **problem_3d)
    if not (answer["success"] and answer["feasible"]):
        raise AssertionError(f"check_3d judged its own solve infeasible: {answer}")
    print("check_3d: feasible")

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

    print(f"README examples ran: {run_readme_examples()}")
    print("wheel smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
