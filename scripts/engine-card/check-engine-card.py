#!/usr/bin/env python3
"""Check engine cards.

An engine card (engine-card.json at a package root) declares, per operation, what an engine
computes, where it is known to be right, what it costs, what it touches and how it is called.
Every claim points at an artifact in the repository that binds it to the implementation: a
JSON Schema, a contract test, a reference-comparison test, a benchmark command. This script
fails when a card does not match engine-card.schema.json, when its fields contradict each
other, or when something it points at does not exist.

Usage:
    python check-engine-card.py [--schema PATH] [--root DIR] [CARD ...]

CARD defaults to ./engine-card.json. Paths inside a card are relative to --root, which
defaults to the directory holding the card. --schema defaults to engine-card.schema.json
next to this script. Standard library only; no network. Exit status 1 on any failure.
"""

import argparse
import json
import re
import sys
from pathlib import Path

errors = []


def fail(where, msg):
    errors.append(f"{where}: {msg}")


# --- structure: the subset of JSON Schema 2020-12 that engine-card.schema.json uses ---------

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def _is_type(value, name):
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def _resolve(root_schema, ref):
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref!r}")
    node = root_schema
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def validate(value, schema, root_schema, where):
    """Validate value against schema, reporting through fail(). Supports the keywords the
    card schema uses: $ref, type, const, enum, required, properties, additionalProperties,
    items, minItems, minLength, pattern."""
    if "$ref" in schema:
        validate(value, _resolve(root_schema, schema["$ref"]), root_schema, where)
        return
    if "const" in schema and value != schema["const"]:
        fail(where, f"must be {schema['const']!r}, got {value!r}")
        return
    if "enum" in schema and value not in schema["enum"]:
        fail(where, f"must be one of {schema['enum']}, got {value!r}")
        return
    if "type" in schema and not _is_type(value, schema["type"]):
        fail(where, f"must be {schema['type']}, got {type(value).__name__}")
        return
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                fail(where, f"missing required field '{key}'")
        props = schema.get("properties", {})
        for key, sub in value.items():
            if key in props:
                validate(sub, props[key], root_schema, f"{where}.{key}")
            elif schema.get("additionalProperties", True) is False:
                fail(where, f"unknown field '{key}'")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            fail(where, f"must have at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], root_schema, f"{where}[{i}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            fail(where, "must not be empty")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            fail(where, f"must match {schema['pattern']}, got {value!r}")


# --- references: every path a card names must exist ----------------------------------------


def _path_part(ref):
    """'tests/x.rs::case' and 'docs/a.md#anchor' name a file; the selector is not checked."""
    return re.split(r"::|#", ref, maxsplit=1)[0]


def check_file(root, ref, where, counter):
    counter[0] += 1
    path = root / _path_part(ref)
    if not path.is_file():
        fail(where, f"points at '{ref}', which does not exist under {root}")
        return None
    return path


def check_contract_schema(root, ref, where, counter):
    path = check_file(root, ref, where, counter)
    if path is None:
        return
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(where, f"'{ref}' is not JSON: {exc}")
        return
    if not isinstance(doc, dict):
        fail(where, f"'{ref}' is not a JSON Schema object")
    elif "2020-12" not in str(doc.get("$schema", "")):
        fail(where, f"'{ref}' does not declare JSON Schema draft 2020-12 in $schema")


# --- rules a schema does not express --------------------------------------------------------

# How far a result can drift between two calls with the same input; an operation's own kind is
# its worst case, so no condition may be worse than it.
# Marks a card whose error codes are produced by the implementation (errors.source) rather than listed.
FROM_CATALOG = object()

_RANK = {"deterministic": 0, "seeded": 1, "host_dependent": 2, "time_bounded": 2, "nondeterministic": 3}


def check_determinism(kind, rnd, clock, det, where, root, counters):
    if kind == "deterministic" and (rnd != "none" or clock != "none"):
        fail(where, "'deterministic' needs randomness 'none' and clock 'none'")
    if kind == "seeded" and (rnd != "seeded" or clock != "none"):
        fail(where, "'seeded' needs randomness 'seeded' and clock 'none'")
    if rnd == "unseeded" and kind != "nondeterministic":
        fail(where, "unseeded randomness makes the operation 'nondeterministic'")
    if clock == "deadline" and kind not in ("time_bounded", "nondeterministic"):
        fail(where, "a deadline makes the result depend on machine speed: 'time_bounded'")
    if kind == "time_bounded" and clock != "deadline":
        fail(where, "'time_bounded' needs effects.clock 'deadline'")
    if kind in ("deterministic", "seeded"):
        if "test" not in det:
            fail(where, f"'{kind}' must point at the test that shows it (test)")
        else:
            check_file(root, det["test"], f"{where}.test", counters["files"])
    elif kind in ("host_dependent", "time_bounded") and not det.get("note"):
        fail(where, f"'{kind}' must say what the result depends on (note)")


def check_operation(op, where, root, codes, counters):
    hints, effects, det = op.get("hints", {}), op.get("effects", {}), op.get("determinism", {})
    kind, rnd, clock = det.get("kind"), effects.get("randomness"), effects.get("clock")

    # Contracts are bound to the implementation, not described beside it.
    for side in ("input", "output"):
        contract = op.get(side, {})
        ref = contract.get("schema")
        if isinstance(ref, str):
            check_contract_schema(root, ref, f"{where}.{side}.schema", counters["files"])
        if isinstance(contract.get("freshness_test"), str):
            check_file(root, contract["freshness_test"], f"{where}.{side}.freshness_test", counters["files"])
    if isinstance(op.get("contract_test"), str):
        check_file(root, op["contract_test"], f"{where}.contract_test", counters["files"])

    # Errors an operation raises are the engine's declared codes.
    for code in op.get("errors", []):
        if codes is None:
            fail(f"{where}.errors", "lists codes but the card has no top-level 'errors' declaration")
            break
        if codes is FROM_CATALOG:
            break  # the catalog is the list; its own test checks the codes, not this script
        if code not in codes:
            fail(f"{where}.errors", f"'{code}' is not declared in the top-level 'errors.codes'")

    # Effects and hints must say the same thing.
    counters["rules"][0] += 1
    if hints.get("read_only") and (effects.get("filesystem") == "write" or hints.get("destructive")):
        fail(f"{where}.hints", "read_only contradicts a filesystem write or destructive")
    if effects.get("network") == "optional" and not effects.get("network_off"):
        fail(f"{where}.effects", "network 'optional' must say how it is turned off (network_off)")
    if effects.get("network") in ("optional", "required") and not hints.get("open_world"):
        fail(f"{where}.hints", "an operation that can reach the network is open_world")
    if effects.get("network") == "none" and "network_off" in effects:
        fail(f"{where}.effects", "network_off given for an operation that never uses the network")

    # Determinism follows from randomness and the clock — for the operation as a whole (its worst
    # case) and for each input condition that narrows it.
    counters["rules"][0] += 1
    check_determinism(kind, rnd, clock, det, f"{where}.determinism", root, counters)
    for i, case in enumerate(det.get("cases", [])):
        cw = f"{where}.determinism.cases[{i}]"
        case_kind = case.get("kind")
        check_determinism(case_kind, case.get("randomness", rnd), case.get("clock", clock), case, cw, root, counters)
        if _RANK.get(case_kind, 0) > _RANK.get(kind, 0):
            fail(cw, f"'{case_kind}' is worse than the operation-level '{kind}', which must be the worst case")

    # Validation: declared empty, or each claim tied to a reference and a test.
    val = op.get("validation", {})
    checks, status = val.get("checks", []), val.get("status")
    counters["rules"][0] += 1
    if status in ("validated", "partial") and not checks:
        fail(f"{where}.validation", f"status '{status}' needs at least one check")
    if status == "none" and checks:
        fail(f"{where}.validation", "status 'none' cannot carry checks")
    if status == "partial" and not val.get("limits"):
        fail(f"{where}.validation", "status 'partial' must say what is not covered (limits)")
    for i, check in enumerate(checks):
        if isinstance(check, dict) and isinstance(check.get("test"), str):
            check_file(root, check["test"], f"{where}.validation.checks[{i}].test", counters["files"])

    # Performance: declared empty, or measured with hardware and a command that reproduces it.
    perf = op.get("performance", {})
    counters["rules"][0] += 1
    if perf.get("status") == "measured" and not perf.get("measurements"):
        fail(f"{where}.performance", "status 'measured' needs at least one measurement")
    if perf.get("status") == "none" and perf.get("measurements"):
        fail(f"{where}.performance", "status 'none' cannot carry measurements")


def check_card(card_path, schema, root):
    counters = {"files": [0], "rules": [0]}
    where = str(card_path)
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(where, f"cannot read the card: {exc}")
        return counters
    before = len(errors)
    validate(card, schema, schema, where)
    if len(errors) > before or not isinstance(card, dict):
        return counters  # cross-field rules assume the structure holds

    seen = set()
    for i, surface in enumerate(card["surfaces"]):
        key = (surface["kind"], surface["name"])
        if key in seen:
            fail(f"{where}.surfaces[{i}]", f"{key[0]} '{key[1]}' is listed twice")
        seen.add(key)
        if surface.get("wasm") and surface["kind"] not in ("npm", "other"):
            fail(f"{where}.surfaces[{i}]", "wasm is a property of an npm (or other) surface")

    codes = None
    if "errors" in card:
        declared = card["errors"]
        if "codes" not in declared and "source" not in declared:
            fail(f"{where}.errors", "must list the codes or point at the catalog that produces them (source)")
        if "source" in declared:
            check_file(root, declared["source"]["test"], f"{where}.errors.source.test", counters["files"])
        if "codes" in declared:
            codes = [c["code"] for c in declared["codes"]]
            for dup in sorted({c for c in codes if codes.count(c) > 1}):
                fail(f"{where}.errors", f"code '{dup}' is declared twice")
        else:
            codes = FROM_CATALOG

    ids = [op["id"] for op in card["operations"]]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        fail(f"{where}.operations", f"operation id '{dup}' is used twice")
    for i, op in enumerate(card["operations"]):
        check_operation(op, f"{where}.operations[{i}]({op['id']})", root, codes, counters)
    return counters


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cards", nargs="*", default=["engine-card.json"])
    parser.add_argument("--schema", default=str(here / "engine-card.schema.json"))
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    errors.clear()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    total = {"cards": 0, "files": 0, "rules": 0}
    for card in args.cards:
        card_path = Path(card)
        root = Path(args.root) if args.root else card_path.resolve().parent
        counters = check_card(card_path, schema, root)
        total["cards"] += 1
        total["files"] += counters["files"][0]
        total["rules"] += counters["rules"][0]

    print(f"Engine cards: {total['cards']} checked")
    print(f"Referenced files: {total['files']} checked")
    print(f"Cross-field rule groups: {total['rules']} checked")
    for line in errors:
        print(f"FAIL {line}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
