#!/usr/bin/env bash
# Fails when a generated TypeScript declaration does not describe what the
# binding returns:
#
#   1. an exported function whose return type is `any`
#   2. an exported function taking a parameter typed `any` (or `any | ...`)
#   3. a type named in a signature or a field that the file never declares
#
# (1) is the reported harm: `as` is the only thing a consumer can write against
# `any`, and `as` is exactly the construct that silences a wrong assumption
# about a result's shape -- a renderer read a per-point boolean mask as an
# array of indices, compiled, shipped, and printed the same subgroup twelve
# times. (2) is how (1) comes back disguised: the declaration names a type that
# does not exist, because the struct behind it is missing its derive, and the
# consumer's build then fails on our file instead of on their mistake.
#
# (2) is the same harm on the way in: `estimate_effects(full_factorial(3), ...)`
# compiled under strict TypeScript -- the generator's result object handed to
# a parameter that wants its `data` matrix -- and failed only at run time.
#
# This runs on the publish path, not only in CI: the two run side by side on
# the same push, so a check that lives only in CI cannot stop a publish.
#
# Usage: check-typed-dts.sh <file.d.ts> [more.d.ts ...]
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <file.d.ts> [more.d.ts ...]" >&2
    exit 2
fi

# Types a declaration may name without declaring them.
builtins='number|string|boolean|any|void|bigint|symbol|unknown|never|null|undefined|object|Array|Promise|Record|Map|Set|Date|Uint8Array|Uint16Array|Uint32Array|Int8Array|Int16Array|Int32Array|Float32Array|Float64Array|BigInt64Array|BigUint64Array|ArrayBuffer|Function|Error|readonly'

status=0
checked=0
for dts in "$@"; do
    if [ ! -f "$dts" ]; then
        echo "FAIL: $dts does not exist" >&2
        status=1
        continue
    fi

    # wasm-bindgen emits a second declaration file beside the package's own --
    # `<name>_bg.wasm.d.ts` -- describing the raw module's ABI: `export const`
    # bindings with pointer-level signatures, and the module's memory. It is
    # not a public API surface and declares no `export function` by design, so
    # it is skipped. Recognised by that memory export rather than by filename,
    # so a renamed or relocated file is still classified correctly.
    if grep -q '^export const memory: WebAssembly.Memory;' "$dts"; then
        continue
    fi

    total=$(grep -c '^export function' "$dts")
    if [ "$total" -eq 0 ]; then
        echo "FAIL: $dts declares no exported functions -- generated from the wrong build?" >&2
        status=1
        continue
    fi
    checked=$((checked + 1))

    untyped=$(grep '^export function' "$dts" | grep -E '\):[[:space:]]*any;')
    if [ -n "$untyped" ]; then
        count=$(printf '%s\n' "$untyped" | wc -l | tr -d ' ')
        echo "FAIL: $dts returns \`any\` from $count of $total exported function(s):" >&2
        printf '%s\n' "$untyped" | sed 's/^/  /' >&2
        echo "  Derive the declaration from the result struct (tsify) and name it in" >&2
        echo "  #[wasm_bindgen(unchecked_return_type = \"...\")]." >&2
        status=1
    fi

    # The parameter list of each exported function: everything between the
    # function's name and the `):` that opens its return type.
    signatures=$(grep -E '^export function' "$dts" | sed -E 's/^export function [A-Za-z_0-9]*\((.*)\):[^)]*$/\1/')

    untyped_params=$(grep -E '^export function' "$dts" | grep -E '[(,][[:space:]]*[A-Za-z_][A-Za-z_0-9]*\??:[[:space:]]*any([[:space:]]*\|[^,)]*)?[,)]')
    if [ -n "$untyped_params" ]; then
        count=$(printf '%s\n' "$untyped_params" | wc -l | tr -d ' ')
        echo "FAIL: $dts takes an \`any\` parameter in $count of $total exported function(s):" >&2
        printf '%s\n' "$untyped_params" | sed 's/^/  /' >&2
        echo "  Name the parameter's type in #[wasm_bindgen(unchecked_param_type = \"...\")]" >&2
        echo "  (unchecked_optional_param_type for an Option) on the parameter." >&2
        status=1
    fi

    # Every capitalised name a signature or a field mentions has to be declared
    # in the same file. String literals are removed first: a literal union such
    # as `"Maximize" | "Minimize"` names values, not types.
    declared=$(grep -oE '^export (interface|type|class|enum) [A-Za-z_][A-Za-z_0-9]*' "$dts" | awk '{print $3}' | sort -u)
    returns=$(grep -E '^export function' "$dts" | sed -E 's/.*\):[[:space:]]*//')
    fields=$(grep -E '^    [a-z_][a-z_0-9]*\??:' "$dts" | sed -E 's/^[^:]*:[[:space:]]*//')
    referenced=$(printf '%s\n%s\n%s\n' "$returns" "$fields" "$signatures" \
        | sed -E 's/"[^"]*"//g' \
        | grep -oE '[A-Za-z_][A-Za-z_0-9]*' \
        | grep -E '^[A-Z]' \
        | grep -vE "^($builtins)$" \
        | sort -u)

    # Membership by hand: `comm` wants process substitution, which is not
    # dependable in every shell this has to run in.
    missing=''
    for name in $referenced; do
        if ! printf '%s\n' "$declared" | grep -Fxq "$name"; then
            missing="$missing $name"
        fi
    done
    if [ -n "${missing// /}" ]; then
        echo "FAIL: $dts names types it does not declare:" >&2
        for name in $missing; do
            echo "  $name" >&2
        done
        echo "  A struct reached through a declared one needs the derive too." >&2
        status=1
    fi

    if [ -z "$untyped" ] && [ -z "$untyped_params" ] && [ -z "${missing// /}" ]; then
        echo "OK: $dts -- $total exported function(s), every parameter and return type declared"
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "FAIL: none of the given files declares a public API surface -- every one" >&2
    echo "  looked like a raw-module shim. Point this at the package's own .d.ts." >&2
    status=1
fi

exit "$status"
