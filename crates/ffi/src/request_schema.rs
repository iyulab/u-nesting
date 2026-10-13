//! The published JSON schemas (`docs/json-schema`) held to what the engine
//! reads and writes: at every level of a request the schema's keys are the
//! keys the request types take, its enum literals are the names a solve
//! accepts, and the 2D response schema has the keys a response carries.
//!
//! What a request "accepts" is measured by solving: a key or name is accepted
//! when a request with it solves, so a setting a dimension refuses (3D
//! `multi_sheet`, a 3D-only strategy in 2D, an exact solver this build does
//! not carry) is not one its schema may list.

use crate::api::{solve_2d_internal, solve_3d_internal};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use u_nesting_core::api_types::{ConfigRequest, RefusalDetails, SolveResponse};
use u_nesting_core::solver::Strategy;
use u_nesting_core::Error;
use u_nesting_d3::OrientationConstraint;

fn schema(name: &str) -> Value {
    let path = format!(
        "{}/../../docs/json-schema/{name}.schema.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// `node`, or the definition its `$ref` points at.
fn resolve<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    match node.get("$ref").and_then(Value::as_str) {
        Some(r) => {
            let name = r.strip_prefix("#/$defs/").expect("a local definition");
            &root["$defs"][name]
        }
        None => node,
    }
}

/// The keys an object schema defines -- its own `properties` and those of
/// each `oneOf` branch.
fn schema_keys(node: &Value) -> BTreeSet<String> {
    let mut keys: BTreeSet<String> = node
        .get("properties")
        .and_then(Value::as_object)
        .map(|p| p.keys().cloned().collect())
        .unwrap_or_default();
    for branch in node
        .get("oneOf")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        keys.extend(schema_keys(branch));
    }
    keys
}

fn property<'a>(root: &'a Value, node: &'a Value, key: &str) -> &'a Value {
    let node = resolve(root, node);
    let direct = node.get("properties").and_then(|p| p.get(key));
    let in_branch = || {
        node["oneOf"]
            .as_array()?
            .iter()
            .find_map(|b| b.get("properties").and_then(|p| p.get(key)))
    };
    resolve(root, direct.or_else(in_branch).unwrap_or(&Value::Null))
}

fn literals(node: &Value) -> BTreeSet<String> {
    node["enum"]
        .as_array()
        .unwrap_or_else(|| panic!("no enum in {node}"))
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect()
}

fn object_keys(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .unwrap_or_else(|| panic!("not an object: {v}"))
        .keys()
        .cloned()
        .collect()
}

/// A request's object keys, level by level, each against the schema's
/// definition of that level.
fn assert_levels(root: &Value, given: &Value, levels: &[(&str, &Value, &Value)]) {
    assert_eq!(
        object_keys(given),
        schema_keys(root),
        "top level of {}",
        root["title"]
    );
    for (name, value, node) in levels {
        assert_eq!(
            object_keys(value),
            schema_keys(resolve(root, node)),
            "{name} in {}",
            root["title"]
        );
    }
}

/// One valid value for every key a request's `config` takes.
fn every_setting() -> Map<String, Value> {
    let settings = json!({
        "spacing": 0.0, "margin": 0.0, "strategy": "blf", "time_limit_ms": 1000,
        "target_utilization": 0.9, "population_size": 4, "max_generations": 1,
        "crossover_rate": 0.8, "mutation_rate": 0.1, "seed": 1, "multi_sheet": false
    });
    let settings = settings.as_object().expect("an object").clone();
    let keys: BTreeSet<&str> = settings.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        ConfigRequest::KEYS.into_iter().collect(),
        "a setting was added: give it a value here"
    );
    settings
}

fn request_2d(config: Value) -> Value {
    json!({
        "geometries": [{"id": "plate", "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]]}],
        "boundary": {"width": 50, "height": 50},
        "config": config
    })
}

fn request_3d(config: Value, orientation: &str) -> Value {
    json!({
        "geometries": [{"id": "crate", "dimensions": [2, 3, 4], "orientation": orientation}],
        "boundary": {"dimensions": [10, 10, 10]},
        "config": config
    })
}

fn solves_2d(request: &Value) -> bool {
    solve_2d_internal(&request.to_string()).success
}

fn solves_3d(request: &Value) -> bool {
    solve_3d_internal(&request.to_string()).success
}

/// A request with every key the 2D schema defines is read, and read back
/// has those keys and no others, at every level.
#[test]
fn the_2d_request_schema_has_the_keys_the_request_takes() {
    let root = schema("request-2d");
    let full = json!({
        "version": "1.0", "mode": "2d",
        "geometries": [{
            "id": "plate", "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
            "holes": [[[1, 1], [2, 1], [2, 2]]], "quantity": 1,
            "rotations": [0, 90], "allow_flip": false
        }],
        "boundary": {"width": 50, "height": 50},
        "config": Value::Object(every_setting())
    });
    let defs = &root["properties"];
    assert_levels(
        &root,
        &full,
        &[
            (
                "geometries[]",
                &full["geometries"][0],
                &defs["geometries"]["items"],
            ),
            ("config", &full["config"], &defs["config"]),
        ],
    );
    // The boundary's two shapes, together.
    let both = json!({"width": 1, "height": 1, "polygon": []});
    assert_eq!(
        object_keys(&both),
        schema_keys(resolve(&root, &defs["boundary"]))
    );

    let read: u_nesting_core::api_types::Request2D =
        u_nesting_core::api_types::from_json(&full.to_string()).expect("every key is read");
    let back = serde_json::to_value(&read).expect("serialize");
    assert_eq!(object_keys(&back), object_keys(&full));
    assert_eq!(
        object_keys(&back["geometries"][0]),
        object_keys(&full["geometries"][0])
    );
    assert_eq!(object_keys(&back["boundary"]), object_keys(&both));
    assert_eq!(object_keys(&back["config"]), object_keys(&full["config"]));
    assert!(solves_2d(&full), "the full request solves");
}

#[test]
fn the_3d_request_schema_has_the_keys_the_request_takes() {
    let root = schema("request-3d");
    let mut config = every_setting();
    config.remove("multi_sheet");
    let full = json!({
        "version": "1.0", "mode": "3d",
        "geometries": [{
            "id": "crate", "dimensions": [2, 3, 4], "quantity": 1, "mass": 1.0,
            "orientation": "any"
        }],
        "boundary": {"dimensions": [10, 10, 10], "max_mass": 100.0,
                     "gravity": false, "stability": false},
        "config": Value::Object(config)
    });
    let defs = &root["properties"];
    assert_levels(
        &root,
        &full,
        &[
            (
                "geometries[]",
                &full["geometries"][0],
                &defs["geometries"]["items"],
            ),
            ("boundary", &full["boundary"], &defs["boundary"]),
            ("config", &full["config"], &defs["config"]),
        ],
    );

    let read: u_nesting_core::api_types::Request3D =
        u_nesting_core::api_types::from_json(&full.to_string()).expect("every key is read");
    let back = serde_json::to_value(&read).expect("serialize");
    assert_eq!(object_keys(&back), object_keys(&full));
    assert_eq!(
        object_keys(&back["geometries"][0]),
        object_keys(&full["geometries"][0])
    );
    assert_eq!(
        object_keys(&back["boundary"]),
        object_keys(&full["boundary"])
    );
    assert!(solves_3d(&full), "the full request solves");
}

/// The settings each schema lists are the settings a solve of that
/// dimension accepts -- the `config` type is shared, so the 3D one takes
/// `multi_sheet` and has to refuse it rather than ignore it.
#[test]
fn each_config_schema_lists_the_settings_its_dimension_accepts() {
    let settings = every_setting();
    let accepted = |solves: &dyn Fn(&Value) -> bool| -> BTreeSet<String> {
        settings
            .iter()
            .filter(|(key, value)| solves(&json!({ key.as_str(): value })))
            .map(|(key, _)| key.clone())
            .collect()
    };
    let d2 = schema("request-2d");
    let d3 = schema("request-3d");
    assert_eq!(
        schema_keys(resolve(&d2, &d2["properties"]["config"])),
        accepted(&|c| solves_2d(&request_2d(c.clone())))
    );
    assert_eq!(
        schema_keys(resolve(&d3, &d3["properties"]["config"])),
        accepted(&|c| solves_3d(&request_3d(c.clone(), "any")))
    );

    let refused = solve_3d_internal(&request_3d(json!({"multi_sheet": false}), "any").to_string());
    assert_eq!(refused.code.as_deref(), Some("invalid_option"));
    assert_eq!(
        refused.details.and_then(|d| d.parameter).as_deref(),
        Some("config.multi_sheet")
    );
}

/// The strategy names each schema lists are exactly the spellings a solve of
/// that dimension runs, in this build: the exact solvers need a feature the
/// published libraries do not carry, so their names are refused here and
/// left out of the schemas.
#[test]
fn each_strategy_enum_is_what_its_dimension_runs() {
    let runs = |solves: &dyn Fn(&str) -> bool| -> BTreeSet<String> {
        Strategy::SPELLINGS
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| solves(name))
            .map(str::to_string)
            .collect()
    };
    let d2 = schema("request-2d");
    let d3 = schema("request-3d");
    let config = |root: &Value| property(root, &root["properties"]["config"], "strategy").clone();
    // A build with the `milp` feature (an all-features test run) also runs
    // the exact solvers; the schemas describe the published libraries, which
    // do not, so those spellings -- and only those -- may run beyond them.
    let exact: BTreeSet<String> = ["exact", "milpexact", "hybrid", "hybridexact"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let beyond_schema = |schema: BTreeSet<String>, runs: BTreeSet<String>| {
        assert!(schema.is_subset(&runs), "{schema:?} not all run: {runs:?}");
        let extra: BTreeSet<String> = runs.difference(&schema).cloned().collect();
        assert!(
            extra.is_empty() || extra == exact,
            "runs beyond the schema: {extra:?}"
        );
    };
    beyond_schema(
        literals(&config(&d2)),
        runs(&|name| solves_2d(&request_2d(json!({ "strategy": name })))),
    );
    beyond_schema(
        literals(&config(&d3)),
        runs(&|name| solves_3d(&request_3d(json!({ "strategy": name }), "any"))),
    );
    // Every short name is one of the spellings.
    for name in Strategy::NAMES {
        assert!(
            Strategy::SPELLINGS.iter().any(|(s, _)| *s == name),
            "{name}"
        );
    }
}

#[test]
fn the_orientation_enum_is_what_a_box_accepts() {
    let d3 = schema("request-3d");
    let orientation = property(&d3, &d3["properties"]["geometries"]["items"], "orientation");
    let names: BTreeSet<String> = OrientationConstraint::NAMES.map(String::from).into();
    assert_eq!(literals(orientation), names);
    for name in OrientationConstraint::NAMES {
        assert!(solves_3d(&request_3d(json!({}), name)), "{name}");
    }
    let refused = solve_3d_internal(&request_3d(json!({}), "sideways").to_string());
    assert_eq!(refused.code.as_deref(), Some("unknown_option"));
    let expected: BTreeSet<String> = refused
        .details
        .and_then(|d| d.expected)
        .expect("expected names")
        .into_iter()
        .collect();
    assert_eq!(expected, names);
}

/// The 2D response schema has the keys a response carries, its placements'
/// and refusal details' keys, and the refusal codes the engine gives.
#[test]
fn the_response_schema_has_the_keys_a_response_carries() {
    let root = schema("response");
    let solved = solve_2d_internal(&request_2d(json!({})).to_string());
    assert!(solved.success);
    let solved = serde_json::to_value(&solved).expect("serialize");
    assert_eq!(object_keys(&solved), schema_keys(&root), "response keys");
    assert_eq!(
        object_keys(&solved["placements"][0]),
        schema_keys(resolve(&root, &root["properties"]["placements"]["items"])),
        "placement keys"
    );

    let every_detail = RefusalDetails {
        parameter: Some(String::new()),
        id: Some(String::new()),
        first: Some(0),
        index: Some(0),
        min: Some(0.0),
        max: Some(0.0),
        exclusive_min: Some(0.0),
        exclusive_max: Some(0.0),
        got: Some(u_nesting_core::api_types::RefusalValue::Number(0.0)),
        expected: Some(Vec::new()),
    };
    assert_eq!(
        object_keys(&serde_json::to_value(every_detail).expect("serialize")),
        schema_keys(&root["properties"]["details"]),
        "refusal detail keys"
    );

    let codes: BTreeSet<String> = [
        Error::OutOfRange {
            parameter: String::new(),
            id: None,
            min: None,
            max: None,
            exclusive_min: None,
            exclusive_max: None,
            got: 0.0,
            range: String::new(),
        },
        Error::empty_input(""),
        Error::invalid_option("", String::new()),
        Error::UnknownOption {
            parameter: String::new(),
            got: String::new(),
            expected: Vec::new(),
        },
        Error::malformed(None, String::new()),
        Error::Cancelled,
        Error::invalid_geometry(None, String::new()),
        Error::DuplicateId {
            id: String::new(),
            first: 0,
            second: 1,
        },
        Error::invalid_boundary(None, String::new()),
        Error::Internal(String::new()),
    ]
    .iter()
    .map(|e| SolveResponse::refused(e).code.expect("a code"))
    .collect();
    assert_eq!(
        literals(&root["properties"]["code"]),
        codes,
        "refusal codes"
    );
}
