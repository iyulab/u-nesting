//! Engine card support (tests only).
//!
//! `engine-card.json` describes each exported WebAssembly function as an
//! operation whose input and output are JSON Schema 2020-12. Those schemas are
//! generated here from the types the binding reads and returns, into one file
//! (`schemas/wasm.schema.json`): an operation `f` has `$defs/f.input` -- an
//! object of its arguments by name -- and `$defs/f.output`.
//!
//! Three checks keep the card, the schema file and the binding one list:
//! the committed file is what the types generate now ([`Schemas::check`]),
//! the exports in the binding's source are the operations with the same
//! arguments in the same order ([`Schemas::check_exports`]), and the card
//! names every operation and points at its two schemas ([`Schemas::check_card`]).
//!
//! This file is the same in every crate that carries a card.

use std::cell::RefCell;
use std::path::PathBuf;

use schemars::generate::{SchemaGenerator, SchemaSettings};
use schemars::{JsonSchema, Schema};
use serde_json::{json, Map, Value};

/// Where the generated schema file sits, from the crate root; the card's
/// references are written against it.
pub const SCHEMA_FILE: &str = "schemas/wasm.schema.json";

/// Exports that are not operations: `codes` lists the refusal vocabulary,
/// which the card reaches through `errors.source`, and `version` names the
/// build.
const NOT_OPERATIONS: &[&str] = &["codes", "version"];

/// One argument of an exported function.
pub struct Arg {
    name: &'static str,
    schema: Schema,
    required: bool,
}

/// An argument the caller must pass.
pub fn arg(name: &'static str, schema: Schema) -> Arg {
    Arg {
        name,
        schema,
        required: true,
    }
}

/// An argument the caller may leave out (`undefined` or `null`).
#[allow(dead_code, reason = "this file is shared; not every binding has one")]
pub fn optional(name: &'static str, schema: Schema) -> Arg {
    Arg {
        name,
        schema,
        required: false,
    }
}

/// A string that is one of `names` -- for a field whose Rust type is a
/// `String` (or a type from another crate) but whose values are a closed
/// list; pass the table the binding parses or prints them from, so the
/// schema cannot drift from it.
#[allow(dead_code, reason = "this file is shared; not every binding has one")]
pub fn string_enum<'a>(names: impl IntoIterator<Item = &'a str>) -> Schema {
    let names: Vec<&str> = names.into_iter().collect();
    Schema::try_from(json!({ "type": "string", "enum": names })).expect("an object is a schema")
}

struct Operation {
    id: &'static str,
    args: Vec<Arg>,
    /// `None`: the card points at schemas written by hand elsewhere.
    output: Option<Schema>,
}

/// The operations of a binding and the generator their schemas share, so a
/// type used by several of them is defined once under `$defs`.
pub struct Schemas {
    generator: RefCell<SchemaGenerator>,
    operations: RefCell<Vec<Operation>>,
}

impl Schemas {
    pub fn new() -> Self {
        Self {
            generator: RefCell::new(SchemaSettings::draft2020_12().into_generator()),
            operations: RefCell::new(Vec::new()),
        }
    }

    /// The schema of `T`, as a reference into `$defs` when `T` is named.
    pub fn of<T: JsonSchema>(&self) -> Schema {
        self.generator.borrow_mut().subschema_for::<T>()
    }

    /// Declares one exported function: its arguments in call order and what
    /// it returns.
    pub fn op(&self, id: &'static str, args: Vec<Arg>, output: Schema) {
        self.operations.borrow_mut().push(Operation {
            id,
            args,
            output: Some(output),
        });
    }

    /// Declares an exported function whose schemas are written by hand, not
    /// generated here: its argument names are still held to the binding, and
    /// the card may point anywhere for its schemas.
    #[allow(dead_code, reason = "this file is shared; not every binding has one")]
    pub fn op_by_hand(&self, id: &'static str, args: &[&'static str]) {
        let args = args
            .iter()
            .map(|name| arg(name, Schema::default()))
            .collect();
        self.operations.borrow_mut().push(Operation {
            id,
            args,
            output: None,
        });
    }

    /// The schema file's content: every type the operations use and, per
    /// operation, `<id>.input` and `<id>.output`.
    pub fn document(&self, title: &str) -> Value {
        let mut defs: Map<String, Value> = self.generator.borrow_mut().take_definitions(true);
        for op in self.operations.borrow().iter() {
            let Some(output) = &op.output else {
                continue;
            };
            let names: Vec<&str> = op.args.iter().map(|a| a.name).collect();
            let mut properties = Map::new();
            for a in &op.args {
                properties.insert(a.name.to_string(), a.schema.clone().to_value());
            }
            let required: Vec<&str> = op
                .args
                .iter()
                .filter(|a| a.required)
                .map(|a| a.name)
                .collect();
            defs.insert(
                format!("{}.input", op.id),
                json!({
                    "description": format!(
                        "The arguments of {}({}), by name; the WebAssembly function takes them in that order.",
                        op.id,
                        names.join(", ")
                    ),
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false,
                }),
            );
            defs.insert(format!("{}.output", op.id), output.clone().to_value());
        }
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": title,
            "$defs": defs,
        })
    }

    /// The committed schema file is what the types generate now. With
    /// `UPDATE_SCHEMAS=1` set, writes it instead.
    pub fn check(&self, title: &str) {
        let path = root().join(SCHEMA_FILE);
        let generated = serde_json::to_string_pretty(&self.document(title))
            .expect("a JSON value serializes")
            + "\n";
        if std::env::var_os("UPDATE_SCHEMAS").is_some() {
            std::fs::create_dir_all(path.parent().expect("the file has a directory"))
                .expect("the schema directory can be created");
            std::fs::write(&path, &generated).expect("the schema file can be written");
            return;
        }
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        assert!(
            committed == generated,
            "{SCHEMA_FILE} is not what the binding's types generate; \
             regenerate it with UPDATE_SCHEMAS=1 cargo test --features wasm --lib engine_card"
        );
    }

    /// Every function the binding source exports is an operation here with
    /// the same arguments in the same order, and no operation is missing from
    /// the source.
    pub fn check_exports(&self, source: &str) {
        let exported: Vec<(String, Vec<String>)> = exports(source)
            .into_iter()
            .filter(|(name, _)| !NOT_OPERATIONS.contains(&name.as_str()))
            .collect();
        let declared: Vec<(String, Vec<String>)> = self
            .operations
            .borrow()
            .iter()
            .map(|op| {
                (
                    op.id.to_string(),
                    op.args.iter().map(|a| a.name.to_string()).collect(),
                )
            })
            .collect();
        let sorted = |mut v: Vec<(String, Vec<String>)>| {
            v.sort();
            v
        };
        assert_eq!(sorted(exported), sorted(declared));
    }

    /// `engine-card.json` lists exactly these operations, each generated one
    /// pointing at its two schemas in [`SCHEMA_FILE`].
    pub fn check_card(&self) {
        let card: Value = serde_json::from_str(
            &std::fs::read_to_string(root().join("engine-card.json"))
                .expect("engine-card.json is at the repository root"),
        )
        .expect("engine-card.json is JSON");
        let listed = card["operations"]
            .as_array()
            .expect("the card has operations");
        let mut ids: Vec<&str> = listed
            .iter()
            .map(|op| op["id"].as_str().expect("an operation has an id"))
            .collect();
        ids.sort_unstable();
        let mut expected: Vec<&str> = self.operations.borrow().iter().map(|op| op.id).collect();
        expected.sort_unstable();
        assert_eq!(ids, expected, "the card's operations are the binding's");
        let by_hand: Vec<&str> = self
            .operations
            .borrow()
            .iter()
            .filter(|op| op.output.is_none())
            .map(|op| op.id)
            .collect();
        for op in listed {
            let id = op["id"].as_str().expect("an operation has an id");
            if by_hand.contains(&id) {
                continue;
            }
            for side in ["input", "output"] {
                assert_eq!(
                    op[side]["schema"].as_str(),
                    Some(format!("{SCHEMA_FILE}#/$defs/{id}.{side}").as_str()),
                    "{id}.{side}.schema"
                );
            }
        }
    }
}

/// The repository root: the nearest directory up from this crate that holds
/// `engine-card.json` -- the crate itself, or the workspace above a member
/// crate -- and the crate when there is no card yet.
fn root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .find(|dir| dir.join("engine-card.json").is_file())
        .map_or(manifest.clone(), PathBuf::from)
}

/// Calls that read a clock or draw randomness no seed controls.
const CLOCK_AND_ENTROPY: &[&str] = &[
    "Instant::now",
    "SystemTime",
    "Date::now",
    "Math::random",
    "thread_rng",
    "rand::rng()",
    "rand::random",
    "from_entropy",
    "from_os_rng",
    "OsRng",
    "getrandom",
];

/// Every place in the repository's `src/` trees (each crate's, in a
/// workspace) that reads a clock or draws unseeded randomness, as `(path from
/// the repository root, call)`. A card calls an operation
/// deterministic, or seeded, only where none of these is on its path; a
/// crate's test pins this list so a new one cannot arrive unnoticed. It reads
/// this crate's source only: a clock or randomness reached through a
/// dependency is the card's to declare.
pub fn clock_and_entropy_uses() -> Vec<(String, &'static str)> {
    let mut found = Vec::new();
    let mut dirs = vec![root()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("a source directory is readable") {
            let path = entry.expect("a directory entry").path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            if path.is_dir() {
                if !name.is_some_and(|n| n == "target" || n.starts_with('.')) {
                    dirs.push(path);
                }
                continue;
            }
            let in_src = path
                .strip_prefix(root())
                .expect("under the repository root")
                .components()
                .any(|c| c.as_os_str() == "src");
            if !in_src
                || path.extension().is_none_or(|e| e != "rs")
                || path.ends_with("engine_card.rs")
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a source file is readable");
            // Comments and `#[cfg(test)] mod … { … }` blocks are not paths an
            // operation takes.
            let mut kept = Vec::new();
            let mut lines = text.lines().peekable();
            while let Some(line) = lines.next() {
                if line == "#[cfg(test)]" && lines.peek().is_some_and(|l| l.starts_with("mod ")) {
                    lines.by_ref().find(|l| *l == "}");
                    continue;
                }
                if !line.trim_start().starts_with("//") {
                    kept.push(line);
                }
            }
            let code = kept.join("\n");
            for needle in CLOCK_AND_ENTROPY {
                if code.contains(needle) {
                    let rel = path
                        .strip_prefix(root())
                        .expect("under the repository root");
                    found.push((rel.to_string_lossy().replace('\\', "/"), *needle));
                }
            }
        }
    }
    found.sort();
    found
}

/// The functions a `wasm-bindgen` source exports -- a `pub fn` at the start of
/// a line whose attributes include `#[wasm_bindgen` -- with their argument
/// names in order.
pub fn exports(source: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = source.lines().collect();
    let mut found = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("pub fn ") else {
            continue;
        };
        // The attributes and doc comments above it, up to the previous item.
        let exported = lines[..i]
            .iter()
            .rev()
            .take_while(|l| {
                let t = l.trim_start();
                l.starts_with("#[")
                    || t.starts_with("///")
                    || l.starts_with(' ')
                    || l.starts_with(")]")
            })
            .any(|l| l.starts_with("#[wasm_bindgen"));
        if !exported {
            continue;
        }
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        // `rest` is a slice of `source`; the parameter list follows the name.
        let start = rest.as_ptr() as usize - source.as_ptr() as usize + name.len();
        found.push((name, argument_names(&source[start..])));
    }
    found
}

/// The argument names of the parameter list that `text` starts with
/// (`(...)`), skipping attributes and string literals inside them.
fn argument_names(text: &str) -> Vec<String> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut attribute_depth: Option<usize> = None;
    let mut current = String::new();
    let mut names = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_string {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '#' if chars.peek() == Some(&'[') => attribute_depth = Some(depth),
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => {
                depth -= 1;
                if c == ']' && attribute_depth == Some(depth) {
                    attribute_depth = None;
                    continue;
                }
                if depth == 0 {
                    break;
                }
            }
            ',' if depth == 1 => {
                names.extend(argument_name(&current));
                current.clear();
                continue;
            }
            _ => {}
        }
        if depth >= 1 && attribute_depth.is_none() && !(depth == 1 && c == '(') {
            current.push(c);
        }
    }
    names.extend(argument_name(&current));
    names
}

fn argument_name(text: &str) -> Option<String> {
    let (name, _) = text.split_once(':')?;
    let name = name.trim().trim_start_matches("mut ").trim();
    (!name.is_empty()).then(|| name.to_string())
}
