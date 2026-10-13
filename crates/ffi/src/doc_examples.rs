//! The JSON examples in the documentation held to what the engine takes:
//! every ```` ```json ```` block is either a whole request -- it has
//! `geometries` and `boundary` -- that solves, or is marked as a fragment.
//!
//! # The fragment marker
//!
//! A block that is not a whole request (one geometry, a boundary, a `config`
//! on its own, a response, a progress report, a settings file) is marked by
//! the line `<!-- fragment -->` as the last non-blank line before its opening
//! fence. GitHub and rustdoc render the comment as nothing. Every other block
//! must be JSON, a whole request, and solve: a key the engine does not read,
//! a value out of range, a strategy the dimension does not offer is a failure
//! here rather than a refusal a reader meets after copying the example.
//!
//! A request is solved as the C ABI solves it, with its `config.time_limit_ms`
//! held to at most a second so a search example does not make the test slow.
//! It is 3D when its `mode` is `"3d"` or its boundary has `dimensions`.

use crate::api::{solve_2d_internal, solve_3d_internal};
use serde_json::Value;

/// The line that marks the block after it as a fragment.
const FRAGMENT: &str = "<!-- fragment -->";

/// The longest a documented request may run here, in milliseconds.
const TIME_LIMIT_MS: u64 = 1000;

/// The pages whose examples a reader copies, relative to the repository root:
/// the READMEs and every page of the user guide.
fn pages() -> Vec<String> {
    let root = root();
    let mut pages: Vec<String> = [
        "README.md",
        "npm/README.md",
        "crates/python/README.md",
        "bindings/csharp/README.md",
        "docs/json-schema/README.md",
    ]
    .iter()
    .map(|p| p.to_string())
    .collect();
    let guide = root.join("docs/user-guide");
    let mut guide_pages: Vec<String> = std::fs::read_dir(&guide)
        .unwrap_or_else(|e| panic!("{}: {e}", guide.display()))
        .map(|entry| entry.expect("a directory entry").file_name())
        .filter_map(|name| name.to_str().map(str::to_string))
        .filter(|name| name.ends_with(".md"))
        .map(|name| format!("docs/user-guide/{name}"))
        .collect();
    guide_pages.sort();
    pages.extend(guide_pages);
    pages
}

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A ```` ```json ```` block: where it opens, its text, and whether it is
/// marked as a fragment.
struct Block {
    at: String,
    text: String,
    fragment: bool,
}

fn blocks(page: &str) -> Vec<Block> {
    let path = root().join(page);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{page}: {e}"));
    let lines: Vec<&str> = text.lines().collect();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim() != "```json" {
            i += 1;
            continue;
        }
        let fragment = lines[..i]
            .iter()
            .rev()
            .find(|l| !l.trim().is_empty())
            .is_some_and(|l| l.trim() == FRAGMENT);
        let start = i;
        let mut body = Vec::new();
        i += 1;
        while i < lines.len() && !lines[i].trim().starts_with("```") {
            body.push(lines[i]);
            i += 1;
        }
        assert!(
            i < lines.len(),
            "{page}:{}: the block is not closed",
            start + 1
        );
        blocks.push(Block {
            at: format!("{page}:{}", start + 1),
            text: body.join("\n"),
            fragment,
        });
        i += 1;
    }
    blocks
}

/// `request` with its time limit held to [`TIME_LIMIT_MS`].
fn held_to_the_limit(mut request: Value) -> Value {
    let object = request.as_object_mut().expect("a request is an object");
    let config = object
        .entry("config")
        .or_insert_with(|| Value::Object(Default::default()));
    if let Some(config) = config.as_object_mut() {
        let given = config.get("time_limit_ms").and_then(Value::as_u64);
        let held = match given {
            Some(ms) if ms > 0 => ms.min(TIME_LIMIT_MS),
            _ => TIME_LIMIT_MS,
        };
        config.insert("time_limit_ms".to_string(), Value::from(held));
    }
    request
}

#[test]
fn every_json_example_is_a_request_the_engine_takes_or_a_marked_fragment() {
    let mut requests = 0;
    for page in pages() {
        for block in blocks(&page) {
            if block.fragment {
                continue;
            }
            let at = &block.at;
            let value: Value = serde_json::from_str(&block.text).unwrap_or_else(|e| {
                panic!(
                    "{at}: not JSON ({e}); a block that is not a whole request is marked \
                     `{FRAGMENT}` on the line before it"
                )
            });
            assert!(
                value.get("geometries").is_some() && value.get("boundary").is_some(),
                "{at}: not a whole request (`geometries` and `boundary`); mark a fragment \
                 `{FRAGMENT}` on the line before it"
            );
            let three_d = value.get("mode").and_then(Value::as_str) == Some("3d")
                || value["boundary"].get("dimensions").is_some();
            let request = held_to_the_limit(value).to_string();
            let (success, code, error) = if three_d {
                let r = solve_3d_internal(&request);
                (r.success, r.code, r.error)
            } else {
                let r = solve_2d_internal(&request);
                (r.success, r.code, r.error)
            };
            assert!(
                success,
                "{at}: the engine refuses this example: {} -- {}",
                code.unwrap_or_default(),
                error.unwrap_or_default()
            );
            requests += 1;
        }
    }
    // The pages hold requests of both dimensions; finding none would mean the
    // scan, not the documentation, is broken.
    assert!(requests >= 6, "only {requests} requests found");
}

/// The marker marks a block: a stray one -- not followed by a JSON block --
/// would be a fragment marker that marks nothing.
#[test]
fn every_fragment_marker_marks_a_json_block() {
    for page in pages() {
        let text = std::fs::read_to_string(root().join(&page)).expect("readable");
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim() != FRAGMENT {
                continue;
            }
            let next = lines[i + 1..].iter().find(|l| !l.trim().is_empty());
            assert_eq!(
                next.map(|l| l.trim()),
                Some("```json"),
                "{page}:{}: `{FRAGMENT}` is not followed by a ```json block",
                i + 1
            );
        }
    }
}
