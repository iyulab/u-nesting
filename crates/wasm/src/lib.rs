//! # U-Nesting WASM
//!
//! WebAssembly bindings for the U-Nesting spatial optimization engine.
//!
//! Every function takes and returns plain JavaScript objects of the shapes the
//! C ABI reads and writes as JSON (the types of `u_nesting_core::api_types`,
//! declared in the package's `.d.ts`). A request the engine refuses is thrown
//! as an `Error` whose `message` is the readable text and which carries `code`
//! -- a stable reason -- and the values behind it (`parameter`, `index`, `id`,
//! `min`, `got`, ...). A layout `check_2d` / `check_3d` finds infeasible is a
//! result, not a refusal.
//!
//! ## Functions
//!
//! - [`solve_2d`] — Solve a 2D nesting problem
//! - [`solve_3d`] — Solve a 3D bin packing problem
//! - [`optimize_cutting_path`] — Optimize cutting path for placed parts
//! - [`version`] — Get API version
//! - [`check_2d`] — Check a 2D layout against its problem
//! - [`check_3d`] — Check a 3D packing against its problem
//! - [`available_strategies`] — The strategies each dimension offers and the settings each reads
//! - [`codes`] — Every refusal and checker violation `code`

use serde::Serialize;
use u_nesting_core::api_types::*;
use u_nesting_core::geometry::Boundary;
use u_nesting_core::solver::{Config, Solver};
use u_nesting_core::Error;
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};
use wasm_bindgen::prelude::*;

/// Solves a 2D nesting problem.
///
/// # Errors
/// Throws an `Error` carrying `code` and the values behind it for a request
/// the engine refuses (`malformed_input`, `value_not_finite`,
/// `parameter_out_of_range`, `invalid_geometry`, ... -- see [`codes`]).
#[wasm_bindgen(unchecked_return_type = "SolveResponse")]
pub fn solve_2d(
    #[wasm_bindgen(unchecked_param_type = "Request2D")] request: JsValue,
) -> Result<JsValue, JsValue> {
    answer::<Request2D, _>(request, solve_2d_internal)
}

/// Solves a 3D bin packing problem.
///
/// # Errors
/// Throws a structured `Error` for a refused request, as [`solve_2d`] does.
#[wasm_bindgen(unchecked_return_type = "Pack3DResponse")]
pub fn solve_3d(
    #[wasm_bindgen(unchecked_param_type = "Request3D")] request: JsValue,
) -> Result<JsValue, JsValue> {
    answer::<Request3D, _>(request, solve_3d_internal)
}

/// Optimizes the cutting path for the parts a successful `solve_2d` placed:
/// `{ geometries, solve_result, cutting_config? }`.
///
/// # Errors
/// Throws a structured `Error` for a refused request, as [`solve_2d`] does.
#[wasm_bindgen(unchecked_return_type = "CuttingResponse")]
pub fn optimize_cutting_path(
    #[wasm_bindgen(unchecked_param_type = "CuttingRequest")] request: JsValue,
) -> Result<JsValue, JsValue> {
    answer::<CuttingRequest, _>(request, optimize_cutting_path_internal)
}

/// Returns the API version string.
#[wasm_bindgen]
pub fn version() -> String {
    API_VERSION.to_string()
}

/// Every refusal `code` a function of this package throws, with the fields
/// that can come with it, and every violation `code` `check_2d` / `check_3d`
/// report: `{ refusals: [{ code, fields }], violations: [{ code, fields, checkers }] }`.
///
/// It is read off the engine's own refusal and violation types, so a caller
/// that translates codes can check its list against the package's.
#[wasm_bindgen(unchecked_return_type = "CodeCatalog")]
pub fn codes() -> Result<JsValue, JsValue> {
    to_js(&CodeCatalog::of_engine())
}

/// The strategies each dimension offers in this build and the settings each
/// reads: `{ "2d": [{ name: "blf", reads: ["spacing", ...] }, ...], "3d": [...] }`.
///
/// It is the table a solve's refusal of a setting the strategy does not read
/// (`invalid_option`) comes from, so a form can show only the settings that
/// change the chosen strategy's run. The exact (MILP) strategies are absent:
/// this build has no MILP solver.
#[wasm_bindgen(unchecked_return_type = "StrategyCatalog")]
pub fn available_strategies() -> Result<JsValue, JsValue> {
    to_js(&StrategyCatalog::of_build())
}

/// Checks a 2D layout against the problem it answers: `{ problem, solve_result }`
/// -- the `solve_2d` request and a successful `solve_2d` response. The answer
/// is `{ feasible, violations: [{ code, ... }] }`: a layout that breaks its
/// problem is a result, not a refusal.
///
/// # Errors
/// Throws a structured `Error` for a problem `solve_2d` refuses, or a
/// `solve_result` that is itself a refusal.
#[wasm_bindgen(unchecked_return_type = "CheckResponse")]
pub fn check_2d(
    #[wasm_bindgen(unchecked_param_type = "Check2DRequest")] request: JsValue,
) -> Result<JsValue, JsValue> {
    answer::<Check2DRequest, _>(request, check_2d_internal)
}

/// Checks a 3D packing against the problem it answers: `{ problem, solve_result }`,
/// answered as [`check_2d`] answers -- every box inside the container and its
/// margin, no overlap or spacing breach, allowed orientations, one container,
/// copies, `max_mass`, support under `gravity` / `stability`, and `unplaced`.
///
/// # Errors
/// Throws a structured `Error`, as [`check_2d`] does.
#[wasm_bindgen(unchecked_return_type = "CheckResponse")]
pub fn check_3d(
    #[wasm_bindgen(unchecked_param_type = "Check3DRequest")] request: JsValue,
) -> Result<JsValue, JsValue> {
    answer::<Check3DRequest, _>(request, check_3d_internal)
}

// --- The boundary: reading a request, answering or throwing ---

/// A response that is either the engine's answer or its refusal of the
/// request -- the shape the C ABI writes. In JavaScript the refusal is thrown.
trait Answer: Serialize {
    fn refused(error: &Error) -> Self;
    /// The refusal's message, `code` and fields, when this is one.
    fn refusal(&self) -> Option<(&str, &str, Option<&RefusalDetails>)>;
}

macro_rules! answer_impl {
    ($($t:ty),*) => {$(
        impl Answer for $t {
            fn refused(error: &Error) -> Self {
                <$t>::refused(error)
            }
            fn refusal(&self) -> Option<(&str, &str, Option<&RefusalDetails>)> {
                if self.success {
                    return None;
                }
                Some((
                    self.error.as_deref().unwrap_or("refused"),
                    self.code.as_deref().unwrap_or("internal"),
                    self.details.as_ref(),
                ))
            }
        }
    )*};
}
answer_impl!(
    SolveResponse,
    Pack3DResponse,
    CuttingResponse,
    CheckResponse
);

/// Reads `request` as a `Q`, runs `run` on it and hands back its answer, or
/// throws the refusal -- of the request as read, or the engine's.
fn answer<Q: serde::de::DeserializeOwned, R: Answer>(
    request: JsValue,
    run: impl FnOnce(Q) -> R,
) -> Result<JsValue, JsValue> {
    let response = match read(&request).and_then(from_value::<Q>) {
        Ok(request) => run(request),
        Err(e) => R::refused(&e),
    };
    match response.refusal() {
        Some((message, code, details)) => Err(throw(message, code, details)),
        None => to_js(&response),
    }
}

/// `value` as `serde_json`, which the engine's request reading (and its path
/// in a refusal) starts from -- refusing first what that conversion would
/// hide: a string (a request passed as `JSON.stringify(...)`) and a number
/// that is not finite (JSON has none; it would arrive as `null` and be refused
/// as a value of the wrong type).
fn read(value: &JsValue) -> Result<serde_json::Value, Error> {
    if value.as_string().is_some() {
        return Err(Error::malformed(
            Some("request"),
            "request: expected an object, got a string -- pass the object itself, \
             not JSON.stringify(...)"
                .to_string(),
        ));
    }
    if let Some(found) = find_non_finite(value, "") {
        return Err(found);
    }
    serde_wasm_bindgen::from_value(value.clone())
        .map_err(|e| Error::malformed(None, format!("request: {e}")))
}

/// The first NaN or ±Infinity in `value`, as the refusal naming it: the path
/// of the value -- or of the list it is in, with `index` its position there.
fn find_non_finite(value: &JsValue, path: &str) -> Option<Error> {
    let refused = |parameter: &str, index: Option<usize>, got: f64| Error::NotFinite {
        parameter: if parameter.is_empty() {
            "request"
        } else {
            parameter
        }
        .to_string(),
        id: None,
        index,
        got,
    };
    if let Some(n) = value.as_f64() {
        return (!n.is_finite()).then(|| refused(path, None, n));
    }
    if !value.is_object() {
        return None;
    }
    if js_sys::Array::is_array(value) {
        let items: &js_sys::Array = value.unchecked_ref();
        for (i, item) in items.iter().enumerate() {
            match item.as_f64() {
                Some(n) if !n.is_finite() => return Some(refused(path, Some(i), n)),
                Some(_) => {}
                None => {
                    if let Some(found) = find_non_finite(&item, &format!("{path}[{i}]")) {
                        return Some(found);
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = value.unchecked_ref();
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = entry.unchecked_into();
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = if path.is_empty() {
            key
        } else {
            format!("{path}.{key}")
        };
        if let Some(found) = find_non_finite(&pair.get(1), &inner) {
            return Some(found);
        }
    }
    None
}

/// The refusal as a JavaScript `Error`: `message`, `code`, and the fields of
/// its reason as further properties.
fn throw(message: &str, code: &str, details: Option<&RefusalDetails>) -> JsValue {
    let error = js_sys::Error::new(message);
    let set = |key: &str, value: &JsValue| {
        // Setting a property on a fresh `Error` cannot fail.
        let _ = js_sys::Reflect::set(&error, &JsValue::from_str(key), value);
    };
    set("code", &JsValue::from_str(code));
    if let Some(fields) = details.and_then(|d| to_js(d).ok()) {
        js_sys::Object::assign(&error, &fields.into());
    }
    error.into()
}

/// `value` as a plain JavaScript object, `None` as `null` -- what `JSON.parse`
/// of the C ABI's answer gives.
fn to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| {
            throw(
                &format!("the answer could not be handed back: {e}"),
                "internal",
                None,
            )
        })
}

// --- Internal implementations ---

/// Assembles a success `SolveResponse` from a solved 2D result, deriving every
/// accounting/metric field through the shared `From` impl (single source of
/// truth) and then filling in the placements it leaves empty.
fn build_solve_response(result: u_nesting_core::SolveResult<f64>) -> SolveResponse {
    let placements = result
        .placements
        .iter()
        .cloned()
        .map(PlacementResponse::from)
        .collect();
    let mut resp: SolveResponse = result.into();
    resp.placements = placements;
    resp
}

fn solve_2d_internal(request: Request2D) -> SolveResponse {
    // Convert geometries
    let geometries = match Geometry2D::from_requests(request.geometries) {
        Ok(g) => g,
        Err(e) => return SolveResponse::refused(&e),
    };

    // Exactly one shape: width and height together, or a polygon.
    let boundary = match Boundary2D::from_request(request.boundary) {
        Ok(b) => b,
        Err(e) => return SolveResponse::refused(&e),
    };

    // Read the multi-sheet flag before `build_config` consumes the config.
    let multi_sheet = request
        .config
        .as_ref()
        .and_then(|c| c.multi_sheet)
        .unwrap_or(false);

    // Build config
    let config = match request
        .config
        .as_ref()
        .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config)
    {
        Ok(c) => c,
        Err(e) => return SolveResponse::refused(&e),
    };

    // Solve — `multi_sheet` distributes overflow across additional sheets.
    let nester = Nester2D::new(config);
    let solved = if multi_sheet {
        nester.solve_multi_strip(&geometries, &boundary)
    } else {
        nester.solve(&geometries, &boundary)
    };
    match solved {
        Ok(mut result) => {
            if multi_sheet {
                // `solve_multi_strip` emits global strip coordinates; localize to the
                // per-sheet frame so each placement's x is relative to its own sheet.
                let (b_min, b_max) = boundary.aabb();
                result.to_boundary_local(b_max[0] - b_min[0]);
            }
            build_solve_response(result)
        }
        Err(e) => SolveResponse::refused(&e),
    }
}

fn solve_3d_internal(request: Request3D) -> Pack3DResponse {
    // Convert geometries
    let geometries = match Geometry3D::from_requests(request.geometries) {
        Ok(g) => g,
        Err(e) => return Pack3DResponse::refused(&e),
    };

    let boundary = Boundary3D::from_request(request.boundary);

    // Build config
    let config = match request
        .config
        .as_ref()
        .map_or_else(|| Ok(Config::default()), ConfigRequest::to_config_3d)
    {
        Ok(c) => c,
        Err(e) => return Pack3DResponse::refused(&e),
    };

    // Solve
    let packer = Packer3D::new(config);
    match packer.solve(&geometries, &boundary) {
        Ok(result) => u_nesting_d3::build_pack3d_response(&result, &geometries),
        Err(e) => Pack3DResponse::refused(&e),
    }
}

fn optimize_cutting_path_internal(request: CuttingRequest) -> CuttingResponse {
    // Validate the solve result
    if !request.solve_result.success {
        return CuttingResponse::refused(&Error::invalid_option(
            "solve_result",
            "the solve result is a refusal (success: false); give a successful one to cut"
                .to_string(),
        ));
    }

    // Convert geometries
    let geometries = match Geometry2D::from_requests(request.geometries) {
        Ok(g) => g,
        Err(e) => return CuttingResponse::refused(&e),
    };

    // Reconstruct SolveResult from the response
    let mut solve_result = u_nesting_core::SolveResult::<f64>::new();
    for p in &request.solve_result.placements {
        solve_result.placements.push(u_nesting_core::Placement {
            geometry_id: p.geometry_id.clone(),
            instance: p.instance,
            position: vec![p.x, p.y],
            rotation: vec![p.rotation.to_radians()],
            boundary_index: p.sheet_index,
            mirrored: p.flipped,
            rotation_index: None,
        });
    }
    solve_result.boundaries_used = request.solve_result.sheets_used;
    solve_result.utilization = request.solve_result.utilization;

    // Build cutting config
    let cutting_config =
        match u_nesting_cutting::CuttingConfig::from_request(request.cutting_config.as_ref()) {
            Ok(c) => c,
            Err(e) => return CuttingResponse::refused(&e),
        };

    // Run cutting path optimization
    let result =
        match u_nesting_cutting::optimize_cutting_path(&solve_result, &geometries, &cutting_config)
        {
            Ok(r) => r,
            Err(e) => return CuttingResponse::refused(&e),
        };

    result.to_response()
}

fn check_2d_internal(request: Check2DRequest) -> CheckResponse {
    u_nesting_d2::check::check_2d(request)
}

fn check_3d_internal(request: Check3DRequest) -> CheckResponse {
    u_nesting_d3::check::check_3d(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn answer(response: impl serde::Serialize) -> Value {
        serde_json::to_value(response).expect("every answer is JSON")
    }

    /// `request` read as the binding reads a JavaScript object, then run.
    fn run<Q: serde::de::DeserializeOwned, R: Answer>(request: &str, f: fn(Q) -> R) -> R {
        let value: Value = serde_json::from_str(request).expect("a test request is JSON");
        from_value(value).map_or_else(|e| R::refused(&e), f)
    }

    fn solve_2d(request: &str) -> SolveResponse {
        run(request, solve_2d_internal)
    }

    fn solve_3d(request: &str) -> Pack3DResponse {
        run(request, solve_3d_internal)
    }

    fn optimize_cutting_path(request: &str) -> CuttingResponse {
        run(request, optimize_cutting_path_internal)
    }

    /// A solve answer as the JSON a request embeds.
    fn json(response: SolveResponse) -> String {
        serde_json::to_string(&response).expect("a solve answer serializes")
    }

    /// Every solve response says why the run stopped -- the reason the engine
    /// recorded, not one derived here -- and a refusal carries `null`.
    #[test]
    fn a_solve_says_why_it_stopped() {
        let plates = |quantity: usize, config: &str| {
            format!(
                r#"{{"geometries": [{{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]], "quantity": {quantity}}}],
                    "boundary": {{"width": 2000, "height": 2000}}, "config": {config}}}"#
            )
        };
        let r = answer(solve_2d(&plates(4, r#"{"strategy": "blf"}"#)));
        assert_eq!(r["stop_reason"], "completed", "{r}");
        let r = answer(solve_2d(&plates(
            4,
            r#"{"strategy": "blf", "multi_sheet": true}"#,
        )));
        assert_eq!(r["stop_reason"], "completed", "{r}");
        let r = answer(solve_2d(&plates(
            4,
            r#"{"strategy": "ga", "target_utilization": 0.0, "seed": 1}"#,
        )));
        assert_eq!(r["stop_reason"], "target_reached", "{r}");
        let r = answer(solve_2d(&plates(
            20_000,
            r#"{"strategy": "blf", "time_limit_ms": 1}"#,
        )));
        assert_eq!(r["stop_reason"], "time_limit", "{r}");
        assert_eq!(r["all_placed"], false, "{r}");

        let r = answer(solve_3d(
            r#"{"geometries": [{"id": "crate", "dimensions": [1, 1, 1], "quantity": 3}],
                "boundary": {"dimensions": [5, 5, 5]}, "config": {"strategy": "ep"}}"#,
        ));
        assert_eq!(r["stop_reason"], "completed", "{r}");

        let r = answer(solve_2d(&plates(0, "{}")));
        assert_eq!(r["success"], false);
        assert_eq!(r["stop_reason"], Value::Null, "{r}");
        let r = answer(solve_3d(
            r#"{"geometries": [], "boundary": {"dimensions": [0, 5, 5]}}"#,
        ));
        assert_eq!(r["success"], false);
        assert_eq!(r["stop_reason"], Value::Null, "{r}");
    }

    /// A count is read as a number: negative and too small are the same
    /// range refusal; a fraction is the wrong shape, named by its path.
    #[test]
    fn counts_are_read_as_given() {
        let rectangle = |quantity: &str| {
            format!(
                r#"{{"geometries": [{{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]], "quantity": {quantity}}}],
                    "boundary": {{"width": 50, "height": 50}}}}"#
            )
        };
        for (quantity, got) in [("-1", -1.0), ("0", 0.0)] {
            let r = answer(solve_2d(&rectangle(quantity)));
            assert_eq!(r["success"], false);
            assert_eq!(r["code"], "parameter_out_of_range", "{r}");
            assert_eq!(
                r["details"],
                json!({ "parameter": "geometries[0].quantity", "id": "plate", "min": 1.0, "got": got })
            );
        }
        let r = answer(solve_2d(&rectangle("2.5")));
        assert_eq!(r["code"], "malformed_input", "{r}");
        assert_eq!(
            r["details"],
            json!({ "parameter": "geometries[0].quantity" })
        );
        assert_eq!(answer(solve_2d(&rectangle("2")))["total_requested"], 2);

        let r = answer(solve_3d(
            r#"{"geometries": [{"id": "crate", "dimensions": [1, 1, 1]}],
                "boundary": {"dimensions": [5, 5, 5]}, "config": {"strategy": "ga", "population_size": -1}}"#,
        ));
        assert_eq!(r["code"], "parameter_out_of_range", "{r}");
        assert_eq!(
            r["details"],
            json!({ "parameter": "config.population_size", "min": 2.0, "got": -1.0 })
        );

        let solved = json(solve_2d(&rectangle("1")));
        let r = answer(optimize_cutting_path(&format!(
            r#"{{"geometries": [{{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]]}}],
                "solve_result": {solved}, "cutting_config": {{"pierce_candidates": -1}}}}"#
        )));
        assert_eq!(r["code"], "parameter_out_of_range", "{r}");
        assert_eq!(
            r["details"],
            json!({ "parameter": "cutting_config.pierce_candidates", "min": 1.0, "got": -1.0 })
        );
    }

    /// What the engine was given is what it judges: no geometries, a boundary
    /// side of 0 and a machine speed of 0 are refusals naming the field, and
    /// the time estimate is a number.
    #[test]
    fn requests_are_judged_as_given() {
        for r in [
            answer(solve_2d(
                r#"{"geometries": [], "boundary": {"width": 50, "height": 50}}"#,
            )),
            answer(solve_3d(
                r#"{"geometries": [], "boundary": {"dimensions": [5, 5, 5]}}"#,
            )),
        ] {
            assert_eq!(r["code"], "empty_input", "{r}");
            assert_eq!(r["details"], json!({ "parameter": "geometries" }));
        }

        let r = answer(solve_2d(
            r#"{"geometries": [{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]]}],
                "boundary": {"width": 0, "height": 50}}"#,
        ));
        assert_eq!(r["code"], "parameter_out_of_range", "{r}");
        assert_eq!(
            r["details"],
            json!({ "parameter": "boundary.width", "exclusive_min": 0.0, "got": 0.0 })
        );
        // A strategy the dimension does not offer is refused by the solve,
        // and named by its path like a setting refused on reading.
        let r = answer(solve_3d(
            r#"{"geometries": [{"id": "crate", "dimensions": [1, 1, 1]}],
                "boundary": {"dimensions": [5, 5, 5]}, "config": {"strategy": "nfp"}}"#,
        ));
        assert_eq!(r["code"], "invalid_option", "{r}");
        assert_eq!(r["details"], json!({ "parameter": "config.strategy" }));
        let r = answer(solve_3d(
            r#"{"geometries": [{"id": "crate", "dimensions": [1, 1, 1]}],
                "boundary": {"dimensions": [5, 5, 5], "max_mass": -1}}"#,
        ));
        assert_eq!(
            r["details"],
            json!({ "parameter": "boundary.max_mass", "exclusive_min": 0.0, "got": -1.0 })
        );

        let solved = json(solve_2d(
            r#"{"geometries": [{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]]}],
                "boundary": {"width": 50, "height": 50}}"#,
        ));
        let cut = |config: &str| {
            answer(optimize_cutting_path(&format!(
                r#"{{"geometries": [{{"id": "plate", "polygon": [[0,0],[10,0],[10,5],[0,5]]}}],
                    "solve_result": {solved}, "cutting_config": {{{config}}}}}"#
            )))
        };
        let r = cut(r#""cut_speed": 0"#);
        assert_eq!(r["code"], "parameter_out_of_range", "{r}");
        assert_eq!(
            r["details"],
            json!({ "parameter": "cutting_config.cut_speed", "exclusive_min": 0.0, "got": 0.0 })
        );
        assert!(
            cut("")["estimated_time_seconds"]
                .as_f64()
                .expect("a number")
                > 0.0
        );
    }
}

/// The layouts and packings the operations return and judge, held against
/// plain geometry: axis-aligned boxes are placed, rotated by quarter turns or
/// permuted by orientation, and tested for containment and overlap with
/// interval arithmetic that shares nothing with the engine. The coordinate
/// conventions are the documented ones, read off the responses: a 2D
/// placement rotates the polygon counter-clockwise about its own origin and
/// then moves it by `(x, y)`; a 3D orientation label's i-th letter is the
/// original axis that lies along world axis i.
#[cfg(test)]
mod oracle_tests {
    use super::*;
    use serde_json::{json, Value};

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                | 1)
        }
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn between(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    const TOL: f64 = 1e-6;

    /// A 2D piece: `(id, width, height, quantity)`.
    type Piece = (String, f64, f64, usize);

    fn pieces(seed: u64, count: usize) -> Vec<Piece> {
        let mut r = Rng::new(seed);
        (0..count)
            .map(|i| {
                let (w, h) = (r.between(2.0, 9.0), r.between(2.0, 9.0));
                (format!("p{i}"), w, h, 1 + (r.unit() * 3.0) as usize)
            })
            .collect()
    }

    fn request_2d(pieces: &[Piece], width: f64, height: f64, config: Value) -> Value {
        json!({
            "geometries": pieces.iter().map(|(id, w, h, q)| json!({
                "id": id,
                "polygon": [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
                "quantity": q,
                "rotations": [0.0, 90.0],
            })).collect::<Vec<_>>(),
            "boundary": { "width": width, "height": height },
            "config": config,
        })
    }

    /// The rectangle a 2D placement occupies: `[x0, y0, x1, y1]`.
    fn rect_2d(pieces: &[Piece], p: &PlacementResponse) -> [f64; 4] {
        let (_, w, h, _) = pieces.iter().find(|q| q.0 == p.geometry_id).expect("piece");
        // Counter-clockwise about the origin by a quarter turn maps the
        // rectangle [0, w] x [0, h] to [-h, 0] x [0, w].
        let quarter = ((p.rotation / 90.0).round() as i64).rem_euclid(4);
        assert!(
            (p.rotation - 90.0 * quarter as f64).abs() < 1e-9,
            "rotation {}",
            p.rotation
        );
        let (x0, y0, x1, y1) = match quarter {
            0 => (0.0, 0.0, *w, *h),
            1 => (-h, 0.0, 0.0, *w),
            2 => (-w, -h, 0.0, 0.0),
            _ => (0.0, -w, *h, 0.0),
        };
        [x0 + p.x, y0 + p.y, x1 + p.x, y1 + p.y]
    }

    /// The gap between two boxes (0 when they touch or overlap) and their
    /// overlap measure per axis, as `(gap, overlaps)`: boxes overlap when
    /// every axis overlaps by more than `TOL`.
    fn gap_and_overlap(a: &[f64], b: &[f64]) -> (f64, bool) {
        let n = a.len() / 2;
        let mut sq = 0.0;
        let mut overlaps = true;
        for k in 0..n {
            let d = (b[k] - a[k + n]).max(a[k] - b[k + n]);
            if d > 0.0 {
                sq += d * d;
            }
            if d > -TOL {
                overlaps = false;
            }
        }
        (sq.sqrt(), overlaps)
    }

    /// The strategies of this build that run on their own: the exact (MILP)
    /// ones, which a build with the `milp` feature lists, need a solver the
    /// test machine may not have and place nothing without one.
    fn strategies(dimension: &str) -> Vec<(String, Vec<String>)> {
        let c = serde_json::to_value(StrategyCatalog::of_build()).expect("catalog");
        c[dimension]
            .as_array()
            .expect("list")
            .iter()
            .map(|s| {
                (
                    s["name"].as_str().expect("name").to_string(),
                    s["reads"]
                        .as_array()
                        .expect("reads")
                        .iter()
                        .map(|r| r.as_str().expect("r").to_string())
                        .collect(),
                )
            })
            .filter(|(name, _)| !name.ends_with("exact"))
            .collect()
    }

    /// A small, quick configuration for a strategy: only the settings it reads.
    fn config_for(name: &str, reads: &[String], extra: Value) -> Value {
        let mut c = json!({ "strategy": name });
        let mut set = |key: &str, value: Value| {
            if reads.iter().any(|r| r == key) {
                c[key] = value;
            }
        };
        set("seed", json!(7));
        set("population_size", json!(10));
        set("max_generations", json!(8));
        set("max_iterations", json!(150));
        set("time_limit_ms", json!(2000));
        for (k, v) in extra.as_object().expect("object") {
            set(k, v.clone());
        }
        c
    }

    /// What a successful 2D answer must be, by definition.
    fn assert_layout(
        pieces: &[Piece],
        (width, height): (f64, f64),
        margin: f64,
        spacing: f64,
        multi_sheet: bool,
        r: &SolveResponse,
        what: &str,
    ) {
        assert!(r.success, "{what}: {:?}", r.error);
        let total: usize = pieces.iter().map(|p| p.3).sum();
        assert_eq!(r.total_requested, total, "{what}: total_requested");
        let mut seen = std::collections::HashSet::new();
        let mut area = 0.0;
        let boxes: Vec<[f64; 4]> = r.placements.iter().map(|p| rect_2d(pieces, p)).collect();
        for (p, b) in r.placements.iter().zip(&boxes) {
            assert!(
                seen.insert((p.geometry_id.clone(), p.instance)),
                "{what}: {} #{} twice",
                p.geometry_id,
                p.instance
            );
            let (_, w, h, q) = pieces.iter().find(|q| q.0 == p.geometry_id).expect("piece");
            assert!(
                p.instance < *q,
                "{what}: copy {} of {}",
                p.instance,
                p.geometry_id
            );
            area += w * h;
            assert!(
                b[0] >= margin - TOL
                    && b[1] >= margin - TOL
                    && b[2] <= width - margin + TOL
                    && b[3] <= height - margin + TOL,
                "{what}: {} #{} leaves the sheet: {b:?}",
                p.geometry_id,
                p.instance
            );
        }
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                if r.placements[i].sheet_index != r.placements[j].sheet_index {
                    continue;
                }
                let (gap, overlap) = gap_and_overlap(&boxes[i], &boxes[j]);
                assert!(
                    !overlap,
                    "{what}: placements {i} and {j} overlap: {:?} {:?}",
                    boxes[i], boxes[j]
                );
                assert!(
                    gap >= spacing - TOL,
                    "{what}: placements {i} and {j} are {gap} apart, spacing {spacing}"
                );
            }
        }
        let sheets = r
            .placements
            .iter()
            .map(|p| p.sheet_index + 1)
            .max()
            .unwrap_or(0);
        assert!(
            r.sheets_used >= sheets,
            "{what}: sheets_used {} below the sheet used {sheets}",
            r.sheets_used
        );
        // One sheet: the area over the sheet's. Several (`multi_sheet`): the
        // area over each sheet's height times the length of it the pieces
        // reach, the last sheet not charged its unused length.
        let material = if multi_sheet {
            (0..r.sheets_used)
                .map(|s| {
                    let reach = r
                        .placements
                        .iter()
                        .zip(&boxes)
                        .filter(|(p, _)| p.sheet_index == s)
                        .map(|(_, b)| b[2])
                        .fold(0.0, f64::max);
                    height * reach
                })
                .sum::<f64>()
        } else {
            width * height
        };
        let expected = if area > 0.0 { area / material } else { 0.0 };
        assert!(
            (r.utilization - expected).abs() <= 1e-6,
            "{what}: utilization {} vs {expected}",
            r.utilization
        );
        let placed: std::collections::HashMap<&str, usize> =
            r.placements.iter().fold(Default::default(), |mut m, p| {
                *m.entry(p.geometry_id.as_str()).or_default() += 1;
                m
            });
        for (id, _, _, q) in pieces {
            let missing = *q > placed.get(id.as_str()).copied().unwrap_or(0);
            assert_eq!(
                r.unplaced.contains(id),
                missing,
                "{what}: unplaced list for {id}"
            );
        }
        assert_eq!(
            r.all_placed,
            r.placements.len() == total,
            "{what}: all_placed"
        );
        assert_eq!(
            r.unplaced_count,
            total - r.placements.len(),
            "{what}: unplaced_count"
        );
    }

    #[test]
    fn every_2d_strategy_returns_a_layout_that_is_inside_and_apart() {
        for (name, reads) in strategies("2d") {
            for (seed, margin, spacing) in [(1_u64, 0.0, 0.0), (2, 1.5, 0.5)] {
                let ps = pieces(seed, 6);
                let config = config_for(
                    &name,
                    &reads,
                    json!({ "margin": margin, "spacing": spacing }),
                );
                let r = solve_2d_internal(
                    from_value(request_2d(&ps, 40.0, 30.0, config)).expect("request"),
                );
                assert_layout(
                    &ps,
                    (40.0, 30.0),
                    margin,
                    spacing,
                    false,
                    &r,
                    &format!("{name} seed {seed}"),
                );
                assert!(
                    !r.placements.is_empty(),
                    "{name} seed {seed}: nothing placed on a sheet with room"
                );
            }
        }
    }

    #[test]
    fn multi_sheet_layouts_are_valid_on_every_sheet() {
        for (name, reads) in strategies("2d") {
            if !reads.iter().any(|r| r == "multi_sheet") {
                continue;
            }
            let ps = pieces(5, 6);
            let config = config_for(&name, &reads, json!({ "multi_sheet": true }));
            let r = solve_2d_internal(
                from_value(request_2d(&ps, 14.0, 14.0, config)).expect("request"),
            );
            assert_layout(
                &ps,
                (14.0, 14.0),
                0.0,
                0.0,
                true,
                &r,
                &format!("{name} multi-sheet"),
            );
            assert!(
                r.sheets_used > 1,
                "{name}: {} sheets for parts needing more",
                r.sheets_used
            );
        }
    }

    /// A perfect tiling exists and is easy: 24 unit-by-two bars in a 6 x 8
    /// sheet. No layout can pass full utilization, and the engine's answer
    /// can never place more bars than fit.
    #[test]
    fn utilization_never_passes_one_and_a_tiling_is_not_overfilled() {
        let ps = vec![("bar".to_string(), 1.0, 2.0, 24)];
        for (name, reads) in strategies("2d") {
            let config = config_for(&name, &reads, json!({}));
            let r =
                solve_2d_internal(from_value(request_2d(&ps, 6.0, 8.0, config)).expect("request"));
            assert_layout(&ps, (6.0, 8.0), 0.0, 0.0, false, &r, &name);
            assert!(
                r.placements.len() <= 24 && r.utilization <= 1.0 + 1e-9,
                "{name}: {}",
                r.utilization
            );
        }
    }

    /// `check_2d` against an exact predicate on random layouts: pieces dropped
    /// anywhere, so some leave the sheet and some overlap.
    #[test]
    fn check_2d_agrees_with_interval_arithmetic_on_random_layouts() {
        let (mut infeasible, mut feasible) = (0, 0);
        for seed in 0..60 {
            let mut r = Rng::new(seed + 100);
            let ps = pieces(seed, 4);
            let problem = request_2d(&ps, 30.0, 24.0, json!({}));
            let mut layout = solve_2d_internal(from_value(problem.clone()).expect("request"));
            assert!(layout.success);
            // A third keep the solver's own layout, a third move one piece, a
            // third drop every piece anywhere.
            let moved = layout.placements.len();
            for (i, p) in layout.placements.iter_mut().enumerate() {
                if seed % 3 == 0
                    || (seed % 3 == 1 && i != (r.unit() * moved as f64) as usize % moved)
                {
                    continue;
                }
                p.x = r.between(-4.0, 30.0);
                p.y = r.between(-4.0, 24.0);
                p.rotation = if r.unit() < 0.5 { 0.0 } else { 90.0 };
            }
            let boxes: Vec<[f64; 4]> = layout.placements.iter().map(|p| rect_2d(&ps, p)).collect();
            let outside: Vec<usize> = (0..boxes.len())
                .filter(|&i| {
                    boxes[i][0] < -TOL
                        || boxes[i][1] < -TOL
                        || boxes[i][2] > 30.0 + TOL
                        || boxes[i][3] > 24.0 + TOL
                })
                .collect();
            let mut overlapping = Vec::new();
            for i in 0..boxes.len() {
                for j in i + 1..boxes.len() {
                    if gap_and_overlap(&boxes[i], &boxes[j]).1 {
                        overlapping.push((i, j));
                    }
                }
            }
            let checked = u_nesting_d2::check::check_2d(Check2DRequest {
                problem: from_value(problem).expect("problem"),
                solve_result: layout,
            });
            assert!(checked.success, "seed {seed}: {:?}", checked.error);
            let found_outside: Vec<usize> = checked
                .violations
                .iter()
                .filter_map(|v| match v {
                    LayoutViolationResponse::OutsideBoundary { placement, .. } => Some(*placement),
                    _ => None,
                })
                .collect();
            let found_overlap: Vec<(usize, usize)> = checked
                .violations
                .iter()
                .filter_map(|v| match v {
                    LayoutViolationResponse::Overlap { placement, other } => {
                        Some((*placement.min(other), *placement.max(other)))
                    }
                    _ => None,
                })
                .collect();
            let mut sorted_outside = found_outside.clone();
            sorted_outside.sort_unstable();
            let mut sorted_overlap = found_overlap.clone();
            sorted_overlap.sort_unstable();
            assert_eq!(sorted_outside, outside, "seed {seed}: outside");
            assert_eq!(sorted_overlap, overlapping, "seed {seed}: overlap");
            assert_eq!(
                checked.feasible,
                outside.is_empty() && overlapping.is_empty(),
                "seed {seed}: feasible"
            );
            if checked.feasible {
                feasible += 1;
            } else {
                infeasible += 1;
            }
        }
        assert!(
            feasible >= 3 && infeasible >= 3,
            "the layouts should be both: {feasible} / {infeasible}"
        );
    }

    // --- 3D ---

    /// A 3D box: `(id, [a, b, c], quantity)`.
    type Boxy = (String, [f64; 3], usize);

    fn boxes_3d(seed: u64, count: usize) -> Vec<Boxy> {
        let mut r = Rng::new(seed);
        (0..count)
            .map(|i| {
                (
                    format!("b{i}"),
                    [
                        r.between(2.0, 6.0),
                        r.between(2.0, 6.0),
                        r.between(2.0, 6.0),
                    ],
                    1 + (r.unit() * 3.0) as usize,
                )
            })
            .collect()
    }

    fn request_3d(boxes: &[Boxy], container: [f64; 3], config: Value) -> Value {
        json!({
            "geometries": boxes.iter().map(|(id, d, q)| json!({
                "id": id, "dimensions": d, "quantity": q,
            })).collect::<Vec<_>>(),
            "boundary": { "dimensions": container },
            "config": config,
        })
    }

    /// The box a 3D placement occupies: `[x0, y0, z0, x1, y1, z1]`.
    fn box_3d(boxes: &[Boxy], p: &Placement3DResponse) -> [f64; 6] {
        let (_, d, _) = boxes.iter().find(|b| b.0 == p.geometry_id).expect("box");
        let axis = |c: char| match c {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            other => panic!("orientation letter {other}"),
        };
        let labels: Vec<usize> = p.orientation.chars().map(axis).collect();
        let e = [d[labels[0]], d[labels[1]], d[labels[2]]];
        [p.x, p.y, p.z, p.x + e[0], p.y + e[1], p.z + e[2]]
    }

    fn assert_packing(
        boxes: &[Boxy],
        container: [f64; 3],
        margin: f64,
        r: &Pack3DResponse,
        what: &str,
    ) {
        assert!(r.success, "{what}: {:?}", r.error);
        let total: usize = boxes.iter().map(|b| b.2).sum();
        assert_eq!(r.total_requested, total, "{what}: total_requested");
        let placed: Vec<[f64; 6]> = r.placements.iter().map(|p| box_3d(boxes, p)).collect();
        let mut volume = 0.0;
        for (p, b) in r.placements.iter().zip(&placed) {
            let (_, d, q) = boxes.iter().find(|x| x.0 == p.geometry_id).expect("box");
            assert!(
                p.instance < *q,
                "{what}: copy {} of {}",
                p.instance,
                p.geometry_id
            );
            volume += d[0] * d[1] * d[2];
            for k in 0..3 {
                assert!(
                    b[k] >= margin - TOL && b[k + 3] <= container[k] - margin + TOL,
                    "{what}: {} #{} leaves the container on axis {k}: {b:?}",
                    p.geometry_id,
                    p.instance
                );
            }
        }
        for i in 0..placed.len() {
            for j in i + 1..placed.len() {
                if r.placements[i].bin_index == r.placements[j].bin_index {
                    assert!(
                        !gap_and_overlap(&placed[i], &placed[j]).1,
                        "{what}: boxes {i} and {j} overlap"
                    );
                }
            }
        }
        let bins = r
            .placements
            .iter()
            .map(|p| p.bin_index + 1)
            .max()
            .unwrap_or(0);
        assert!(
            r.bins_used >= bins,
            "{what}: bins_used {} below {bins}",
            r.bins_used
        );
        let used = r.bins_used.max(1) as f64;
        let capacity = used * container[0] * container[1] * container[2];
        assert!(
            (r.utilization - volume / capacity).abs() <= 1e-6,
            "{what}: utilization {} vs {}",
            r.utilization,
            volume / capacity
        );
    }

    #[test]
    fn every_3d_strategy_returns_a_packing_that_is_inside_and_apart() {
        for (name, reads) in strategies("3d") {
            for (seed, margin) in [(1_u64, 0.0), (2, 0.5)] {
                let bs = boxes_3d(seed, 5);
                let config = config_for(&name, &reads, json!({ "margin": margin }));
                let r = solve_3d_internal(
                    from_value(request_3d(&bs, [14.0, 12.0, 10.0], config)).expect("request"),
                );
                assert_packing(
                    &bs,
                    [14.0, 12.0, 10.0],
                    margin,
                    &r,
                    &format!("{name} seed {seed}"),
                );
                assert!(
                    !r.placements.is_empty(),
                    "{name} seed {seed}: nothing placed in a container with room"
                );
            }
        }
    }

    /// `check_3d` against interval arithmetic on random packings, orientations
    /// included.
    #[test]
    fn check_3d_agrees_with_interval_arithmetic_on_random_packings() {
        let labels = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"];
        let (mut infeasible, mut feasible) = (0, 0);
        for seed in 0..60 {
            let mut r = Rng::new(seed + 300);
            let bs = boxes_3d(seed, 3);
            let container = [14.0, 12.0, 10.0];
            let problem = request_3d(&bs, container, json!({ "strategy": "ep" }));
            let mut packing = solve_3d_internal(from_value(problem.clone()).expect("request"));
            assert!(packing.success);
            let moved = packing.placements.len();
            for (i, p) in packing.placements.iter_mut().enumerate() {
                if seed % 3 == 0
                    || (seed % 3 == 1 && i != (r.unit() * moved as f64) as usize % moved)
                {
                    continue;
                }
                p.x = r.between(0.0, 14.0);
                p.y = r.between(0.0, 12.0);
                p.z = r.between(0.0, 10.0);
                p.orientation = labels[(r.unit() * 6.0) as usize % 6].to_string();
            }
            let placed: Vec<[f64; 6]> = packing.placements.iter().map(|p| box_3d(&bs, p)).collect();
            let outside: Vec<usize> = (0..placed.len())
                .filter(|&i| {
                    (0..3).any(|k| placed[i][k] < -TOL || placed[i][k + 3] > container[k] + TOL)
                })
                .collect();
            let mut overlapping = Vec::new();
            for i in 0..placed.len() {
                for j in i + 1..placed.len() {
                    if gap_and_overlap(&placed[i], &placed[j]).1 {
                        overlapping.push((i, j));
                    }
                }
            }
            let checked = u_nesting_d3::check::check_3d(Check3DRequest {
                problem: from_value(problem).expect("problem"),
                solve_result: packing,
            });
            assert!(checked.success, "seed {seed}: {:?}", checked.error);
            let mut found_outside: Vec<usize> = checked
                .violations
                .iter()
                .filter_map(|v| match v {
                    LayoutViolationResponse::OutsideBoundary { placement, .. } => Some(*placement),
                    _ => None,
                })
                .collect();
            found_outside.sort_unstable();
            let mut found_overlap: Vec<(usize, usize)> = checked
                .violations
                .iter()
                .filter_map(|v| match v {
                    LayoutViolationResponse::Overlap { placement, other } => {
                        Some((*placement.min(other), *placement.max(other)))
                    }
                    _ => None,
                })
                .collect();
            found_overlap.sort_unstable();
            assert_eq!(found_outside, outside, "seed {seed}: outside");
            assert_eq!(found_overlap, overlapping, "seed {seed}: overlap");
            if checked.feasible {
                feasible += 1;
            } else {
                infeasible += 1;
            }
        }
        assert!(
            feasible >= 3 && infeasible >= 3,
            "the packings should be both: {feasible} / {infeasible}"
        );
    }

    // --- cutting path ---

    /// A cutting plan is read against the parts it cuts: one step per part,
    /// each for the part's perimeter, rapid moves between the pierce points,
    /// and the totals the steps add up to.
    #[test]
    fn a_cutting_plan_cuts_each_part_once_and_adds_up() {
        for seed in 0..6 {
            let ps = pieces(seed + 40, 4);
            let problem = request_2d(&ps, 40.0, 30.0, json!({ "strategy": "blf" }));
            let layout = solve_2d_internal(from_value(problem.clone()).expect("request"));
            assert!(layout.success && layout.all_placed, "seed {seed}");
            let (rapid_speed, cut_speed) = (50.0, 5.0);
            let r = optimize_cutting_path_internal(
                from_value(json!({
                    "geometries": problem["geometries"],
                    "solve_result": layout,
                    "cutting_config": { "kerf_width": 0.0, "rapid_speed": rapid_speed, "cut_speed": cut_speed },
                }))
                .expect("cutting request"),
            );
            assert!(r.success, "seed {seed}: {:?}", r.error);
            assert_eq!(
                r.sequence.len(),
                layout.placements.len(),
                "seed {seed}: one step per part"
            );
            let mut cut = 0.0;
            let mut rapid = 0.0;
            let mut previous: Option<[f64; 2]> = None;
            for step in &r.sequence {
                let (_, w, h, _) = ps.iter().find(|p| p.0 == step.geometry_id).expect("piece");
                assert!(
                    (step.cut_distance - 2.0 * (w + h)).abs() <= 1e-6,
                    "seed {seed}: perimeter of {}",
                    step.geometry_id
                );
                cut += step.cut_distance;
                match (step.rapid_from, previous) {
                    (None, None) => assert!(step.rapid_distance <= 1e-9, "seed {seed}: first move"),
                    (Some(from), Some(prev)) => {
                        assert!(
                            ((from[0] - prev[0]).hypot(from[1] - prev[1])) <= 1e-6,
                            "seed {seed}: a rapid starts where the last cut ended"
                        );
                        let d =
                            (step.pierce_point[0] - from[0]).hypot(step.pierce_point[1] - from[1]);
                        assert!(
                            (step.rapid_distance - d).abs() <= 1e-6,
                            "seed {seed}: rapid distance"
                        );
                    }
                    other => panic!(
                        "seed {seed}: rapid_from and the previous pierce disagree: {other:?}"
                    ),
                }
                rapid += step.rapid_distance;
                previous = Some(step.pierce_point);
            }
            let rel = |a: f64, b: f64| (a - b).abs() <= 1e-6 * b.abs().max(1.0);
            assert!(rel(r.total_cut_distance, cut), "seed {seed}: total cut");
            assert!(
                rel(r.total_rapid_distance, rapid),
                "seed {seed}: total rapid"
            );
            assert_eq!(r.total_pierces, r.sequence.len(), "seed {seed}: pierces");
            assert!(
                rel(
                    r.estimated_time_seconds,
                    rapid / rapid_speed + cut / cut_speed
                ),
                "seed {seed}: time"
            );
            assert!(
                rel(r.efficiency, cut / (cut + rapid)),
                "seed {seed}: efficiency"
            );
        }
    }

    /// The same parts on the same sheet in other units of length (micrometres
    /// to hundreds of kilometres) are the same problem: each strategy that
    /// runs without a clock or a seed gives a valid layout and places as many
    /// parts at the same utilization.
    #[test]
    fn the_unit_of_length_does_not_change_the_layout() {
        let ps = pieces(11, 6);
        let (width, height) = (40.0, 30.0);
        for name in ["blf", "nfp"] {
            let mut base: Option<(usize, f64)> = None;
            for scale in [1.0, 1e-6, 1e-4, 1e-2, 1e2, 1e4, 1e8] {
                let scaled: Vec<Piece> = ps
                    .iter()
                    .map(|(id, w, h, q)| (id.clone(), w * scale, h * scale, *q))
                    .collect();
                let mut r = solve_2d_internal(
                    from_value(request_2d(
                        &scaled,
                        width * scale,
                        height * scale,
                        json!({ "strategy": name }),
                    ))
                    .expect("request"),
                );
                for p in &mut r.placements {
                    p.x /= scale;
                    p.y /= scale;
                }
                assert_layout(
                    &ps,
                    (width, height),
                    0.0,
                    0.0,
                    false,
                    &r,
                    &format!("{name} scale {scale:e}"),
                );
                let found = (r.placements.len(), r.utilization);
                let reference = *base.get_or_insert(found);
                assert_eq!(found.0, reference.0, "{name} scale {scale:e}: parts placed");
                assert!(
                    (found.1 - reference.1).abs() <= 1e-9,
                    "{name} scale {scale:e}: utilization {} vs {}",
                    found.1,
                    reference.1
                );
            }
        }
    }

    /// A tight set (twenty-four bars that tile the sheet exactly) in units
    /// from micrometres to hundreds of kilometres: where the pieces touch,
    /// rounding decides whether a corner is a contact or an overlap, so how
    /// many bars fit may differ between units -- but whatever is placed never
    /// overlaps and never leaves the sheet.
    #[test]
    fn a_tight_set_never_overlaps_in_any_unit_of_length() {
        let ps = vec![("bar".to_string(), 1.0, 2.0, 24)];
        for name in ["blf", "nfp"] {
            for scale in [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8] {
                let scaled: Vec<Piece> = ps
                    .iter()
                    .map(|(id, w, h, q)| (id.clone(), w * scale, h * scale, *q))
                    .collect();
                let mut r = solve_2d_internal(
                    from_value(request_2d(
                        &scaled,
                        6.0 * scale,
                        8.0 * scale,
                        json!({ "strategy": name }),
                    ))
                    .expect("request"),
                );
                for p in &mut r.placements {
                    p.x /= scale;
                    p.y /= scale;
                }
                assert_layout(
                    &ps,
                    (6.0, 8.0),
                    0.0,
                    0.0,
                    false,
                    &r,
                    &format!("{name} scale {scale:e}"),
                );
                assert!(
                    r.placements.len() >= 12,
                    "{name} scale {scale:e}: only {} bars of 24",
                    r.placements.len()
                );
            }
        }
    }

    /// The 3D counterpart: the same boxes in a container in other units of
    /// length give a valid packing, as many boxes at the same utilization.
    #[test]
    fn the_unit_of_length_does_not_change_the_packing() {
        let bs = boxes_3d(21, 5);
        let container = [14.0, 12.0, 10.0];
        for (name, reads) in strategies("3d") {
            if reads.iter().any(|r| r == "seed") {
                continue; // the searches draw random numbers: not unit-exact
            }
            let placed: Vec<(usize, f64)> = [1e-6, 1e-3, 1.0, 1e3, 1e8]
                .iter()
                .map(|&scale| {
                    let scaled: Vec<Boxy> = bs
                        .iter()
                        .map(|(id, d, q)| {
                            (id.clone(), [d[0] * scale, d[1] * scale, d[2] * scale], *q)
                        })
                        .collect();
                    let c = [
                        container[0] * scale,
                        container[1] * scale,
                        container[2] * scale,
                    ];
                    let mut r = solve_3d_internal(
                        from_value(request_3d(&scaled, c, json!({ "strategy": name })))
                            .expect("request"),
                    );
                    for p in &mut r.placements {
                        p.x /= scale;
                        p.y /= scale;
                        p.z /= scale;
                    }
                    assert_packing(&bs, container, 0.0, &r, &format!("{name} scale {scale:e}"));
                    (r.placements.len(), r.utilization)
                })
                .collect();
            assert!(
                placed
                    .windows(2)
                    .all(|w| w[0].0 == w[1].0 && (w[0].1 - w[1].1).abs() <= 1e-9),
                "{name}: boxes placed and utilization at scales 1e-6 .. 1e8: {placed:?}"
            );
        }
    }

    // --- non-rectangular parts ---

    type Pt = (f64, f64);

    /// A part made of convex pieces (counter-clockwise), with the outline the
    /// request gives.
    struct Shape {
        id: &'static str,
        outline: Vec<Pt>,
        convex: Vec<Vec<Pt>>,
        quantity: usize,
    }

    fn shapes() -> Vec<Shape> {
        vec![
            Shape {
                id: "triangle",
                outline: vec![(0.0, 0.0), (9.0, 0.0), (0.0, 6.0)],
                convex: vec![vec![(0.0, 0.0), (9.0, 0.0), (0.0, 6.0)]],
                quantity: 3,
            },
            Shape {
                id: "trapezoid",
                outline: vec![(0.0, 0.0), (10.0, 0.0), (7.0, 5.0), (2.0, 5.0)],
                convex: vec![vec![(0.0, 0.0), (10.0, 0.0), (7.0, 5.0), (2.0, 5.0)]],
                quantity: 2,
            },
            Shape {
                id: "ell",
                outline: vec![
                    (0.0, 0.0),
                    (8.0, 0.0),
                    (8.0, 3.0),
                    (4.0, 3.0),
                    (4.0, 8.0),
                    (0.0, 8.0),
                ],
                convex: vec![
                    vec![(0.0, 0.0), (8.0, 0.0), (8.0, 3.0), (0.0, 3.0)],
                    vec![(0.0, 3.0), (4.0, 3.0), (4.0, 8.0), (0.0, 8.0)],
                ],
                quantity: 2,
            },
        ]
    }

    fn request_polygons(shapes: &[Shape], scale: f64, sheet: (f64, f64), config: Value) -> Value {
        json!({
            "geometries": shapes.iter().map(|s| json!({
                "id": s.id,
                "polygon": s.outline.iter().map(|p| [p.0 * scale, p.1 * scale]).collect::<Vec<_>>(),
                "quantity": s.quantity,
                "rotations": [0.0, 90.0, 180.0, 270.0],
            })).collect::<Vec<_>>(),
            "boundary": { "width": sheet.0 * scale, "height": sheet.1 * scale },
            "config": config,
        })
    }

    /// A point rotated counter-clockwise about the origin by a multiple of a
    /// quarter turn, then moved by `(x, y)`.
    fn place(p: Pt, rotation: f64, x: f64, y: f64) -> Pt {
        let (mut px, mut py) = p;
        for _ in 0..(((rotation / 90.0).round() as i64).rem_euclid(4)) {
            (px, py) = (-py, px);
        }
        (px + x, py + y)
    }

    fn polygon_area(p: &[Pt]) -> f64 {
        (0..p.len())
            .map(|i| p[i].0 * p[(i + 1) % p.len()].1 - p[(i + 1) % p.len()].0 * p[i].1)
            .sum::<f64>()
            / 2.0
    }

    /// The area two convex counter-clockwise polygons share (Sutherland and
    /// Hodgman: clip one by each edge of the other).
    fn shared_area(subject: &[Pt], clip: &[Pt]) -> f64 {
        let mut out: Vec<Pt> = subject.to_vec();
        for i in 0..clip.len() {
            let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
            let side = |p: Pt| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            let input = std::mem::take(&mut out);
            for j in 0..input.len() {
                let (p, q) = (input[j], input[(j + 1) % input.len()]);
                let (sp, sq) = (side(p), side(q));
                if sp >= 0.0 {
                    out.push(p);
                }
                if (sp >= 0.0) != (sq >= 0.0) {
                    let t = sp / (sp - sq);
                    out.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
                }
            }
            if out.is_empty() {
                return 0.0;
            }
        }
        polygon_area(&out).abs()
    }

    /// A placed part: its convex pieces in the sheet, rotation and position
    /// applied, in the units of the problem (the response divided by `scale`).
    fn placed_pieces(shapes: &[Shape], p: &PlacementResponse) -> Vec<Vec<Pt>> {
        let shape = shapes
            .iter()
            .find(|s| s.id == p.geometry_id)
            .expect("shape");
        shape
            .convex
            .iter()
            .map(|piece| {
                piece
                    .iter()
                    .map(|&v| place(v, p.rotation, p.x, p.y))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn parts_that_are_not_rectangles_are_valid_in_every_unit_of_length() {
        let sheet = (40.0, 30.0);
        let ss = shapes();
        let total: usize = ss.iter().map(|s| s.quantity).sum();
        let tolerance = 1e-6;
        for name in ["blf", "nfp"] {
            let mut reference: Option<f64> = None;
            for scale in [1.0, 1e-6, 1e-3, 1e3, 1e8] {
                let mut r = solve_2d_internal(
                    from_value(request_polygons(
                        &ss,
                        scale,
                        sheet,
                        json!({ "strategy": name }),
                    ))
                    .expect("request"),
                );
                let what = format!("{name} scale {scale:e}");
                assert!(r.success, "{what}: {:?}", r.error);
                for p in &mut r.placements {
                    p.x /= scale;
                    p.y /= scale;
                }
                assert_eq!(
                    r.placements.len(),
                    total,
                    "{what}: every part fits this sheet"
                );
                let pieces: Vec<Vec<Vec<Pt>>> =
                    r.placements.iter().map(|p| placed_pieces(&ss, p)).collect();
                for (p, parts) in r.placements.iter().zip(&pieces) {
                    for v in parts.iter().flatten() {
                        assert!(
                            v.0 >= -tolerance
                                && v.1 >= -tolerance
                                && v.0 <= sheet.0 + tolerance
                                && v.1 <= sheet.1 + tolerance,
                            "{what}: {} #{} leaves the sheet at {v:?}",
                            p.geometry_id,
                            p.instance
                        );
                    }
                }
                for i in 0..pieces.len() {
                    for j in i + 1..pieces.len() {
                        let shared: f64 = pieces[i]
                            .iter()
                            .flat_map(|a| pieces[j].iter().map(move |b| shared_area(a, b)))
                            .sum();
                        assert!(
                            shared <= tolerance,
                            "{what}: parts {i} and {j} share an area of {shared}"
                        );
                    }
                }
                let utilization = r.utilization;
                let first = *reference.get_or_insert(utilization);
                assert!(
                    (utilization - first).abs() <= 1e-9,
                    "{what}: utilization {utilization} vs {first}"
                );
            }
        }
    }
}
