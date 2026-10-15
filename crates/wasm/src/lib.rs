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
