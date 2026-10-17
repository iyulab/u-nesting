# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Depends on u-metaheur 0.14.

### Added

- `SaConfig::temperature_scale` and `AlnsConfig::temperature_scale` (core): what a
  configured temperature is measured in. `Calibrated` (the default) makes the start
  and the floor multiples of the objective's own scale -- the median amount by which
  a neighbour (a destroy/repair candidate, for ALNS) is worse than the start, measured
  on a few of them before the run, cut off at the time limit -- so one configuration
  serves any unit of fitness. `with_relative_temperature` sets it; `with_initial_temp`
  and `with_temperature` still set an absolute temperature. `SaResult` and
  `AlnsResult` carry `temperature_unit`, the factor that was applied.

### Fixed

- The ALNS "random" repair placed the removed pieces in area order. It shuffled them and
  then sorted them back inside the placement, so it was the greedy repair under another
  name and the search had one repair, not two. The placement now takes the order it is
  given.

### Added

- ALNS repair by longest side first (`RepairOperatorId::Longest`): the removed pieces are
  placed longest side first, a third order for the adaptive choice to draw from. With the
  fix above, over three ESICUP instances, three runs each at 10 s, the mean length of the
  ALNS search falls from 69.2 / 85.9 / 10691 to 69.3 / 84.2 / 10503 on shapes1 / marques /
  albano; over the twelve ESICUP instances (30 s, one run each) the mean gap is 18.3 % for
  ALNS (was 19.8) and 19.0 % for SA (was 19.3). (A best-fit repair that placed whichever piece fits earliest next was tried and
  was worse on all three, 71.7 / 86.3 / 10702: it searches every remaining piece at every
  step, so it also does a quadratic number of placements.)

### Changed

- **Breaking**: the default temperatures of the core `SaConfig` and `AlnsConfig` are
  calibrated multiples (start 1, floor 1e-3) instead of 1000 / 0.001 and 100 / 0.01
  in the unit of the fitness.
- 2D simulated annealing and ALNS take their temperatures from that scale (start
  where a typical worsening is accepted one time in three, cool to 0.03 of it)
  instead of constants tuned to this objective. Over three ESICUP instances, three
  runs each at 10 s, the mean length is the same as with the tuned constants within
  the spread of the runs (SA 65.9 / 86.3 / 10708 against 66.5 / 86.1 / 10694, ALNS
  69.2 / 85.9 / 10691 against 70.3 / 84.9 / 10484 on shapes1 / marques / albano).
  The 3D simulated annealing keeps its absolute temperature: one box more or less
  moves its objective by about 4 while the tuning that matters moves it by tenths,
  and a scale measured on neighbours of the start reads the 4 (utilisation over eight
  random sets of 25 boxes: 75.5 % with the constant, 70-73 % calibrated).

## [0.27.0] - 2026-10-17

`UNesting` and the PyPI package are 0.27.0. Depends on u-metaheur 0.13.

### Changed

- The bottom-left placement search tests the corners of the feasible region
  first — the vertices of the inner-fit and no-fit polygons and the points where
  their edges cross — and falls back to the sampling grid only when none of them
  qualifies. The least point of a polygon region in packing order is one of its
  corners, so the grid (tens of thousands of points, most inside a no-fit
  polygon) never improved the answer. A 24-piece layout whose candidates took
  680 ms each takes under 1 ms; with the 30 s budget now spent on the search
  instead of the seed layout, the mean gap to the best known length over the
  twelve ESICUP strip instances falls (one 30 s run each) from 31.3 to 20.4 %
  for GA, 26.9 to 17.7 % for BRKGA, 30.6 to 19.3 % for SA, 29.5 to 19.8 % for
  ALNS and 26.8 to 19.8 % for
  GDRR. Crossings are searched for only up to the first valid vertex along the
  length axis, among the edges inside the overlap of the two polygons' boxes:
  the same points are found, 25–30 % sooner on a 64-piece layout.
- The acceptance temperature of SA and ALNS is on the scale of the objective.
  Both started at 50–100 and cooled to 0.1–1 against objective differences of a
  few hundredths, so they accepted nearly every candidate and walked at random,
  keeping only the best layout seen. They now start where a layout a few percent
  worse is accepted about half the time and cool to where a hundredth rarely is.
  Over six ESICUP instances, three runs each at 30 s, the mean gap of that change
  alone falls from 24.3 to 20.1 % for ALNS and from 20.2 to 18.8 % for SA.
- The 3D simulated annealing has the same temperature fix: it started at 100 and
  cooled to 0.1 against objective differences of a few tenths, so it accepted
  nearly every candidate. On eight random sets of 25 boxes (20 s limit) the mean
  volume utilisation rises from 62.7 to 67.3 %.
- A non-convex no-fit polygon is built from convex parts (the triangulation's
  triangles merged across the diagonals that keep them convex) and the parts are
  united pairwise, not one after another: for a 32-vertex piece against another,
  900 triangle pairs become 324 convex pairs and the union takes 5 ms where it
  took 14 ms.

### Added

- `bench-runner run --clock` lets the time limit, not each strategy's own
  generation and iteration budget, end a run, so strategies compare at equal time.

## [0.26.0] - 2026-10-17

`UNesting` and the PyPI package are 0.26.0. Depends on u-metaheur 0.13.

### Changed

- The search strategies (GA, BRKGA, SA, GDRR, ALNS) keep the no-fit polygons
  between the pieces across the layouts they build, as the NFP-guided
  placement does, instead of computing every pair afresh for every layout: a
  no-fit polygon depends on the two shapes and their relative rotation, not on
  where either lies. The five copies of that loop are one function.
- The bottom-left placement search no longer sorts every candidate point (some
  tens of thousands, nearly all far past the answer): it takes them in packing
  order a block at a time, and ends its grid past the last no-fit polygon,
  where every point is clear of them all. The same point is found; one NFP-guided
  layout of 24 pieces takes 115 ms where it took 270 ms.

### Fixed

- **Breaking** The search strategies (GA, SA, BRKGA, GDRR, ALNS) could not tell
  a short layout from a long one. Their score was the share of the pieces
  placed plus the pieces' area over the area of the whole boundary, which is
  the same for every layout that places the same pieces, so a search over
  complete layouts had nothing to climb and ended on the layout it started
  from (GDRR and ALNS measured the origin of the highest piece on the y axis
  whichever axis the layout advances along). The score is now the share placed
  plus the density of the strip the layout uses: the pieces' area over the
  strip's width times the length they reach along the boundary's longer side.
  On the ESICUP `marques` instance in 20 s the strip length falls from 106 to
  87 (GA), 88 (SA), 85 (BRKGA), 87 (GDRR) and 86 (ALNS). `max_y` of
  `GdrrNestingSolution` and `AlnsNestingSolution` is replaced by `used_length`
  and `cross_extent`.

- A polygon written with its first vertex repeated at the end (GeoJSON, WKT,
  DXF, the ESICUP files) was refused as self-intersecting: the repeat became a
  zero-length edge touching the first edge. `Geometry2D::with_polygon`,
  `with_hole` and `Boundary2D::new`, `with_hole` now take the ring open or
  explicitly closed and store it open.
- `check_layout` / `check_2d` reported an `overlap` for two pieces that only
  touch, when the contact runs along slanted edges at coordinates no binary
  fraction represents (pieces an NFP placement leaves in exact contact): the
  overlay's snapping left a sliver above the area threshold. An intersection no
  thicker than the checker's length tolerance along the shorter piece's
  outline is contact now; a real overlap of any practical depth is still found.

## [0.25.0] - 2026-10-17

`UNesting` and the PyPI package are 0.25.0. Depends on u-metaheur 0.13.

### Changed

- **Breaking** The ALNS and GDRR operator sets name what they do. GDRR's three
  recreate operators (`recreate_best_fit`, `recreate_blf`, `recreate_nfp`)
  all ran the same largest-first bottom-left fill, so cycling through them
  added nothing: `GdrrProblem` has one `recreate`, and `RecreateType` and
  `RecreateResult::recreate_type` are removed. ALNS's `Related` removal was
  `Shaw` under another name, and its `BottomLeftFill` and `Regret` repairs were
  `Greedy` (`Regret` was not even offered): `DestroyOperatorId::Related`,
  `RepairOperatorId::BottomLeftFill` and `RepairOperatorId::Regret` are
  removed, so the adaptive weights choose among operators that differ.
- **Breaking** `PermutationChromosome` is removed: no strategy used it (2D and
  3D carry their own chromosomes), and it drew rotation genes from one count
  for every item, the fault fixed in 0.24.0 for the real ones.
- The user guide describes GDRR's acceptance (late acceptance hill climbing)
  and both searches' operators as they are.

## [0.24.0] - 2026-10-16

`UNesting` and the PyPI package are 0.24.0.

### Fixed

- 3D `brkga` tries every item. Its decoder ranked all 2N keys — the
  orientation keys too — and kept the first N indices, so about half the items
  were dropped before any was tried (2D had the same fault, fixed earlier).
  `RandomKeyChromosome::decode_as_permutation(n)` now ranks the first `n`
  keys, and both dimensions use it.
- 2D `ga` and `sa`, and 3D `ga` and `sa`, draw each instance's rotation or
  orientation from its own allowed ones, uniformly. They drew from the largest
  count among all geometries and folded the index back with a modulo, so a
  geometry with fewer options favoured its first angles; the 2D `ga` mutation
  drew from four options whatever the geometry allowed, so a fifth angle and
  beyond were unreachable; 3D `ga` drew from six. A rotation move now always
  turns an instance that has a choice (and reorders when none has).
- 2D and 3D `sa` read each instance's rotation by instance, not by its place
  in the sequence: a swap, relocation, inversion or chain move changed the
  rotations of the instances it moved.
- The exact MILP strategies (`milp` feature) place each piece only at the
  rotations its geometry allows; they tried 0°, 90°, 180° and 270° for every
  piece, so a piece fixed in orientation could come back turned.
- `hybrid_exact` returns the exact layout only when it places every piece; a
  partial one (the solver stopped on its limit) competes with the heuristic
  fallback, and the layout with more pieces placed, then the shorter strip,
  wins. It returned any exact layout that placed one piece. The exact attempt
  uses the candidate grid derived from the pieces, as `milp_exact` does (it
  used a 1-unit grid).
- The npm README's omitted `max_iterations` is 500 for `sa` and 1000 for
  `gdrr` and `alns` (it said 50).

### Changed

- **Breaking** `PermutationSolution` (`rotation_counts` per item, replacing
  `rotation_options`), `NestingChromosome` and `PackingChromosome` take each
  instance's number of options (`new(counts)`, `random…(counts, rng)`);
  `rotation_mutate` / `orientation_mutate` take no count and return whether
  they changed anything. `ExactConfig::rotation_steps`,
  `with_rotation_steps` and `rotation_angles` are removed — the exact
  strategies read each geometry's rotations.
  `RandomKeyChromosome::decode_as_permutation` takes the number of items.

## [0.23.0] - 2026-10-16

### Changed

- **Breaking (npm):** the WebAssembly functions take and return objects, and
  their types are declared in the package's `.d.ts` — `solve_2d(request:
  Request2D): SolveResponse`, `solve_3d(Request3D): Pack3DResponse`,
  `optimize_cutting_path(CuttingRequest): CuttingResponse`,
  `check_2d(Check2DRequest)` / `check_3d(Check3DRequest): CheckResponse`,
  `available_strategies(): StrategyCatalog`, `codes(): CodeCatalog`. They took
  and returned JSON strings, so the shapes lived only in the README and a
  program kept its own copy of them. A refused request is **thrown** as an
  `Error` with `code` and the fields (`parameter`, `index`, `id`, `min`, `got`,
  ...) as properties, instead of returned with `success: false`. Migration:
  `JSON.parse(solve_2d(JSON.stringify(req)))` becomes `solve_2d(req)` inside a
  `try`, reading `e.code` where `result.code` was read. A request passed as a
  string is refused (`malformed_input`, `parameter: "request"`), and a NaN or
  ±Infinity in it is refused as `value_not_finite` naming its path. The C ABI,
  .NET and Python are unchanged.
- A request read from a JavaScript or Python value names the field at fault
  without a line and column — those pointed into JSON text the caller never
  wrote (Rust: `api_types::from_value`).

## [0.22.0] - 2026-10-16

### Added

- `codes()`: every refusal `code` (with the `details` fields that can come with
  it) and every `check_2d` / `check_3d` violation `code`, as data --
  `{"refusals": [{"code", "fields"}], "violations": [{"code", "fields",
  "checkers"}]}` -- read off the engine's refusal and violation types, so a
  caller that translates codes can check its list against the package's rather
  than parse the README. Rust: `api_types::CodeCatalog::of_engine` and
  `LayoutViolationResponse::checkers`. The same catalog over the C ABI
  (`unesting_codes()`, allocated once — not freed), in .NET
  (`UNestingInfo.Codes` → `Codes(Refusals, Violations)`) and in Python
  (`codes()`).

### Fixed

- The npm README's refusal table lists `value_not_finite` (sent by the Python
  binding); a test keeps the tables and `codes()` equal.

## [0.21.1] - 2026-10-15

### Fixed

- A multi-sheet solve whose time limit runs out still opens the sheets the
  remaining pieces need. Since 0.21.0 the limit is shared among the sheets, and
  when one sheet's search was given all that was left -- its pieces' area fits
  one sheet, but their shapes need two -- the run stopped there and reported
  the overflow as unplaced. Once the limit is spent, the remaining sheets are
  filled by bottom-left fill (which ends when every piece has been tried), and
  `stop_reason` is still `time_limit`: the limit bounds how hard each sheet is
  optimised, not whether a needed sheet is opened.
- `elapsed_ms` (`computation_time_ms`) is the whole solve. A search reported
  only its own part, which starts after its greedy start on the same limit, so
  a solve cut short by a 2 s limit could report `time_limit` after 150 ms.

## [0.21.0] - 2026-10-15

### Added

- **Every solve says why it stopped**: `stop_reason` on every solve response
  (WASM `solve_2d`/`solve_3d`, C ABI, Python, .NET `NestingResult.StopReason` /
  `PackingResult.StopReason`) and on `SolveResult` — `completed` (the strategy ran
  to its own end), `optimal` (an exact method proved it), `target_reached` (every
  piece placed at `target_utilization` or more), `iteration_limit`,
  `no_improvement` (the search went its stall budget without improving),
  `time_limit` (the layout is the best found when `time_limit_ms` ran out) or
  `cancelled`; `null` on a refusal. Each strategy sets it where it stops, so a
  caller that offered a time limit, a target or a cancel button can tell which of
  them ended the run. Rust: `StopReason` (re-exported from `u-metaheur`, whose
  runners report the same vocabulary), and `stop_reason` on `GaResult`,
  `BrkgaResult`, `SaResult`, `GdrrResult` and `AlnsResult`.
- **`max_iterations`**: the iteration budget of `sa`, `gdrr` and `alns` (and the
  hybrid's ALNS fallback), at least 1, on every binding and in the strategy
  catalog — the counterpart of `max_generations`. Omitted, each uses the budget
  it is tuned for (2D SA 500, GDRR and ALNS 1000, 3D SA 10000). A seeded run
  that ends on it rather than on the clock is the same run every time.
- `GdrrRunner::with_cancellation` and `AlnsRunner::with_cancellation`, as
  `GaRunner` has.
- **Every strategy reports its progress while it runs** — every 2D and 3D search
  per evaluated batch or search step (and its greedy or layer start per piece),
  the extreme-point fill per box. `Nester2D::solve_multi_strip_with_progress`;
  `Nester2D::cancel_handle` and `Packer3D::cancel_handle` (the flag `cancel`
  sets, for a caller that cancels from a progress callback);
  `SaRunner::run_with_rng_and_progress`; `ProgressInfo::search_step`.

- **A 2D layout checker**: `u_nesting_d2::check::check_layout(geometries,
  boundary, &LayoutRules { spacing, margin, multi_sheet }, placements, unplaced)`
  reports every way a layout breaks its problem — a piece outside the boundary or
  its margin, two pieces on a sheet overlapping (decided by shared area, so
  touching is not overlapping) or closer than the spacing (a piece in another's
  hole is measured to that hole's edge), a rotation the geometry does not allow,
  a flip without `allow_flip`, a sheet after the first without `multi_sheet`, an
  unknown geometry, a copy index out of range or placed twice, and `unplaced`
  disagreeing with what is left short. On the wire: WASM `check_2d`, C ABI
  `unesting_check_2d` and .NET `Nester2D.Check(request, result)` (typed
  `LayoutViolation` records) take `{problem, solve_result}` — the `solve_2d`
  request unchanged and a successful response — and answer
  `{success, feasible, violations: [{code, ...}]}`; `check_2d` is one pipeline
  in `u_nesting_d2::check` that every binding calls.
- **A 3D packing checker**: `u_nesting_d3::check::check_packing(geometries,
  boundary, &PackingRules { spacing, margin }, placements, unplaced)` — a box
  outside the container or its margin, two boxes overlapping or closer than the
  spacing (Euclidean distance), an orientation the box's constraint does not
  allow, a bin after the first, a copy out of range or packed twice, the mass
  over `max_mass`, a box unsupported under `gravity` / `stability` (the
  packer's own support analysis), and `unplaced` disagreeing with what is left
  short. WASM `check_3d`, C ABI `unesting_check_3d` and .NET
  `Packer3D.Check(request, result)` take `{problem, solve_result}` and answer as
  the 2D check does; both checks share one response type (`CheckResponse`) and
  one violation vocabulary.
- `Packer3D::validate(geometries, boundary)` (public, as for 2D), refusing a
  strategy 3D does not offer before anything runs.
- `Nester2D::validate(geometries, boundary)`: the judgement every solve makes
  before it runs, public so a checker refuses the same problems.
  `Geometry2D::holes()` exposes a geometry's holes.

### Changed

- **Breaking:** the 2D search entry points (`run_ga_nesting`,
  `run_brkga_nesting`, `run_sa_nesting`, `run_gdrr_nesting`, `run_alns_nesting`)
  and the 3D ones (`run_ga_packing`, `run_brkga_packing`, `run_sa_packing`) take
  `progress: Option<&ProgressCallback>` last.
- The C ABI's progress calls return `UNESTING_ERR_CANCELLED` whenever the
  callback cancelled the run, with the layout placed by then in the result; a
  cancel during the run used to return `UNESTING_OK`.
- **Breaking:** `SolveResult::cancelled` and `SolveResult::target_reached` are
  replaced by `stop_reason` (`== StopReason::Cancelled`,
  `== StopReason::TargetReached`), and `target_reached` on `GaResult`,
  `BrkgaResult` and `SaResult` likewise; `SolveResult::completed_normally()` is
  removed. .NET: `NestingResult` and `PackingResult` take `StopReason` before
  `ElapsedMs`.
- A 2D search runs for the whole of `time_limit_ms`, and `time_limit_ms: 0` is
  unlimited for it as documented (each search still ends on its generation or
  iteration budget). A search used to get a quarter of the limit (at least
  5 s) — so a 60 s request searched for 15 s — and 10–15 s when the limit was 0.
- `time_limit_ms` bounds a multi-sheet solve as a whole: what is left of it is
  shared among the sheets the remaining pieces need. Every sheet used to get the
  full limit afresh, so ten sheets could run ten times over it.
- A strategy 2D nesting does not offer (`ep`, or an exact strategy without the
  `milp` feature) is refused before anything runs, by the same validation.

### Fixed

- 0.19.0 said `ga`, `brkga` and `sa` "report `target_reached`", but no binding
  carried it — the flag was set in Rust and dropped by every response. It is now
  `stop_reason: "target_reached"`.
- A cancelled bottom-left or NFP fill returned no placements at all; it now
  returns what it placed, as a timed-out one does.
- A 2D strategy that stopped early (time limit, cancellation) could report an
  `unplaced` list that left out pieces it never reached; `unplaced` now follows
  from the placements, as in 3D.
- `gdrr` and `alns` ignored cancellation and ran on to their iteration cap or the
  time limit.
- A multi-sheet solve reported `elapsed_ms: 0`.
- A target met in the last generation of `ga` or `brkga` was reported as not met.
- The exact strategies' proof of optimality, or the time limit that stopped them
  short of one, was only logged; it is now the stop reason.
- A cancel from the C ABI's progress callback — and so from .NET's
  `CancellationToken` and `ProgressChanged` — never reached a running solve: the
  callback was called only before the solve and after it, and only the fills and
  the 2D GA reported progress at all. The run now reports as it goes, and a 0 from
  the callback stops it at its next report.
- .NET `SolveAsync` without a progress reporter ran the solve without a callback,
  so its `CancellationToken` could not stop it.
- The npm README's response schema left out `unplaced_count`, `all_placed`,
  `used_bounding_box` and `used_utilization`.

## [0.20.0] - 2026-10-14

### Added

- **A strategy catalog: which strategies each dimension offers and which settings
  each reads**, on every binding — `available_strategies()` (WASM, Python),
  `unesting_strategies()` (C ABI) and `UNestingInfo.Strategies` (.NET, a typed
  `StrategyCatalog` with a `NestingSetting` enum and `Reads(strategy, setting)`):
  `{"2d": [{"name": "blf", "reads": ["spacing", ...]}, ...], "3d": [...]}`. It is
  made from the same table as the refusal of a setting the strategy does not read,
  so a form can show only the fields that change the run without restating it.
  Rust: `Strategy::reads(dimension)`, `Strategy::offered(dimension)`,
  `Dimension`, `SETTINGS` and `api_types::StrategyCatalog::of_build()`.

### Changed

- **Breaking (WASM, Python):** `available_strategies()` returns the catalog above
  instead of names only (WASM: `{"2d": [names], "3d": [names]}`; Python: one list
  of both dimensions' names). Migrate: `catalog["2d"].map(s => s.name)` for the old
  2D list.
- **Breaking (Rust API):** `Strategy::reads_setting` and `Strategy::readers_of` take
  the `Dimension`; `api_types::Mode` is replaced by `solver::Dimension` (same wire
  names, `"2d"` / `"3d"`).
- The refusal of a setting the strategy does not read covers every setting
  through one table; in 3D, `multi_sheet` is refused as "not read by any 3D
  strategy". A strategy the dimension does not offer is refused naming the
  offered strategies by their request names (`blf, nfp, ...`), not the Rust ones.

- Depends on `i_overlay` 9 (was 8.1). Polygon offsets (spacing, kerf) are
  computed by the new version; a layout can differ where a part now fits a
  little closer at exactly the requested spacing.
- **Breaking (Rust API):** settings that no solver or packer read are
  removed instead of accepted — `ExactConfig::gap_tolerance`, `verbosity` and
  `seed` (with `with_gap_tolerance`, `with_verbosity`, `with_seed`; the MILP
  solve uses none of them), `CuttingConfig::lead_in`, `bridge` and `thermal`
  (with `with_lead_in`, `with_bridge`, `with_thermal`; the path optimizer never
  applied them — `generate_lead_in_out`, `place_bridges` and the thermal
  functions remain as standalone tools), `Geometry3D::with_stackable`,
  `with_max_stack_weight` and `is_stackable` with the `Geometry3DExt::stackable`
  and `max_stack_load` defaults (no packer enforces stacking limits), and the
  C ABI constants `DEFAULT_CALLBACK_INTERVAL_MS`, `MIN_CALLBACK_INTERVAL_MS` and
  `MAX_CALLBACK_INTERVAL_MS` (progress callbacks are not throttled).
- **Breaking (.NET `UNesting`):** `NestingException` names its members the way the other
  .NET bindings' exceptions do — `Status` is the native status (was `ErrorCode`),
  `ErrorCode` the error body's `code` (was `Reason`), and `Parameter` / `Index`
  now say which value the refusal is about without reading `Details`. Migrate: `ex.ErrorCode` → `ex.Status`, `ex.Reason` → `ex.ErrorCode`.

### Fixed

- 3D `ep` (extreme point) now stops on `time_limit_ms` and on cancellation,
  keeping the boxes placed so far; it ran to the end regardless. An extreme-point
  packing that places nothing reports 0 bins used (it said 1).

## [0.19.0] - 2026-10-13

### Changed

- **Breaking:** a setting only some strategies read is refused as
  `invalid_option` (`parameter` = the setting) when the request's strategy does
  not read it, with the strategies that do in the message:
  `crossover_rate` / `mutation_rate` (`ga`), `population_size` /
  `max_generations` (`ga`, `brkga`), `seed` (`ga`, `brkga`, `sa`, `gdrr`,
  `alns`, `hybridexact`) and `target_utilization` (`ga`, `brkga`, `sa`). They
  used to be accepted and ignored. A request without a `strategy` runs `blf`,
  which reads none of them. `Strategy::reads_setting` / `readers_of` state the
  table; the Rust `Config` is unchanged.

- **Breaking:** `Config` has no `threads` field. It was never read: a thread
  count given there changed nothing. A build with the `parallel` feature runs
  its parallel work (GA and BRKGA population evaluation, no-fit polygon
  computation) on the Rayon pool the solve is called from, so
  `rayon::ThreadPool::install` around the solve, or `RAYON_NUM_THREADS`, sets
  how many threads it uses; without the feature a solve runs on the calling
  thread. The `u-nesting` crate gains the `parallel` feature, forwarding to
  the 2D and 3D crates.
- **Breaking:** the Python `solve_2d` and `solve_3d` raise no exception for a
  request they refuse: every refusal comes back in the result dict
  (`success: False`, `error`, `code`, `details`), as the solver's own refusals
  already did. A malformed or unknown key, a value of the wrong type and a
  setting out of range used to raise `ValueError`; they are now
  `malformed_input` and `parameter_out_of_range` with `details.parameter` the
  path of the value in the request (`config.stratgy`, `config.population_size`,
  `geometries[0].quantity`) -- the paths the C interface gives. A boundary with
  both a rectangle and a polygon is `invalid_boundary`, and a 3D `multi_sheet`
  is `invalid_option` (`config.multi_sheet`), in the result.
- **Breaking:** a Python number that is not finite (`nan`, `inf`) is refused as
  the new code `value_not_finite`, with `details.parameter` the path of the
  value -- or of the list it is in, with its position as `index`
  (`geometries[0].polygon[2]`, `index` 1) -- and the geometry's `id`. It used
  to be refused as JSON that did not parse, naming nothing. `Error` gains
  `NotFinite { parameter, id, index, got }`; the response schema lists the
  code. A value that is not JSON (a set, a key that is not a string) is
  `malformed_input` naming its path.
- **Breaking:** `u_nesting_cutting::gtsp::discretize_contours` returns
  `Result`: 0 pierce candidates is `parameter_out_of_range`
  (`pierce_candidates`, `min` 1). It used to place one candidate.
- **Breaking:** `u_nesting_d2::ga_nesting::run_ga_nesting` takes an optional
  progress callback as its last argument; `run_ga_nesting_with_progress` is
  removed.
- **Breaking:** `SaRunner::run_parallel` is removed. Nothing in the engine used
  it, and it read 0 restarts as 1 and drew from an unseeded generator.

### Fixed

- `target_utilization` stops the searches. It was validated and never read:
  no search stopped on it, though the README said one would. `ga`, `brkga` and
  `sa` (2D and 3D) now end once every part is placed at that utilization or
  more, and report `target_reached`.

- The ESICUP benchmark measures against the literature. Its reference table
  held strip lengths, some of them impossible for the instance (MARQUES at a
  density of 105 %, ALBANO above any published layout), and nothing read it
  into a run, so no ESICUP run ever reported a gap. The table now holds
  best-known densities with their source, and loading an instance derives its
  reference length from the instance's own pieces and strip
  (`Dataset::length_at_density`), so `gap_percent` is reported.

- `MilpExact` keeps `time_limit_ms` across the whole solve. The limit was
  looked at between piece pairs of the conflict pass only, not inside a pair,
  not in the coarser pass that takes over at a quarter of the limit, not while
  the model was built -- and the search was then given the full limit again:
  a 1 ms limit ran for about 20 s. Every phase now reads one deadline, and the
  search gets what is left of it. Loading the finished model into the solver
  is the one step outside it. When the limit runs out before the search, every
  piece is reported in `unplaced`; a cancelled solve sets `cancelled`.
- `HybridExact` runs its heuristic fallback in what the exact attempt left of
  the limit. It started a fresh budget, so a limit under ten seconds ran over
  by up to half again.
- The exact search no longer reports a layout it stopped at the time limit as
  optimal.

- The 2D genetic search runs the same with or without a progress callback. With
  a callback (`Solver::solve_with_progress`, the C interface's
  `*_with_progress` functions, the .NET `SolveWithProgress` and `SolveAsync`) it
  used to get the whole `time_limit_ms` for its greedy start and its search --
  and no time limit at all when `time_limit_ms` was 0 -- where a plain solve
  gives it a per-strip budget (a quarter of the limit, at least 5 s, or 15 s
  when the limit is 0). Bottom-left fill, NFP-guided placement and the genetic
  search now run one implementation for both paths, so the same request and
  seed give the same layout through either.
- The user guide and READMEs showed request keys the engine refuses: BRKGA
  `elite_ratio`, `mutant_ratio`, `bias`; SA `initial_temp`, `min_temp`,
  `iterations_per_temp`; GDRR `ruin_ratio`, `iterations`; ALNS `segment_size`,
  `reaction_factor` and the destroy ratios; `gravity` and `stability` in a 3D
  `config` (they belong to the `boundary`); a 2D boundary of several `sheets`;
  a `$schema` key in a request. They also showed Python functions that do not
  exist (`solve_2d_with_progress`, `solve_3d_with_progress`), Python result
  keys of the C response (`id`, `x`, `sheets_used`), ALNS and GDRR as 3D
  strategies and several 3D containers per solve. Each page now shows what the
  engine takes; the fixed settings of each search are named. Every JSON request
  example in the documentation is solved by a test, and a block that is not a
  whole request is marked `<!-- fragment -->`.
- The `u-nesting` crate's feature list names `cutting` and `parallel`, its
  example compiles, and the README's install line names the real features (`d2`,
  `d3`; there is no `3d`).

## [0.18.0] - 2026-10-13

Depends on u-geometry 0.5 and u-metaheur 0.8. The .NET client is `UNesting` 0.18.0.

### Changed

- **Breaking:** an empty `geometries` list is refused with the new code
  `empty_input` (`details: { parameter: "geometries" }`) by `solve_2d`,
  `solve_3d` and `solve` on every transport, and by `Solver::solve`,
  `Solver::solve_with_progress` and `Nester2D::solve_multi_strip`. It used to
  answer `success` with one sheet (or bin) used. `Error` gains
  `EmptyInput { parameter }` (`Error::empty_input`).
- **Breaking:** a boundary measure of 0 or below is `parameter_out_of_range`
  with `exclusive_min: 0` and `got`, named by its place in the request:
  `boundary.width`, `boundary.height`, `boundary.dimensions[0]` / `[1]` / `[2]`
  (width, depth, height) and `boundary.max_mass`. It used to be
  `invalid_boundary` with no bound and no value (and in 2D no `parameter`). A
  strip's width is checked too (it was not), and a side that is not a number
  (NaN) is refused (it passed). A boundary of a shape the engine cannot place
  into -- both or neither of size and polygon, a polygon of no area or crossing
  itself -- stays `invalid_boundary`. A box's own dimension of 0 or below and a
  negative mass are `parameter_out_of_range` the same way
  (`geometries[i].dimensions[j]`, `geometries[i].mass`, with the box's `id`;
  `dimensions[j]` / `mass` from `Geometry3D::validate`), no longer
  `invalid_geometry`.
- **Breaking:** the cutting optimizer refuses a `cut_speed` or `rapid_speed`
  that is not above 0 (`exclusive_min: 0`) and a `pierce_weight` or `tolerance`
  below 0 (`min: 0`), as `parameter_out_of_range`. With both speeds above 0,
  `estimated_time_seconds` is always a number -- the rapid distance at
  `rapid_speed` plus the cut distance at `cut_speed`, 0 when nothing is cut and
  on a refusal; a speed of 0 or below used to leave it `null`.
  `CuttingPathResult::estimated_time_seconds` and
  `CuttingResponse::estimated_time_seconds` are `f64`, and the npm type is
  `number`. New builders `CuttingConfig::with_speeds` and `with_tolerance`.
- **Breaking:** `Config::population_size` and `Config::max_generations` are
  `Option<usize>` / `Option<u32>`. `None` -- the default, and a request that
  leaves the field out -- runs each strategy with the budget it is tuned for:
  2D GA and BRKGA 30 individuals and 50 generations, 3D GA 100 and 500, 3D
  BRKGA 50 and 100 (the budgets an unset configuration ran with before).
  `Some(n)` runs `n`. The 2D nester used to cut a population above 30 to 30
  and a generation budget above 50 to 50, 2D BRKGA ran 50 generations and 3D
  BRKGA 50 individuals for 100 generations whatever was asked.
  `Config::with_population_size` and `Config::with_max_generations` set them;
  the GA strategies also use `Config::elite_count`, which they ignored (its
  default, 5, is the number they kept before).
- **Breaking:** the algorithm builders store what they are given and
  `validate` refuses a value out of range; they used to move it into range
  without a word: `SaConfig::with_initial_temp` (raised to 0.001),
  `with_final_temp` (0.0001), `with_iterations_per_temp` (1), `with_reheating`
  (a factor below 1.1 became 1.1, so 1.05 is now run as given),
  `ExactConfig::with_grid_step` (0.1 -- a step of 0.05 is now run as given),
  `with_max_items` and `with_rotation_steps` (1), `GdrrConfig::with_ruin_weights`
  (a negative weight, or NaN, became 0), `GaConfig::with_population_size` (2),
  `BrkgaConfig::with_population_size` (4 -- a population of 2 or 3 is now run
  as given), `AlnsConfig::with_segment_size` (1),
  `CuttingConfig::with_pierce_candidates` (0 became 1), and in 3D
  `PhysicsConfig::with_time_step`, `with_friction`, `with_restitution`,
  `with_shaking` and `StabilityConstraint::partial_base`. `validate` gains the
  checks those floors stood for: `SaConfig` `iterations_per_temp`,
  `ExactConfig` `max_items` and `rotation_steps`, `AlnsConfig` `segment_size`
  (each at least 1); `BrkgaConfig` refuses a population with no room for the
  elite and mutants its fractions ask for (`invalid_option`,
  `population_size`; it used to underflow). New `PhysicsConfig::validate` and
  `StabilityConstraint::validate`; `PhysicsSimulator::new`,
  `StabilityAnalyzer::new` and `Packer3D::validate_stability` return `Result`.
- **Breaking:** `nfp_cm_solver::run_nfp_cm_nesting` returns `Result` and
  refuses a `Config` or `ExactConfig` out of range before placing anything (it
  read the grid step as given, 0 included). The stub for builds without `milp`
  is removed; the module exists only with that feature.
- 3D: every entry point judges the request the same way before a strategy
  runs -- the configuration (`solve_with_progress` did not), the boundary, each
  box and their ids.
- `Boundary2D::from_request` and `Boundary3D::from_request` build a boundary
  from its request; every binding uses them. `error::check_range`,
  `check_at_least` and `check_above` are public.
- C#: `Config2D` / `Config3D` `PopulationSize`, `MaxGenerations` and
  `TimeLimitMs` are `int?`. `null` leaves the setting to the engine; 0 is sent
  as a value -- 0 generations, or no time limit. A 0 used to be dropped, so
  `TimeLimitMs = 0` ran with the 30-second default.
- Python: a boundary given as both shapes, or half of a rectangle, raises
  `ValueError` with the engine's message ("Invalid boundary: give either width
  and height, or polygon -- not both, not one half").
- The request JSON schemas no longer give `population_size` and
  `max_generations` defaults of 100 and 500 -- each strategy's own, listed in
  the descriptions, applies -- and the response schema lists `empty_input`.
  The user guide's defaults (population 50, generations 100) and the npm
  README's `crossover_rate` / `mutation_rate` defaults (0.8 / 0.1; they are
  0.85 / 0.05) are corrected, and examples with keys the engine refuses
  (`rotation_step`, `support_ratio`, `elite_ratio`) are fixed.
- **Breaking:** a `parameter_out_of_range` refusal tells a closed bound from an
  open one. `min` / `max` are now only bounds the value may equal;
  a bound it may not equal goes out as the new `exclusive_min` /
  `exclusive_max` (in `details` on every transport, and as fields of
  `Error::OutOfRange`, which gains `exclusive_min: Option<f64>` and
  `exclusive_max: Option<f64>`). A side carries at most one of the two, and the
  open-bound keys appear only when used, so a refusal with closed bounds keeps
  its shape. `SaConfig::validate` (`initial_temp`, `final_temp`) and
  `ExactConfig::validate` (`grid_step`) used to report "> 0" as
  `min: 2.2250738585072014e-308`; they now report `exclusive_min: 0`, and the
  message reads "a finite number > 0". The response JSON schema lists the two
  keys under `details`.
- **Breaking:** counts and other whole-number settings in a request are read
  as numbers, not into unsigned types: a geometry's `quantity`, and in `config`
  `time_limit_ms`, `population_size`, `max_generations` and `seed`, and in
  `cutting_config` `max_2opt_iterations`, `time_limit_ms` and
  `pierce_candidates`. A negative value used to be refused as
  `malformed_input` ("invalid value: integer `-1`, expected usize"); it is now
  `parameter_out_of_range`, with the same bound as a too-small positive value:
  `population_size` -1 and 1 are both `{ parameter: "config.population_size",
  min: 2, got }`, and a setting with no lower limit beyond its type (`seed`,
  `time_limit_ms`, `max_generations`, `max_2opt_iterations`) has `min: 0`. A
  value above what the setting holds (`max_generations` above 4294967295) also
  carries `max`. A fraction (`2.5`) is `malformed_input`, "must be a whole
  number"; a whole number written with a fraction part (`3.0`) is that number.
- **Breaking:** a quantity below 1 -- 0 or negative -- is
  `parameter_out_of_range` (`parameter: "geometries[i].quantity"`, `min: 1`,
  `got`, and the geometry's `id`), no longer `invalid_geometry`.
  `Geometry2D::validate` / `Geometry3D::validate` refuse a quantity of 0 the
  same way (`parameter: "quantity"`). `Error::OutOfRange` gains
  `id: Option<String>`, the geometry a setting belongs to, sent as `details.id`.
- **Breaking:** `Config::validate` refuses a `population_size` below 2 as
  `parameter_out_of_range`; a request's `population_size` of 0 or 1 used to be
  raised to 2 without a word.
- **Breaking:** a `malformed_input` refusal names the field at fault by its path
  in `details.parameter` (`geometries[2].quantity`, `config.seed`,
  `cutting_config.time_limit_ms`); text that is not JSON names none. The
  message is the path and the reason ("geometries[2].quantity: must be a whole
  number, got 2.5", then where in the text); only text that is not JSON still
  starts "Invalid JSON". `api_types::from_json` reads any request type this
  way, and every binding uses it.
- **Breaking:** the request DTOs hold those settings as `i128`
  (`Geometry2DRequest::quantity`, `ConfigRequest::seed`, ...) -- wide enough for
  every `u64` and its negatives -- and judge them when they are converted:
  `ConfigRequest::to_config`, `CuttingConfig::from_request`, and the new
  `Geometry2D::from_request` / `Geometry3D::from_request` (`from_requests` for
  a list), which every binding now builds its geometries with. The cutting
  optimizer's geometries are built the same way, so a cutting request's
  geometry `rotations` and `allow_flip` are read as for a solve.
- Python: a geometry the engine refuses before solving -- a quantity below 1,
  a 3D `orientation` nobody knows -- comes back in the result dict (`success:
  False`, `code`, `details`) like a refusal from the solver; an unknown
  orientation used to raise `ValueError`.
- The request JSON schemas list `seed` and give `population_size` (`minimum`
  2) and `max_generations` (`minimum` 0, `maximum` 4294967295) the ranges the
  engine accepts; they said 10 and 1.
- **Breaking (.NET):** `Geometry3D` and `Boundary3D` take their size as
  `Width`, `Depth` and `Height` (`Height` the vertical) and write it as
  `dimensions: [width, depth, height]`, the order the engine reads.
  `Geometry3D.Box(id, width, depth, height, quantity)` names its arguments that
  way, and `Boundary3D.Box(width, depth, height)` is new. The size used to be a
  `Dimensions` array documented as `[width, height, depth]`, and
  `Box(id, width, height, depth)` sent its `height` as the depth; a call that
  passed the height third must now pass it fourth. `Dimensions` is the read-only
  wire form.
- **Breaking (.NET):** `Geometry3D.Fragile`, `MaxStack` and `Orientations` are
  removed: the engine has no such settings, and setting any of them made it refuse
  the whole request (`malformed_input`, unknown field). `Geometry3D.Orientation`
  (`OrientationConstraint`: `Any`, `Upright`, `Fixed`) is the engine's
  `orientation`. `Geometry3D.Mass` and `Boundary3D.MaxMass` are `double?`; a
  `MaxMass` of 0 is sent (and refused) instead of read as "no limit".
- **Breaking (.NET):** strategies are enums with the engine's names,
  `Strategy2D` (`BottomLeftFill`, `NfpGuided`, `GeneticAlgorithm`, `Brkga`,
  `SimulatedAnnealing`, `Gdrr`, `Alns`) and `Strategy3D` (`BottomLeftFill`,
  `ExtremePoint`, `GeneticAlgorithm`, `Brkga`, `SimulatedAnnealing`), in place
  of strings. `NestingRequest.Mode` and `PackingRequest.Mode` are removed: each
  solver calls its own entry point.
- .NET: every request key the engine reads can be set. `Boundary2D` takes an
  arbitrary `Polygon` as well as `Width` / `Height` (now `double?`;
  `Boundary2D.Rectangle(width, height)` is new); `Config2D` gains
  `TargetUtilization`, `CrossoverRate` and `MutationRate`; `Config3D` gains
  `Spacing`, `Margin`, `TargetUtilization`, `CrossoverRate`, `MutationRate` and
  `Seed`. `Config2D.Spacing` / `Margin` are `double?` like the other settings: a
  `null` is not sent and the engine's default applies.
- **Breaking (.NET):** results are records read strictly: `NestingResult`,
  `Placement2D`, `PackingResult`, `Placement3D` and `ProgressInfo` take every
  value through their constructor, and a key missing from the engine's answer
  fails the call instead of reading as 0. They carry what the engine sends:
  `Version`, and each placement's `Instance` (which copy of its geometry).
  `Placement3D.Orientation` is an `AxisPermutation` (`Xyz` ... `Zyx`: which of the
  box's axes lies along the container's x, y and z). `Success` and `Error` are
  removed from `NestingResult` and `PackingResult`: a result is only returned for
  a solve that succeeded, and a refusal raises `NestingException`.
- .NET: a NaN or infinite number in a request is refused before the engine is
  called, as `NestingException` with `Reason` `value_not_finite` and `Details`
  `parameter` (the path, e.g. `geometries[1].polygon[2]` or `config.margin`),
  `index` (its position in the innermost array) and `id` (the geometry's). It
  used to surface as the serializer's `ArgumentException`.
- The 2D request JSON schema lists the `gdrr` and `alns` strategies.
- **Breaking:** a refusal on the wire names a setting by its path in the
  request, whichever step refused it -- reading the configuration or the solve
  itself: `config.spacing`, `config.strategy`, `config.seed`,
  `cutting_config.kerf_width`, `cutting_config.cut_speed`, like
  `boundary.width` and `geometries[2].quantity` already were. It used to be the
  bare field (`spacing`, `kerf_width`), while a mistyped value of the same
  field was named `config.seed`. This applies to `details.parameter` and to the
  `error` text of the C ABI, WASM and .NET responses and of the Python result
  dict. The Rust API keeps the field name (`Error::parameter()` is `spacing`
  for a `Config` built in code); `Error::located`, `ConfigRequest::locate` and
  `CuttingConfigRequest::locate` place it, and `ConfigRequest::KEYS` /
  `CuttingConfigRequest::KEYS` list the keys each object takes.
- **Breaking:** a 3D request's `config.multi_sheet` is refused as
  `invalid_option` (`config.multi_sheet`); 3D packs one container, and the
  setting was read and ignored. New `ConfigRequest::to_config_3d`.
- **Breaking:** `AlnsProblem::repair` takes the run's random number generator,
  like `destroy`. `run_ga_packing`, `run_brkga_packing` and `run_sa_packing`
  take the layout to start from (`None` for random genes), and the 3D problems
  gain `with_seed_layout`. `SolveResult::floored_at` returns the better of a
  search's result and baselines, keeping the search's provenance.
  `RandomKeyChromosome::discrete_key` and `permutation_keys` write a known
  solution as keys. `Strategy::SPELLINGS` lists every name `Strategy::parse`
  reads; `OrientationConstraint::NAMES` the orientation names.
- The 3D search strategies honour `time_limit_ms` (they ran to their
  generation or iteration budget whatever the limit).
- The response JSON schema lists `unplaced_count`, `all_placed`,
  `used_bounding_box` and `used_utilization`, which every 2D response carries.
  The request and response schemas are tested against the request and
  response types: their keys at every level, the strategy names each dimension
  runs, and the orientation names.

### Fixed

- A seed reproduces a run. The 3D GA, BRKGA and SA ignored `config.seed` --
  the same seed gave a different packing on almost every run -- and 2D GDRR and
  ALNS ignored it too; ALNS also drew its random repair order from fresh
  entropy. With a parallel build, a 2D search could return either of two
  layouts of equal fitness, whichever finished first; ties are now broken by
  the layout itself. The same request with the same seed returns the same
  layout when the run ends on its own budget; a run cut short by
  `time_limit_ms` or a cancellation is not reproduced (see `Config::seed`).
- A 3D search never returns less than a simple fill of the same request. GA,
  BRKGA and SA start from the layer fill and are floored at the better of it
  and the extreme-point fill -- more boxes, then a higher utilization, judged
  after gravity and stability drop what they do not allow -- as 2D searches
  are. On a small budget (four individuals, two generations) they used to
  leave boxes the fill packs out -- for two slabs that fit only lying flat, in
  34 (GA) and 40 (BRKGA) of 40 runs; under gravity and stability, for a stack
  of three, in up to 31 of 40.
- The layer placement the 3D searches decode with is the layer fill's own: a
  box turned wider or deeper than the container is skipped instead of placed
  out of it, and a box that fits nowhere no longer moves the row or layer the
  next box goes into. The 3D searches list a geometry as unplaced when any of
  its copies is left out, not only when all are.
- The 3D benchmark datasets give the engine their height as the vertical axis
  (`Item3D::engine_dimensions`, `Dataset3D::engine_bin`); they passed it as
  the depth, so an upright item kept its depth vertical.
- The algorithm guide names the exact solver `milpexact` (it showed `milp`,
  which is refused, and a `gap_tolerance` setting the request does not take)
  and lists GDRR, ALNS and the exact solver as 2D only.
- The 3D packing guide gives `blf` as the default strategy; it said `ep`.

- .NET: an exception thrown by a `ProgressChanged` handler, or while reading a
  progress report, stops the solve and is rethrown by `SolveWithProgress` /
  `SolveAsync`; it used to be swallowed and the solve ran on.
- The 3D response's `z` is documented as the height axis; it was called depth.
- The 3D packing guide describes the request the engine reads: `dimensions` as
  `[width, depth, height]`, `orientation` (`any`, `upright`, `fixed`) and the
  orientation codes with z vertical. It showed `fragile`, `max_stack` and
  `orientations`, which the engine refuses, and `[width, height, depth]`.
- **Breaking:** a `kerf_width` wider than the gap between two placed parts is
  refused with `parameter_out_of_range` (`parameter:
  "cutting_config.kerf_width"`, `Error` `parameter` `kerf_width`, `max` the
  narrowest gap) -- each cut takes a band the kerf's width beside its part, so the
  plan would cut one part into its neighbour. With the outward offsets of two
  neighbours overlapping, the cutting optimizer used to answer `success` with an
  empty `sequence`, nothing in `skipped_contours` and 0 s: the two parts each
  waited for the other, and every part near them waited too.
- A part placed inside another part's hole is cut before that hole; it used to
  be ordered only before the other part's exterior, so the hole could be cut
  first and free the slug the part sits on. Precedence is taken from the parts as
  designed, not from their kerf offsets.
- The cutting optimizer never returns a plan that leaves a contour out silently:
  every contour is in `sequence` or in `skipped_contours`, and a sequence that
  cannot hold them all is an `internal` refusal.

## [0.17.0] - 2026-10-07

Depends on u-metaheur 0.7. The .NET client is `UNesting` 0.17.0.

### Added

- Every refusal carries a stable `code` and its values in `details`
  (`parameter`, `id`, `first`, `index`, `min`, `max`, `got`, `expected`) beside the
  readable `error`, in the WebAssembly and C responses, the Python result dict and
  `NestingException.Reason` / `Details` in .NET. The codes: `parameter_out_of_range`,
  `unknown_option`, `invalid_option`, `invalid_geometry`, `duplicate_id`,
  `invalid_boundary`, `malformed_input`, `cancelled`, `internal`.
- `UNesting` carries the native library for `linux-arm64` (glibc 2.39 or later).

### Changed

- **Breaking:** `u_nesting_core::Error` carries what a caller needs as fields:
  `OutOfRange { parameter, min, max, got, range }`, `InvalidOption`, `UnknownOption`,
  `InvalidGeometry { id, message }`, `DuplicateId { id, first, second }`,
  `InvalidBoundary { parameter, message }`, `MalformedInput`, `Cancelled`, `Internal`,
  with `code()`. `ConfigError(String)` and the variants nothing produced (`NfpError`,
  `NoPlacement`, `Timeout`, `SerializationError`) are gone. The cutting optimizer
  returns this `Error` too.
- **Breaking:** `SolveResponse::error`, `Pack3DResponse::error` and
  `CuttingResponse::error` are replaced by `refused(&Error)`.
- `ConfigRequest::to_config` and `CuttingConfig::from_request` build the solver and
  cutting configurations once for every binding (each kept its own copy).

### Fixed

- A `target_utilization` outside [0, 1] is refused (`parameter_out_of_range`). Every
  binding clamped it into range while the engine's own check refused it.
- `UNesting`: a refusal raised `NestingException` with a fixed message per status
  ("Solver failed") and dropped the engine's text; it now carries the engine's
  message, `Reason` and `Details`. A cancelled solve no longer leaks its native result.
- `UNesting`: `Nester2D` and `Packer3D` serialized requests and read results and
  progress reports by reflection, which throws in trimmed and NativeAOT applications.
  They now use source-generated serialization; the package is marked
  `IsAotCompatible`.
- The `UNesting` README told users to supply the native library themselves; the
  package carries it.

## [0.16.0] - 2026-10-04

### Added

- `optimize_cutting_path` reports the contours it cannot cut in
  `skipped_contours` (`contour_id`, `geometry_id`, `instance`, `contour_type`,
  `reason`, `offset`). A contour smaller than the kerf (`reason:
  "kerf_collapsed"`) used to vanish from `sequence` while the response read
  `success: true`, so a part could come off the sheet without its hole. In
  Rust, `CuttingPathResult::skipped` with `SkippedContour`/`SkipReason`, and
  `CuttingPathResult::to_response` — the one conversion every binding uses.

### Changed

- **Breaking:** an empty `rotations` list is refused ("list at least one angle,
  or leave rotations out for a fixed orientation"); it was placed at 0° while
  the core documented an empty list as "any angle". A non-finite angle is
  refused too. `RotationConstraint::Free` is removed — no solver implemented
  it, and it placed parts at 0° only. The unused `Orientation3D` is removed.
- **Breaking (Rust API):** the algorithm configurations store what their
  builders are given and judge it in one place, `validate()`
  (`GaConfig`, `BrkgaConfig`, `SaConfig`, `AlnsConfig`, `GdrrConfig`,
  `ExactConfig`, and the solver `Config`). The builders used to clamp: a value
  out of range ran as a different one, a NaN slipped through some, and
  `GdrrConfig::with_ruin_ratio(NaN, …)` panicked. The runners
  (`GaRunner`, `BrkgaRunner`, `SaRunner`, `AlnsRunner`, `GdrrRunner`) and the
  `run_*_nesting` / `run_*_packing` functions now return `Result` and refuse
  an out-of-range configuration with `Error::ConfigError`; `Nester2D` and
  `Packer3D` check the solver `Config` before solving (`target_utilization`
  above 1 was clamped to 1).

## [0.15.0] - 2026-10-03

### Changed

- Depends on u-metaheur 0.6 and u-geometry 0.4.

### Fixed

- **Breaking:** a strategy the dimension does not provide is refused
  (`ConfigError`) instead of running another one with only a log warning:
  2D `ExtremePoint` (3D only), `MilpExact`/`HybridExact` without the `milp`
  feature (including the `exact` alias in WebAssembly), and 3D `NfpGuided`,
  `Gdrr`, `Alns`, `MilpExact`, `HybridExact`. With a progress callback, a
  strategy that has no progress reporting yet runs itself without progress;
  it used to be replaced by NFP-guided BLF (2D) or layer packing (3D).
- The 3D layer packer no longer places a box that is wider or deeper than the
  container: it only checked the height, so such a box was reported placed,
  hanging out of the container.
- 3D requests apply a geometry's `orientation` (`any`, `upright`, `fixed`);
  the field was accepted and ignored. An unknown name is refused.
- A 2D boundary gives either `width` and `height` or `polygon`; both, or half
  of the rectangle, are refused (part of the request used to be dropped).
- **Breaking:** `optimize_cutting_path` returns `Result`: a placement whose
  geometry is missing from the request is refused (it was left out of the
  cut plan with `success: true`), and so is a `CuttingConfig` with a negative
  or non-finite `kerf_width` (a negative one turned compensation off) or
  `pierce_candidates` 0 (it became 1). `CuttingConfig::validate` and
  `CutDirectionPreference::parse` are public; a misspelt
  `exterior_direction`/`interior_direction` is refused instead of read as
  `auto`.
- C FFI `unesting_solve`/`unesting_solve_with_callback` refuse a `mode` other
  than `"2d"` or `"3d"`; a misspelt one ran the 2D solver.

## [0.14.0] - 2026-09-30

### Changed

- Depends on u-metaheur 0.5 and u-geometry 0.3. Their modules re-exported
  here (`generic_ga`, `collision`, ...) are unchanged in Rust, but come from
  the new versions, so types from them do not mix with u-metaheur 0.4 or
  u-geometry 0.2 types a consumer depends on directly.

- The README says a browser without a bundler is not supported (the package
  loads its `.wasm` through an ES module import, which browsers refuse), instead
  of listing only the environments that work.

## [0.13.0] - 2026-09-30

C# `UNesting` NuGet **0.13.0** (lockstep) · PyPI `u-nesting` 0.13.0 · npm `@iyulab/u-nesting` 0.13.0.

### Changed

- Depends on `u-geometry` 0.2 (0.1 before). No change in behaviour: the new
  minor changes only the TypeScript declarations of its WebAssembly binding.
- The publishing workflow runs the README's JavaScript examples against the
  built package before it publishes, so an example that throws is caught
  before a reader copies it.

### Fixed

- **A geometry id given to two geometries is refused** by every 2D and 3D
  solve entry point, naming the id and both positions. Placements are named by
  geometry id and instance, so two such geometries came back as placements
  nobody could tell apart -- and the 2D placement check looked the shape up by
  id, so one of them was checked against the other's shape. Input that used to
  be accepted is now refused. The check is `geometry::ensure_unique_ids` in
  `u-nesting-core`.
- The README's Rust examples did not compile: they used `Config2D` and
  `Config3D` types, a `polygon!` macro and a `box_shape` constructor that the
  crate does not have, and `solve` without importing the `Solver` trait. They
  now use `Config`, `Geometry2D::with_polygon`, `Geometry3D::new` and
  `Boundary3D` for gravity and stability.
  The README's Rust examples are now compiled and run with the doc-tests,
  so an example that stops matching the API fails CI.

## [0.12.0] - 2026-09-16

### Fixed

- **`Strategy::Brkga` never considered about half the pieces.** Its chromosome
  holds one key block for placement order and one or two more for rotation and
  mirroring, and the order was read by ranking *every* key it held and taking
  the first N of that ranking. With uniform keys, roughly half those N are
  indices into the rotation or mirror block; they are not instances, so they
  were skipped, and the instances they displaced were never attempted at all --
  on every chromosome, in every generation. The order is now read from the
  first N keys, as the encoding always said it was.

  The effect was hidden because a search that fails to beat the greedy pass
  falls back to it. On an irregular instance with a 160-second budget, this
  strategy used to stop early and return the greedy layout of 18 pieces; it now
  places 19.

### Added

- **`Strategy::Brkga` starts from the greedy layout.** The greedy pass that
  runs before a search already produces a layout, and the other search
  strategies are seeded with it; this one computed it only to floor its result
  against it, then began from random keys. It is now encoded as a chromosome
  and placed in the first generation, the same warm start the others get.
  `BrkgaProblem` gained an `initial_population` method for this, defaulting to
  the random population BRKGA assumes, so nothing else changes.

- **`Strategy::MilpExact` placed nothing for more than about two pieces.** Four
  50x50 rectangles on a 400x100 sheet -- which holds sixteen -- came back as an
  empty layout after more than three minutes. Candidate positions were
  enumerated on a grid, which spends them evenly and mostly on positions no
  good layout ever uses; every pair of candidates from two pieces becomes a
  constraint, so the model grew past what the search could finish.

  Positions now come from stacking the pieces: `0`, then every piece extent,
  then every sum of two, and so on, bounded by how many pieces there are and by
  the room they have, plus the far edge of the sheet. These are the normal
  patterns of Herz (1972) and Christofides & Whitlock (1977); for axis-aligned
  boxes an optimal layout is always among them, and a piece that is pushed
  against neither the sheet nor another piece can always be pushed without
  making the layout worse. Shapes whose extents interlock rather than stack can
  still reach more positions than are worth enumerating, and those fall back to
  the grid.

  The four rectangles above are now all placed, in about seven seconds. Eight
  of them are placed too.

- **`Strategy::MilpExact` ignored `spacing`, and could place pieces on top of
  each other.** Two 20x20 pieces asked to stay 10 apart came back touching.
  The no-fit polygons that decide which pairs of positions conflict were being
  grown by moving each vertex as `v - spacing.copysign(v)` -- towards the
  coordinate origin, which *shrinks* the polygon rather than growing it, and
  does so relative to a point that has nothing to do with the polygon. A
  shrunken no-fit polygon reports no conflict for positions that really do
  clash, so the model permitted them. The crate's own polygon offset is now
  used, applied once per cached no-fit polygon. The coarser fallback used when
  conflict computation runs long kept no spacing at all; it now does.

- **`Strategy::MilpExact` returned an empty layout whenever the pieces did not
  all fit.** The model required every piece to be placed and minimised the
  strip length, which is a strip-packing formulation applied to a bounded
  sheet: three 50x50 pieces offered a 60x60 sheet were infeasible rather than
  one placed and two reported. It now places what fits and reports the rest in
  `unplaced`, like every other strategy, preferring the shortest strip among
  layouts that place the same pieces. `utilization` counts the pieces that were
  placed; it used to sum every piece handed in, reporting the number the layout
  would have had if all of them had fitted.

- **`Strategy::MilpExact` ran for minutes on a request for a second, and placed
  nothing past a single piece.** Two things, in the strategy's setup rather
  than in the model:

  Its time limit was raised to a minute whatever the caller asked for, so a
  1-second request on four rectangles took over three minutes. The caller's
  limit is now the limit, and 0 means unlimited as it does everywhere else in
  this crate. The same request now returns in about two seconds.

  Its candidate positions came from a fixed one-unit grid, so each piece
  carried about a thousand of them and every pair of candidates from two
  pieces becomes a constraint -- a million for two pieces, which the search
  never finished. The grid now comes from the pieces: a quarter of the
  smallest side any of them has, never finer than a unit and never fine
  enough for the sheet alone to carry more than a few hundred steps. Two
  50x50 pieces on a 200x100 sheet now come back placed; they used to come
  back as an empty layout however long the search was given.

  This strategy is still not usable for more than about two pieces: four
  rectangles still return an empty layout. The candidate model is what needs
  replacing, and the README's placement guarantees continue to exclude it.

- **A sheet with a hole came back a third empty on six of the seven
  strategies.** Bottom-Left-Fill reads the sheet as an axis-aligned box and
  laid its rows straight across a hole; the placement filter then dropped every
  piece that landed on one, so the position was lost rather than skipped. The
  search strategies seed from that layout, so they inherited it. On a 1000x1000
  sheet with a 600x600 hole and 100x100 pieces, the ring holds 64 and the
  strategies returned **44** -- `nfp` alone returned 64. All seven now return
  64.

  Bottom-Left-Fill now slides a candidate past the holes it runs into, taking
  each hole as its own box grown by `margin`, which is the same way it already
  reads the sheet. For a hole that is not a rectangle this blocks slightly more
  than the hole occupies; the strategies that follow the hole's actual outline
  (`nfp` and the searches, since 0.11.0) are unaffected and remain the ones to
  use when that matters.

  Layout coordinates on sheets with holes differ from 0.11.0. Sheets without
  holes are unchanged.

## [0.11.0] - 2026-09-16

A minor release because public functions change signature or are removed (see
**Changed**). Most of it is fixes to what the solvers promise: the distances
`spacing` and `margin` describe, the shape of the boundary, `time_limit_ms`, and
the length of the layouts they return. Layout coordinates from the NFP-based and
search strategies differ from 0.10.0.

### Fixed

- **`spacing` was under-delivered by the NFP-based strategies** (`nfp`, `gdrr`,
  `alns`, and `ga`/`brkga`/`sa` whenever their own layout was returned). The
  no-fit polygon was grown by moving each vertex away from the polygon's
  centroid, which moves an edge only by the part of that displacement along its
  normal — two squares requested 50 apart were placed 40 apart, and L-shaped
  parts could end up at under two thirds of the requested distance. It is now a
  true polygon offset: every pair of placed parts is at least `spacing` apart,
  exceeding it by at most 0.12 %.
- **`margin` was applied twice by the `nfp` strategy**: a part was placed at
  `2 × margin` from the edge, and a sheet with room for a part inside a single
  margin placed nothing.
- **The NFP-based strategies also kept `spacing` from the boundary edge**, which
  the bottom-left-fill strategy never did; clearance to the edge is `margin`
  alone for every strategy now.
- **A part that must touch both a placed part and the boundary could only land
  on the next sampling point**, leaving a gap up to the sampling step wider than
  `spacing`. The placement search now also tries the points where the boundary
  and the no-fit polygons cross.
- `compute_ifp_with_margin` on a non-rectangular boundary pulled its vertices
  toward the vertex centroid instead of moving its edges inward by `margin`.
- **A non-rectangular boundary was packed as its bounding box.** The NFP-based
  strategies computed where a part fits from the boundary's bounding box and
  relied on a final filter to drop parts outside the actual shape, so parts
  near a slanted or concave edge were either discarded — a triangular sheet
  held fewer parts than it had room for — or kept closer to that edge than
  `margin`, which the filter did not check. Parts are now fitted to the
  boundary polygon moved inward by `margin`, and the final filter enforces
  `margin` against every boundary edge, including holes.
- The final boundary filter ignored mirroring (`allow_flip`): it checked the
  unmirrored outline of a mirrored part.
- **A boundary with holes placed nothing.** Deciding whether a part lies inside
  a boundary with holes (or any non-rectangular boundary) counted a part flush
  against an edge as crossing it, so every part touching the sheet edge was
  rejected; a 1000 × 1000 sheet with a 600 × 600 hole placed 0 of the 64 parts
  its border holds. The boundary is now a closed region, decided by area.
- **Holes were invisible to placement.** The NFP-based strategies placed parts
  over a hole and lost them to the boundary check; holes are now regions a
  part keeps `margin` away from, and the same sheet places all 64.
- **`time_limit_ms` was overrun many times over by the search strategies.**
  The genetic algorithm and BRKGA evaluated a whole population, and simulated
  annealing a whole temperature step, before looking at the clock; each
  evaluation is a full placement, and the best individual was placed once more
  at the end. A one-second limit took 23 s for 80 parts. The clock is now checked
  between evaluations and between the pieces of each placement, the best layout
  found is returned directly, and every strategy finishes within its limit.
- **GDRR and ALNS could return fewer parts than the greedy baseline** when their
  time ran out mid-search (56 of 160 parts on a 2 s limit, where bottom-left fill
  places all of them). Like the other search strategies, they now return the
  greedy layout when it is better.
- **Every genetic-algorithm and simulated-annealing solve left a thread behind**
  (2D and 3D): a thread was started to forward cancellation and only exited when
  the solve was cancelled. The runners now watch the caller's cancellation flag
  directly, which also makes cancellation work on WebAssembly, where that thread
  was never started.
- **Layouts ignored which way the sheet runs.** Placement by no-fit polygons
  always filled along `x` first and bottom-left fill always filled rows along
  `x`, so one of the two orientations used several times the length it needed:
  40 L-shaped parts took 1560 on a 500 × 5000 sheet with `nfp` (229 with rows),
  and 2040 on a 5000 × 500 sheet with `blf`. Both now fill across the sheet and
  advance along its longer side — the axis the search strategies already
  measured layouts by — so `nfp` and the search strategies built on it produce
  shorter layouts than bottom-left fill on either orientation.
- **A search strategy could return a longer layout than the one greedy pass it
  is built on.** Starting from random orders, a time-limited genetic algorithm or
  simulated annealing with four rotations ended at 328 where greedy no-fit
  placement reached 286, and adding rotations made results worse. Every search
  strategy now runs greedy no-fit placement first, inside its time limit, and
  never returns anything worse than it; the genetic algorithm and simulated
  annealing also start their search from that layout.
- **3D extreme-point packing lost boxes whenever `spacing` was positive.** New
  candidate corners were created flush against the last box, where a neighbour
  may not start once a gap is required, so eight 40-unit boxes that fit a
  100-unit container with a 5-unit gap placed one. Candidates now start
  `spacing` past each face.

### Added

- `GaRunner::with_cancellation` and `SaRunner::with_cancellation`, matching
  `BrkgaRunner::with_cancellation`: the runner stops when the caller's flag is set.

### Changed

- **Breaking:** `placement_utils::expand_nfp` and `placement_utils::shrink_ifp`
  are removed — they implemented the incorrect growth above. Use
  `placement_utils::offset_nfp` (same arguments); there is no replacement for
  `shrink_ifp`, because the inner-fit polygon is no longer shrunk by `spacing`.
  `placement_utils::inset_boundary` is the one place a boundary is inset by
  `margin`.
- **Breaking:** `ga_nesting::run_ga_nesting`, `run_ga_nesting_with_progress` and
  `sa_nesting::run_sa_nesting` take a final `seed_layout` (pass `None` to start
  from random solutions as before); `placement_utils::seed_genes` turns a layout
  into the genes that reproduce it.
- **Breaking:** `nfp::find_bottom_left_placement` takes the packing axis to
  order candidates by (`nfp::PackingAxis::of(&boundary)`).
- **Breaking:** `is_placement_within_bounds` and `validate_and_filter_placements`
  take the `margin` to enforce (`is_placement_within_bounds(placement, geometry,
  boundary, margin, tolerance)`); pass `0.0` for the previous check.
- Dropped dependencies that no code used: `log` and `u-numflow` from
  `u-nesting-core`; `thiserror` and `rand` from `u-nesting-cutting`;
  `thiserror` from `u-nesting-d2`; `thiserror` and `rayon` from
  `u-nesting-d3` (its `parallel` feature now only forwards to
  `u-nesting-core/parallel`, which is where the parallel code lives); `libc`
  and `log` from `u-nesting-ffi`. Each was declared, and kept current through
  version bumps, without being named anywhere in the source -- it only added a
  crate to every build. CI now checks for this with `cargo-machete`.

## [0.10.0] - 2026-09-07

A minor rather than a patch release for two independent reasons: the Python
bindings now reject input they used to accept, and `rand`'s types — which appear
in this crate's public generic bounds — have been replaced wholesale.

### Added

- **Unit tests for the Python bindings** (`u-nesting-python`), which previously
  had none: the crate was compiled by CI but nothing asserted its runtime
  behavior, unlike every other binding surface. The new module covers the
  interpreter-independent half of the contract — configuration building
  (strategy parsing, rejection of unknown names, non-negative/finite `spacing`
  and `margin`, `target_utilization` clamping, tuning parameters and seed),
  the derived output accounting (instance-level unplaced count with saturating
  subtraction, `all_placed`, and the zero-footprint guard in
  `used_utilization`), and the `serde` defaults of every input type.

### Changed (breaking, Python bindings)

- **The Python bindings now reject unknown keys in `geometries`, `boundary` and
  `config`** instead of ignoring them. They deserialize the same canonical
  request types as the C and WebAssembly bindings, which have rejected unknown
  keys since the strictness sweep; the Python bindings kept a private copy of
  those types and so were the one surface still accepting them. A misspelled key
  used to be discarded silently — `{"quantiy": 5}` produced a solve for one copy
  instead of five, with no error anywhere — and now raises `ValueError` naming
  the unknown key and listing the accepted ones. Callers passing extra keys must
  remove them. Field names, defaults and result shapes are unchanged.

### Fixed

- **`available_strategies()` in the Python bindings omitted `gdrr` and `alns`.**
  Both solve 2D problems in this build — and are advertised by the WebAssembly
  bindings — but Python callers had no way to discover them from the library
  itself. The published list, the type stub and the README now include them.
  Exact (MILP) strategies remain unlisted on purpose: their public path handles
  axis-aligned rectangles only.

### Changed

- **The minimum supported Rust version is now 1.89** (previously declared as
  1.82). The declared value is now verified by building on that exact toolchain:
  1.87 and below fail.
- **`rand` is now 0.10** (previously 0.9), and this is a breaking change at the
  API boundary rather than an internal one: `rand::Rng` appears in the public
  generic bounds of `u-nesting-core`'s genetic-algorithm and BRKGA entry points
  (`run_with_rng`, `biased_crossover`, `order_crossover`, `swap_mutate` and
  their neighbours), so a generator from `rand` 0.9 no longer satisfies them.
  Callers that drive those APIs with their own generator must move to 0.10 as
  well. The sequences produced for a given seed are unchanged, so seeded results
  are identical to the previous release.
- **`u-metaheur` is now required at 0.4 and `u-numflow` at 0.4** (previously 0.3
  for both), following those crates' own `rand` 0.10 releases.
- **`getrandom` is now 0.4** on WebAssembly targets. It reaches the browser
  entropy source through its `wasm_js` crate feature alone; the
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that 0.3 required is no longer
  needed. A separate 0.2 dependency, kept only to configure a transitive
  requirement that no longer exists, has been dropped.
- The `i_overlay` bump below is now verified on the Python bindings as well:
  `u-nesting-python` compiles against 8.1 and, driven through a real
  interpreter, `solve_2d` (`nfp`, the strategy that exercises `i_overlay`) and
  `solve_3d` (`ep`) place every requested instance.
- Bump `i_overlay` dependency from 1.9 to 8.1 (dependency freshness sweep,
  Tier 3 breaking migration). Crate usage is confined to three stable
  paths — `core::fill_rule::FillRule`, `core::overlay_rule::OverlayRule`,
  and the `float::single::SingleFloatOverlay::overlay()` trait method on
  `Vec<Vec<[f64; 2]>>`/`Vec<[f64; 2]>` — none of which changed shape
  across the six major releases in between. No API changes required in
  `u-nesting-d2`; full workspace check (`d2`/`d3`/`cutting`/`ffi`/`wasm`/
  `benchmark`/`u-nesting`, excluding `python`) green, clippy 0 warnings,
  and all `d2` tests unchanged in count and outcome: 197 lib (default) /
  202 lib (`--features milp`) / 40 integration / 4 fuzz — including the
  fuzz suite that uses `i_overlay`'s own boolean intersection as an
  independent oracle for overlap correctness, which would be the first
  place a solver-precision or contour-winding-order change (introduced
  upstream at `i_overlay` 3.0.1) surfaced as a regression.

## [0.9.0] - 2026-08-23

### Removed

- **`ExactConfig::use_symmetry_breaking` and `ExactConfig::use_cuts`** (and their
  `with_symmetry_breaking`/`with_cuts` builders) — both fields were silent no-ops.
  `use_symmetry_breaking` was read by the MILP solver but guarded a loop whose body
  never added any constraint; `use_cuts` was never read anywhere at all. Neither
  affected solution correctness (both were meant as pure performance hooks), but a
  documented config knob that does nothing is worse than no knob — a caller could
  reasonably believe toggling it changed solver behavior. Removed rather than
  implemented: no demonstrated performance need justifies the correctness risk of
  adding real symmetry-breaking constraints to a solver whose exactness this same
  release line has spent three prior fixes repairing (see 0.8.1 below). If a real
  need for either surfaces, they can be reintroduced with an actual implementation
  behind them. Breaking for any direct consumer of `u_nesting_core::exact::ExactConfig`
  building these fields; not reachable through the JSON/FFI/WASM config surface, which
  never exposed them.

## [0.8.1] - 2026-08-23

### Fixed

- **`Strategy::MilpExact` (and `HybridExact`'s exact-first attempt) placed
  almost nothing except axis-aligned rectangles.** After picking a candidate
  position, the solver converted its own grid-anchor coordinate directly into
  the piece's placement origin without translating by the piece's rotated
  local-frame offset — every other placement strategy already applies this
  conversion when turning a grid candidate into a `Placement`. For an
  axis-aligned rectangle that offset happens to be zero, so rectangles placed
  correctly by coincidence; any other shape (concave or convex, at any
  rotation the solver picked to minimize strip length) was silently
  positioned outside the boundary and then dropped, coming back as unplaced
  with no error. Fixed by applying the same anchor-to-origin correction the
  other strategies use.
- **`Strategy::MilpExact` could select two genuinely overlapping placements
  once a piece's own rotation was non-zero.** Its internal conflict check
  between two candidates compared their positions directly in global
  coordinates against a No-Fit Polygon computed assuming the reference piece
  sits unrotated at its own local origin — correct only while that reference
  candidate happened to be unrotated. Only reachable with two or more
  instances (a single piece never has a conflict to check), so this was
  latent for as long as the bug above kept multi-piece exact solves from
  reaching a rotated candidate at all. Fixed by carrying the No-Fit Polygon
  into absolute space the same way the other strategies already do: rotate
  it by the reference candidate's own rotation, then translate to its actual
  placement origin, before testing the other candidate's position against it.
- **`Strategy::MilpExact`'s `time_limit_ms` config bounded only its internal
  conflict-precomputation heuristic, never the actual MIP solve** — the one
  step that can legitimately run the longest on a hard instance. A caller
  setting a short time budget got no such bound where it mattered most.
  Fixed by passing the configured limit to the underlying HiGHS solver
  itself before solving.

## [0.8.0] - 2026-08-21

### Added

- **`allow_flip` (mirror reflection) is now supported end-to-end, across every
  placement strategy** (Bottom-Left Fill, NFP-guided BLF, Genetic Algorithm,
  BRKGA, Simulated Annealing, GDRR, ALNS, and the MILP-exact solver). A piece
  marked `allow_flip: true` can now be placed either in its original
  orientation or mirrored across its local axis, whichever a strategy finds
  better — previously `allow_flip: true` was rejected outright with an
  explicit error, since no strategy implemented mirroring yet. `Placement`'s
  existing `mirrored` field (and the 2D response's `flipped` field) now
  actually reflects what was solved; no wire-schema changes were needed since
  both fields already existed.

### Fixed

- **Mirrored placements could overlap an already-placed piece under NFP-guided
  strategies.** The boundary-clamp step that runs after a strategy picks a
  placement candidate computed the piece's bounding box from its *unmirrored*
  orientation regardless of whether the winning candidate was actually
  mirrored. For a piece whose local shape isn't symmetric about its own
  reference point, that bounding box is wrong for a mirrored placement — the
  clamp could then shift an already collision-free position into one that
  overlaps a previously placed piece. Fixed by computing the clamp bounds
  from the mirrored orientation whenever the candidate is mirrored. Only
  reachable together with the `allow_flip` support above, since mirroring had
  no live path before this release.
- **`Strategy::MilpExact` and `Strategy::HybridExact` never actually reached a
  working solver.** Both called into an older continuous-position MILP
  formulation that placed nothing for any input through the public solve
  entry point — not even a single unrotated, unmirrored rectangle — so
  `MilpExact` always returned every piece unplaced and `HybridExact` always
  silently fell back to its heuristic path. Fixed by routing both to the
  NFP Covering Model formulation that already had working, tested coverage
  (including the `allow_flip` support above); the older module is removed.

## [0.7.2] - 2026-07-15

Documentation-only clarification from continued fabric-cutting dogfooding. No
runtime behaviour changes; fully backward compatible.

### Documentation

- **`used_utilization` semantics clarified across all bindings (Rust, C#,
  Python).** The field is a *packing-density* metric: its denominator is the
  tight bounding box of the placed pieces, shrinking on **both** the width and
  length axes. For **fixed-width stock** (fabric rolls, coil, sheet stock) the
  boundary width is real consumed material, not padding, so collapsing the width
  axis over-reports how much material was saved (measured up to +13.4%p on a
  sleeve layout using 79% of a 1580 mm roll width). The docs now state this
  explicitly and direct fixed-width consumers to compute
  `placed_area / (boundary_width × used_bounding_box[1])` instead — the consumed
  length (`used_bounding_box[1]`) is already exposed and boundary-independent.
  A dedicated `stock_utilization` field is deferred pending cross-consumer
  demand (single-consumer use is already covered by the formula above).

## [0.7.1] - 2026-07-14

Follow-up hardening from the same fabric-cutting dogfooding pass. Behaviour of
the greedy strategies is unchanged; the fixes make the metaheuristic quality
floor and reproducibility actually hold. Fully backward compatible.

### Fixed

- **Metaheuristic quality floor now compares strip length, not bounding-box
  area.** `not_worse_than_blf` guaranteed only that GA/BRKGA/SA place at least as
  many pieces as BLF, comparing ties by bounding-box *area*. On an open-ended
  roll (fixed width, variable length) a taller, narrower layout has a smaller
  area yet consumes a *longer* strip, so a rotation-driven GA result that packed
  the same pieces into a longer roll than plain BLF slipped through the floor
  (e.g. 8 L-shapes: BLF 306, GA up to 812). The floor now compares the extent
  along the boundary's open (longer) axis — the material length consumers
  measure — so the stochastic strategies are provably never worse than BLF, and
  enlarging the allowed `rotations` set never worsens the result.
- **Seed reproducibility on the progress/callback path.** `solve_with_progress`
  (used by the FFI `solve_2d_with_callback` and the WASM/demo path) ran the GA
  from system entropy, so a seeded solve was non-deterministic whenever a
  progress callback was supplied. The seed is now threaded through the progress
  runner, matching the plain `solve` path.
- **Progress GA no longer runs the full default budget.** The callback path used
  the default 500 generations × 100 population instead of the capped 50 × 30 of
  the non-progress path, making a progress-driven solve far slower and prone to
  overrunning a modest time budget. Both paths now use the same caps.

### Added

- Robustness fuzz harness (`crates/d2/tests/fuzz_robustness.rs`): `solve` never
  panics on arbitrary/pathological polygons, and BLF/NFP layouts are verified
  overlap-free (convex and concave) with an independent `i_overlay` oracle.
- End-to-end tests that `guard_panic` converts an internal panic into a
  `success: false` response (previously verified only by construction).

## [0.7.0] - 2026-07-14

Dogfooding hardening pass from a fabric-cutting consumer (open-roll nesting of
irregular parts). All response additions are backward compatible — existing
fields keep their meaning and `#[serde(default)]` keeps old payloads valid.

### Added

- **Piece accounting on the solve response.** `SolveResponse`/`Pack3DResponse`
  gain `all_placed: bool` and `unplaced_count: usize`. `success` keeps its
  meaning ("solve completed without crashing"); full placement is now signalled
  by `all_placed` (`placements.len() == total_requested`), so an over-committed
  run no longer looks successful. `unplaced_count` is the instance-level count,
  complementing the deduplicated `unplaced` geometry-id list.
- **Padding-independent utilization.** 2D responses gain
  `used_bounding_box: [width, height]` (AABB of the placed pieces' actual
  footprint) and `used_utilization` (`piece_area / (used_w × used_h)`). Unlike
  `utilization`, these do not shrink as the boundary height grows — for an
  open-ended roll the larger axis is the material length consumed. Single-sheet
  solves only; `[0, 0]`/`0` for multi-sheet (per-sheet local frames make one
  footprint ill-defined) and on error.
- **Reproducible stochastic runs.** `Config` gains an optional `seed` (`seed?`
  in the binding config JSON), threaded into GA/BRKGA/SA so a fixed seed yields
  a deterministic layout. `None` seeds from system entropy.
- **`Strategy::parse`** — single source of truth for strategy-name parsing
  (case-insensitive, all aliases) shared by the C-FFI, Python, and WASM
  bindings.

### Fixed

- **Pieces no longer escape non-rectangular boundaries.** Placement containment
  used only an AABB test, so pieces could stick out past a triangular/concave
  boundary's edges and still be reported as placed. Containment now does real
  polygon-in-polygon testing (point-in-polygon + edge-intersection) for polygon
  boundaries, while plain rectangles and infinite strips keep the exact fast
  AABB path. Escaped pieces are rejected and correctly counted as unplaced.
- **Unknown strategy names are now an error** instead of a silent fall-back to
  `BottomLeftFill`, which hid consumer typos. (The FFI parser had also been
  missing `brkga`/`gdrr`/`alns`/`exact`, silently downgrading them to BLF.)
- **Invalid input geometry is rejected.** `Geometry2D::validate()` now rejects
  zero-area/collinear polygons (relative-epsilon signed-area test) and
  self-intersecting rings, and `allow_flip = true` errors until mirroring is
  implemented (was silently ignored). Arbitrary polygon boundaries get the same
  degeneracy/self-intersection checks.
- **Input validation can no longer be bypassed.** Per-geometry `validate()` was
  scattered across the per-strategy entry points, so the progress/callback path
  skipped it. Validation is hoisted to the top of `solve`,
  `solve_with_progress`, and `solve_multi_strip`.
- **Progress path respects the BLF quality floor.** The `solve_with_progress`
  GA branch bypassed the `not_worse_than_blf` guard that the plain `solve` path
  uses, so a callback-driven run could return a worse layout than BLF.
- **An internal panic no longer aborts the host process.** `panic = "abort"`
  is removed from the release profile so unwinding reaches the `catch_unwind`
  at the C-FFI entry points, which convert a panic into an error response
  instead of `SIGABRT`-ing the C#/Python/WASM host.

## [0.6.0] - 2026-07-07

### Fixed

- **Cutting-path sequencing is now wall-clock bounded** (`optimize_cutting_path`,
  all bindings). The 2-opt improvement phase had no time limit — only an
  iteration cap — and each candidate move re-evaluated the whole tour, so
  legitimate inputs (e.g. hundreds of identical parts filling a sheet) ran for
  many seconds and blocked the calling thread; in the browser this froze the
  tab (effectively a DoS on normal use). `CuttingConfig` gains
  `time_limit_ms` (default `5000`, `0` = unlimited), plumbed through the WASM
  and C-FFI `cutting_config` JSON (`time_limit_ms?: number`). On timeout the
  best sequence found so far is returned — cut order is heuristic, so early
  termination never invalidates the result. Independently, per-move cost
  evaluation no longer does a linear `contours.iter().find()`, dropping a
  factor of `n` from every 2-opt pass. Applies to both the legacy
  (`pierce_candidates <= 1`) and GTSP (`pierce_candidates > 1`) paths.

## [0.5.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.5.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry** (`@iyulab/u-nesting`). The npm
  package previously shipped only the wasm-bindgen *bundler*-target output,
  whose static `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in
  non-ESM packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports` map
  (`node` → CJS with filesystem wasm loading, `default` → bundler ESM).
  `require()`, native ESM `import`, and CJS TS runners all work without
  loader hooks. A pre-publish smoke test (CJS `require` + ESM `import`) now
  guards this path in CI. Wire schema and Rust/C#/Python APIs unchanged;
  0.5.1 is a lockstep version bump across all bindings (npm re-publish
  requires a new version, and binding versions are pinned in lockstep).

## [0.5.0] - 2026-06-21

### Added

- **`total_requested` on solve responses** (`solve_2d`/`solve_3d`). Both
  `SolveResponse` and `Pack3DResponse` now carry `total_requested: usize` — the
  Σ of every geometry's `quantity` (instance-level request total). `placements`
  is instance-level while `unplaced` lists **deduplicated** unique geometry IDs,
  so previously the per-instance unplaced count could not be derived from the
  response (`unplaced.len()` under-reports when a multi-quantity geometry fails).
  Consumers can now compute it directly as `total_requested - placements.len()`.
  The field is additive and `#[serde(default)]` (absent legacy/`optimize_cutting_path`
  passthrough payloads deserialize to `0`); on error responses it is `0`.
  Mirrored across the FFI, WASM, and Python bindings; C# binding models updated
  in lockstep. Resolves the instance-count under-reporting reported by consumers.

- **`config.multi_sheet` for 2D nesting** (`solve_2d`). New optional config flag
  (`Option<bool>`, default `false`). When `true`, parts that do not fit on a
  single sheet spill onto **additional sheets** instead of becoming `unplaced`:
  `sheets_used` reports the sheet count and each placement's `sheet_index`
  selects its sheet. Placement coordinates are **sheet-local** (relative to each
  sheet's origin), so consumers can render per-sheet panels without recomputing
  offsets. Previously the multi-sheet solver (`solve_multi_strip`) existed but was
  reachable only from the benchmark harness — WASM/FFI/Python/C# now expose it via
  the flag. The field is additive and optional (absent → single-sheet solve).
  Resolves the schema-vs-capability gap where `sheets_used`/`sheet_index` were
  declared but never exceeded 1/0.

### Fixed

- `SolveSummary.total_requested` previously computed `placements.len() +
  unplaced.len()`, which **undercounted** whenever a multi-quantity geometry was
  partially/fully unplaced (because `unplaced` is deduplicated). It now uses the
  authoritative instance-level total recorded at solve time.

- **Multi-sheet overflow no longer silently loses instances.**
  `Nester2D::solve_multi_strip` reduced a geometry's *remaining* set by unique
  geometry ID, so once **any** instance of a `quantity > 1` geometry was placed,
  the rest were dropped from both `placements` **and** `unplaced` (silent loss).
  It now reduces remaining quantity by the **instance count** placed on each
  sheet, carrying the remainder forward, and assigns globally-unique
  `(geometry_id, instance)` pairs across sheets. Genuinely oversized geometries
  are reported via an after-loop `unplaced` sweep. `solve_multi_strip` also now
  records `total_requested` (it previously left it `0`).

- **C# config key drift corrected.** The C# `Config2D`/`Config3D` DTOs serialized
  the GA/BRKGA generation cap as `"generations"`, but the Rust wire contract
  (`ConfigRequest`, `deny_unknown_fields`) only accepts `"max_generations"` — so
  setting it from C# caused the FFI to **reject the whole request**. The property
  is renamed `Generations` → `MaxGenerations` (`[JsonPropertyName("max_generations")]`),
  and the same key was corrected across the user-guide examples. **Breaking for C#
  callers** that set `Config2D.Generations`. A new `UNesting.Tests` project plus a
  CI `csharp` job now pin every emitted C# config key to the canonical Rust wire
  names so this class of drift fails in CI.

### Security

- Upgrade PyO3 `0.24` → `0.29` to remediate two advisories flagged by
  `cargo audit`:
  - **RUSTSEC-2026-0176** — out-of-bounds read in `nth`/`nth_back` for `PyList`
    and `PyTuple` iterators (unchecked arithmetic). Fixed in PyO3 0.29.0.
  - **RUSTSEC-2026-0177** — missing `Sync` bound on `PyCFunction::new_closure`
    closures (thread-safety). Fixed in PyO3 0.29.0.
  Python binding return types (`solve_2d`/`solve_3d`) migrated from the
  prelude-removed `PyObject` to `Bound<'py, PyAny>`; no Python-facing API change.

## [0.4.0] - 2026-06-12

### Changed — BREAKING (wire schema)

- All request objects (`solve_2d`/`solve_3d`/`optimize_cutting_path` —
  including nested geometry, boundary, and config objects) now **reject
  unknown keys** with an explicit `unknown field` error instead of silently
  ignoring them (`serde(deny_unknown_fields)`). Typos and unsupported options
  previously failed silently; remove any extra keys when upgrading.
- Response wire types apply the same strictness when deserialized (relevant
  for the `optimize_cutting_path` `solve_result` passthrough).

### Added

- `Request2D`/`Request3D` gain an optional `mode` field (`"2d"`/`"3d"`) —
  the auto-detect `solve` entry point's discriminator is now a declared part
  of the schema instead of an incidentally-tolerated extra key. It is omitted
  from serialization when absent.

### Changed

- Dependency: `u-metaheur` `^0.2` → `^0.3`.

## [0.3.5] - 2026-06-04

### Fixed
- **3D Extreme-Point under-packing**: the EP heuristic precomputed a per-axis "residual"
  free-space approximation and gated candidate points on it. That model under-counted
  space for boxes placed flush against each other, discarded valid extreme points, and
  stalled the pack (perfect-fill scenarios placed only 4/8 cubes). Replaced with an exact
  fit test against the container bounds and placed boxes at placement time
  (`ExtremePointSet::fits_at`): perfect-fill now 8/8, EP places 18 boxes vs BLF's 13.
  Also corrected an `EP → BLF` mis-routing in `solve_with_progress`.

### Changed
- **3D gravity & stability now enforced in `solve`**: the physics/stability analyzer was
  built but never wired into the solver. Added `enforce_support`, which reuses the
  analyzer to drop unsupported boxes and recompute utilization, with the support floor at
  `floor_z = margin` (a hard-coded `floor_z = 0` removed every box when `margin > 0` and
  gravity was on). Gravity/stability toggles restored in the demo.

## [0.3.4] - 2026-06-04

### Fixed
- **WASM runtime failures (0.3.3 regression)**: solve panicked under `wasm32` because the
  SA loop was not gated for the single-threaded target and used a raw `Instant`. Promoted
  `core::timing::Timer` to a `web-time` backend so `time_limit` works in the browser, and
  gated parallel paths. A single-strip time budget could exceed the total limit; the
  `.max(5000)` floor now also takes `.min(total)`.
- **3D extreme-point out-of-bounds**: a swapped `fits` argument (`extreme_point.rs`) could
  place a box outside the container.
- **GA/BRKGA orientation mis-report**: `rotation_index` was left unset, so chosen
  orientations were reported incorrectly in the response.
- Added a WASM runtime smoke suite (`wasm_runtime_smoke.mjs`, 18 cases) to CI.

## [0.3.3] - 2026-06-03

### Fixed
- **3D FFI/WASM response schema**: 3D solve now emits a dedicated `Pack3DResponse`
  / `Placement3DResponse` wire format matching the C# `PackingResult` / `Placement3D`
  binding. Previously the 3D path reused the 2D `SolveResponse`, so C# (and npm)
  consumers silently lost each placement's depth (`z`), `orientation` label and
  `bin_index` (and received `sheets_used` instead of `bins_used`). Same class as the
  0.3.1→0.3.2 2D `rotation` array-vs-scalar fix.

### Added
- `Geometry3D::orientation_label(idx)` — maps an orientation index to an axis
  permutation string (`"xyz"`, `"xzy"`, …).
- `build_pack3d_response` (u-nesting-d3) — canonical 3D wire-response builder shared
  by the C FFI and WASM bindings.
- FFI JSON schema regression guards pinning the Rust output to the C# binding
  contract for both 2D and 3D (`schema_guard_2d/3d_matches_csharp_*`).

### ⚠️ BREAKING — 3D wire schema (migration from ≤ 0.3.2)
3D `solve` responses moved from the 2D `SolveResponse` shape to the dedicated
`Pack3DResponse` / `Placement3DResponse`. Consumers reading 3D output must rename:

| ≤ 0.3.2 (old) | 0.3.3+ (current) |
|---------------|------------------|
| `placement.geometry_id` | `placement.id` |
| `placement.position: [x, y, z]` | flat `placement.x`, `placement.y`, `placement.z` |
| `placement.rotation: [rx, ry, rz]` (numeric) | `placement.orientation: "xyz"` (axis-permutation **string**) |
| `placement.boundary_index` | `placement.bin_index` |
| `response.boundaries_used` | `response.bins_used` |
| `response.computation_time_ms` | `response.elapsed_ms` |

The numeric `rotation` → string `orientation` change is semantic: `orientation` is
an axis-permutation label (see `Geometry3D::orientation_label`), not Euler angles.
Map it to axis swaps in your renderer rather than treating it as a rotation vector.

## [0.3.2] - 2026-05-12

### ⚠️ BREAKING — 2D wire schema (`align …JSON schema with C# binding`)
2D `solve` `PlacementResponse` / `SolveResponse` JSON fields were renamed and
restructured to match the C# `Geometry2D` / nesting-result binding. This is a
silent runtime break for JSON consumers (the WASM API returns a JSON string, so
TypeScript/`tsc` does **not** catch it). Migration from 0.3.1:

| 0.3.1 (old) | 0.3.2+ (current) |
|-------------|------------------|
| `placement.geometry_id` | `placement.id` |
| `placement.position: [x, y]` | flat `placement.x`, `placement.y` |
| `placement.rotation: [r]` (array) | `placement.rotation` (scalar number) |
| `placement.boundary_index` | `placement.sheet_index` |
| `response.boundaries_used` | `response.sheets_used` |
| `response.computation_time_ms` | `response.elapsed_ms` |
| — | + `placement.flipped` (bool) |

### Fixed
- C# JSON schema mismatch: Rust 2D output now matches the C# binding contract.

### Added
- NuGet release workflow + `.targets` for native DLL auto-copy; auto-trigger on
  `.csproj` version change.

## [0.3.1] - 2026-03-09

### Fixed
- WASM-compatible `Timer` abstraction to prevent `Instant::now()` panic under
  `wasm32` (no monotonic clock in the browser target).

### Added
- npm publish workflow for `@iyulab/u-nesting`.

## [0.3.0] - 2026-03-08

### Added
- `u-nesting-wasm` crate — 2D nesting, 3D packing, and cutting-path optimization
  exposed via WebAssembly (JSON string in / JSON string out).
- Shared `api_types` module: request/response types extracted so the C FFI and
  WASM bindings share one JSON schema definition.
- Cutting-path optimization crate (GTSP solver, kerf compensation, lead-in/out,
  bridge/tab micro-joints, thermal HAZ model, common-edge detection).

### Changed
- `rayon` made optional across `core`/`d2`/`d3` for WASM compatibility; parallel
  vs sequential branching now selected via `cfg` attributes.

## [0.2.0] - 2026-02-09

### Changed

#### Dependencies
- **BREAKING**: Removed `geo`, `geo-types`, `parry2d`, `parry3d` dependencies
- Replaced `geo` polygon operations with `u-geometry` equivalents
- Replaced `parry2d`/`parry3d` collision detection (unused) — removed entirely
- Replaced direct `nalgebra` imports with `u-geometry` re-exports
- Updated `rand` 0.8 → 0.9 (135 API changes across 14 files)
- Updated `pyo3` 0.22 → 0.24
- MSRV raised from 1.75 to 1.82 (uses `is_none_or` stabilized in 1.82)

#### Code Quality
- Eliminated all `unwrap()` in library code (replaced with `expect()`)
- Fixed all clippy warnings across workspace
- Extracted shared placement utilities to reduce code duplication (-561 LOC)
- Replaced duplicate robust predicates with `u-geometry` re-exports

### Added
- `u-geometry` integration for polygon operations (area, centroid, perimeter, convex_hull)
- `u-geometry` integration for convex NFP/Minkowski sum computation
- Polygon with holes support via `u-geometry` (area_with_holes, centroid_with_holes)
- SA, BRKGA, ALNS, GDRR algorithm documentation
- 2D nesting benchmarks (criterion)
- Benchmark build check in CI

### Removed
- `geo` 0.29 dependency
- `geo-types` 0.7 dependency
- `parry2d` 0.17 dependency
- `parry3d` 0.17 dependency
- Duplicate geometry functions (~200 LOC replaced by u-geometry delegation)

### Dependencies
- `u-geometry` 0.1 — Computational geometry primitives
- `u-metaheur` 0.1 — Metaheuristic optimization
- `u-numflow` 0.1 — Mathematical primitives
- `i_overlay` 1.9 — Boolean polygon operations
- `rstar` 0.12 — R*-tree spatial indexing
- `rayon` 1.10 — Parallelization
- `rand` 0.9 — Random number generation
- `pyo3` 0.24 — Python bindings

## [0.1.0] - 2026-01-21

### Added

#### Core Library (`u-nesting-core`)
- **Solver Framework**: Generic `Solver` trait with `Config` and `Strategy` enum
- **Genetic Algorithm**: `GaRunner` with tournament selection, elitism, and parallel evaluation
- **BRKGA**: Biased Random-Key Genetic Algorithm with random-key encoding
- **Simulated Annealing**: Multiple cooling schedules (Geometric, Linear, Adaptive, Lundy-Mees)
- **Placement System**: `Placement` struct with position, rotation, and boundary tracking
- **Transform Utilities**: `Transform2D`, `Transform3D`, `AABB2D`, `AABB3D`
- **Memory Optimization**: `ObjectPool`, `ClearingPool`, `SharedGeometry`, `GeometryCache`, `ScratchBuffer`
- **Progress Callbacks**: `ProgressInfo`, `GaProgress`, `BrkgaProgress` for real-time feedback

#### 2D Nesting (`u-nesting-d2`)
- **Geometry2D**: Polygon representation with holes, area, centroid, convex hull
- **Boundary2D**: Rectangular and arbitrary polygon boundaries
- **Nester2D Solver**: Multiple placement strategies
  - Bottom-Left Fill (BLF)
  - NFP-guided placement
  - Genetic Algorithm optimization
  - BRKGA optimization
  - Simulated Annealing optimization
- **NFP Engine**: No-Fit Polygon computation
  - Convex polygons via Minkowski sum
  - Non-convex polygons via triangulation + union
  - Thread-safe caching system
  - Inner-Fit Polygon (IFP) with margin support
- **Spatial Index**: R*-tree based collision detection

#### 3D Bin Packing (`u-nesting-d3`)
- **Geometry3D**: Box representation with 6 orientation variants
- **Boundary3D**: Container with mass, gravity, and stability constraints
- **Packer3D Solver**: Multiple packing strategies
  - Layer-based packing
  - Extreme Point heuristic
  - Genetic Algorithm optimization
  - BRKGA optimization
  - Simulated Annealing optimization
- **Spatial Index**: AABB-based collision detection

#### FFI Layer (`u-nesting-ffi`)
- **C ABI**: `unesting_solve()`, `unesting_solve_2d()`, `unesting_solve_3d()`
- **JSON API**: Request/Response serialization with serde
- **API Versioning**: Version field in all responses (v1.0)
- **Error Codes**: `UNESTING_OK`, `UNESTING_ERR_NULL_PTR`, etc.
- **Header Generation**: cbindgen for C/C++ headers

#### Python Bindings (`u-nesting-python`)
- **PyO3 Integration**: Native Python module via maturin
- **Functions**: `solve_2d()`, `solve_3d()`, `version()`, `available_strategies()`
- **Type Stubs**: `.pyi` files for IDE autocompletion

#### Benchmark Suite (`u-nesting-benchmark`)
- **2D Benchmarks**: ESICUP dataset parser and runner
- **3D Benchmarks**: Martello-Pisinger-Vigo (MPV) instance generator
- **Result Analysis**: Strategy comparison, rankings, win matrices
- **Report Generation**: Markdown and JSON output formats

#### Documentation
- **JSON Schemas**: `request-2d.schema.json`, `request-3d.schema.json`, `response.schema.json`
- **API Documentation**: Module-level docs with usage examples
- **README**: Quick start guide with C# P/Invoke examples

### Performance
- Parallel NFP computation via rayon
- Parallel GA/BRKGA population evaluation
- Parallel SA restarts
- Thread-safe NFP caching

### Dependencies
- `geo` 0.29 - 2D geometry primitives
- `i_overlay` 1.9 - Boolean polygon operations
- `parry2d`/`parry3d` 0.17 - Collision detection
- `nalgebra` 0.33 - Linear algebra
- `rstar` 0.12 - R*-tree spatial indexing
- `rayon` 1.10 - Parallelization
- `pyo3` 0.22 - Python bindings

[Unreleased]: https://github.com/iyulab/U-Nesting/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/iyulab/U-Nesting/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/iyulab/U-Nesting/releases/tag/v0.1.0
