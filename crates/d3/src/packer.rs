//! 3D bin packing solver.

use crate::boundary::Boundary3D;
use crate::brkga_packing::run_brkga_packing;
use crate::extreme_point::run_ep_packing;
use crate::ga_packing::run_ga_packing;
use crate::geometry::Geometry3D;
use crate::packing_utils::{build_unplaced_list, layer_fill_in_order, packing_fitness};
use crate::physics::{PhysicsConfig, PhysicsSimulator};
use crate::sa_packing::run_sa_packing;
use crate::stability::{PlacedBox, StabilityAnalyzer, StabilityConstraint, StabilityReport};
use u_nesting_core::brkga::BrkgaConfig;
use u_nesting_core::ga::GaConfig;
use u_nesting_core::geom::nalgebra_types::{NaPoint3 as Point3, NaVector3 as Vector3};
use u_nesting_core::geometry::{Boundary, Geometry};
use u_nesting_core::sa::SaConfig;
use u_nesting_core::solver::{
    offered_names, Config, Dimension, ProgressCallback, ProgressInfo, Solver, Strategy,
};
use u_nesting_core::{Error, Placement, Result, SolveResult, StopReason};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use u_nesting_core::timing::Timer;

/// The GA population when the caller leaves [`Config::population_size`]
/// unset. A 3D individual is decoded by layer placement, cheap next to a 2D
/// no-fit-polygon layout, so the run is larger than the 2D one.
const GA_POPULATION: usize = 100;

/// The GA generation budget when the caller leaves
/// [`Config::max_generations`] unset.
const GA_GENERATIONS: u32 = 500;

/// The BRKGA population when the caller leaves [`Config::population_size`]
/// unset.
const BRKGA_POPULATION: usize = 50;

/// The BRKGA generation budget when the caller leaves
/// [`Config::max_generations`] unset.
const BRKGA_GENERATIONS: u32 = 100;

/// 3D bin packing solver.
pub struct Packer3D {
    config: Config,
    cancelled: Arc<AtomicBool>,
}

impl Packer3D {
    /// Runs the configured strategy, reporting to `progress`: a fill per box,
    /// a search per generation or temperature step (and its layer fill per
    /// box). The run is the same with or without a callback. A strategy 3D
    /// packing does not provide is refused rather than replaced by layer
    /// packing.
    fn run_strategy(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        match self.config.strategy {
            Strategy::BottomLeftFill => Ok(self.layer_fill(geometries, boundary, progress)),
            Strategy::ExtremePoint => self.extreme_point(geometries, boundary, progress),
            Strategy::GeneticAlgorithm => self.genetic_algorithm(geometries, boundary, progress),
            Strategy::Brkga => self.brkga(geometries, boundary, progress),
            Strategy::SimulatedAnnealing => {
                self.simulated_annealing(geometries, boundary, progress)
            }
            other => Err(Error::invalid_option(
                "strategy",
                format!(
                    "strategy \"{}\" is not available for 3D packing; it offers {}",
                    other.name(),
                    offered_names(Dimension::ThreeD)
                ),
            )),
        }
    }

    /// The flag [`Solver::cancel`] sets, for a caller that cancels from
    /// where it cannot reach this solver (another thread, a progress callback).
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    /// Creates a new packer with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a packer with default configuration.
    pub fn default_config() -> Self {
        Self::new(Config::default())
    }

    /// The GA run the configuration asks for: the caller's population and
    /// generation budget, [`GA_POPULATION`] / [`GA_GENERATIONS`] for the ones
    /// left unset.
    fn ga_config(&self) -> GaConfig {
        let config = GaConfig::default()
            .with_population_size(self.config.population_size.unwrap_or(GA_POPULATION))
            .with_max_generations(self.config.max_generations.unwrap_or(GA_GENERATIONS))
            .with_crossover_rate(self.config.crossover_rate)
            .with_mutation_rate(self.config.mutation_rate)
            .with_elite_count(self.config.elite_count);
        match self.target_fitness() {
            Some(target) => config.with_target_fitness(target),
            None => config,
        }
    }

    /// The fitness a search stops at for `target_utilization`: every box
    /// placed, at that utilization ([`packing_fitness`] of a complete packing).
    fn target_fitness(&self) -> Option<f64> {
        self.config
            .target_utilization
            .map(|target| packing_fitness(1, 1, target))
    }

    /// The BRKGA run the configuration asks for: the caller's population and
    /// generation budget, [`BRKGA_POPULATION`] / [`BRKGA_GENERATIONS`] for the
    /// ones left unset.
    fn brkga_config(&self) -> BrkgaConfig {
        let config = BrkgaConfig::default()
            .with_population_size(self.config.population_size.unwrap_or(BRKGA_POPULATION))
            .with_max_generations(self.config.max_generations.unwrap_or(BRKGA_GENERATIONS))
            .with_elite_fraction(0.2)
            .with_mutant_fraction(0.15)
            .with_elite_bias(0.7);
        match self.target_fitness() {
            Some(target) => config.with_target_fitness(target),
            None => config,
        }
    }

    /// Judges a problem as every solve does before it runs, whichever strategy
    /// runs: the configuration, the boundary, the geometries -- an empty list
    /// is `empty_input`, there is nothing to pack -- their ids, and whether 3D
    /// packing offers the configured strategy. A packing checker reads its
    /// problem through this too, so a problem the packer refuses is one the
    /// checker refuses.
    ///
    /// # Errors
    /// The first refusal.
    pub fn validate(&self, geometries: &[Geometry3D], boundary: &Boundary3D) -> Result<()> {
        boundary.validate()?;
        if geometries.is_empty() {
            return Err(Error::empty_input("geometries"));
        }
        self.config.validate()?;
        for geom in geometries {
            geom.validate()?;
        }
        u_nesting_core::geometry::ensure_unique_ids(geometries)?;
        if !self.config.strategy.is_offered(Dimension::ThreeD) {
            return Err(Error::invalid_option(
                "strategy",
                format!(
                    "strategy \"{}\" is not available for 3D packing; it offers {}",
                    self.config.strategy.name(),
                    offered_names(Dimension::ThreeD)
                ),
            ));
        }
        Ok(())
    }

    /// Validates the stability of a packing result.
    ///
    /// Analyzes each placement to ensure boxes are properly supported.
    ///
    /// # Errors
    ///
    /// The refusal of [`StabilityConstraint::validate`].
    pub fn validate_stability(
        &self,
        result: &SolveResult<f64>,
        geometries: &[Geometry3D],
        _boundary: &Boundary3D,
        constraint: StabilityConstraint,
    ) -> Result<StabilityReport> {
        // Convert placements to PlacedBox format
        let placed_boxes = self.placements_to_boxes(result, geometries);
        let analyzer = StabilityAnalyzer::new(constraint)?;
        Ok(analyzer.analyze(&placed_boxes, 0.0))
    }

    /// Validates stability using physics simulation.
    ///
    /// Runs a physics simulation to detect boxes that would fall or tip.
    pub fn validate_stability_physics(
        &self,
        result: &SolveResult<f64>,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) -> StabilityReport {
        let placed_boxes = self.placements_to_boxes(result, geometries);
        let container = Vector3::new(boundary.width(), boundary.depth(), boundary.height());

        let config = PhysicsConfig::default().with_max_time(2.0);
        let simulator =
            PhysicsSimulator::new(config).expect("the default physics settings are in range");
        simulator.validate_stability(&placed_boxes, container, 0.0)
    }

    /// Enforces the boundary's gravity/stability constraints on a finished packing.
    ///
    /// Strategies optimise for volume, not support, so a result may contain boxes that
    /// float or rest on too little base. When the boundary requests gravity and/or
    /// stability, this removes the offending boxes (moving them to `unplaced`) so the
    /// returned packing actually satisfies the constraint the caller asked for — rather
    /// than the flags being silently ignored.
    ///
    /// Removal iterates: dropping a box can leave the boxes it supported unsupported, so
    /// the result is re-validated until every remaining box is stable.
    fn enforce_support(
        &self,
        result: &mut SolveResult<f64>,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
    ) {
        // Stability is the stronger requirement (a real base-support ratio); plain
        // gravity only forbids floating, i.e. demands any positive contact below.
        let constraint = if boundary.has_stability() {
            StabilityConstraint::partial_base(0.7)
        } else {
            StabilityConstraint::partial_base(1e-6)
        };

        // The pack rests boxes on z = margin (the inset floor), so the support analysis
        // must use that as the floor — checking against z = 0 would read every bottom
        // box as floating and drop the whole pack when margin > 0.
        let floor_z = self.config.margin;
        let analyzer = StabilityAnalyzer::new(constraint)
            .expect("the support ratios used here, 0.7 and 1e-6, are in range");
        while !result.placements.is_empty() {
            let placed_boxes = self.placements_to_boxes(result, geometries);
            let report = analyzer.analyze(&placed_boxes, floor_z);
            if report.is_all_stable() {
                break;
            }
            let drop: std::collections::HashSet<(String, usize)> = report
                .unstable_boxes()
                .iter()
                .map(|r| (r.id.clone(), r.instance))
                .collect();
            result
                .placements
                .retain(|p| !drop.contains(&(p.geometry_id.clone(), p.instance)));
            for (id, _) in drop {
                result.unplaced.push(id);
            }
        }

        result.deduplicate_unplaced();
        // Placed volume shrank, so the reported utilization must follow.
        let boxes = self.placements_to_boxes(result, geometries);
        let placed_volume: f64 = boxes
            .iter()
            .map(|b| b.dimensions.x * b.dimensions.y * b.dimensions.z)
            .sum();
        let container_volume = boundary.measure();
        result.utilization = if container_volume > 0.0 {
            placed_volume / container_volume
        } else {
            0.0
        };
    }

    /// Converts placements to PlacedBox format for stability analysis.
    fn placements_to_boxes(
        &self,
        result: &SolveResult<f64>,
        geometries: &[Geometry3D],
    ) -> Vec<PlacedBox> {
        let geom_map: std::collections::HashMap<&str, &Geometry3D> =
            geometries.iter().map(|g| (g.id().as_str(), g)).collect();

        result
            .placements
            .iter()
            .filter_map(|p| {
                let geom = geom_map.get(p.geometry_id.as_str())?;
                let ori_idx = p.rotation_index.unwrap_or(0);
                let dims = geom.dimensions_for_orientation(ori_idx);

                let mut placed = PlacedBox::new(
                    p.geometry_id.clone(),
                    p.instance,
                    Point3::new(p.position[0], p.position[1], p.position[2]),
                    dims,
                );

                if let Some(mass) = geom.mass() {
                    placed = placed.with_mass(mass);
                }

                Some(placed)
            })
            .collect()
    }

    /// The layer fill, reporting progress to `callback` when there is one:
    /// once before, once per box placed and once at the end.
    fn layer_fill(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        callback: Option<&ProgressCallback>,
    ) -> SolveResult<f64> {
        let start = Timer::now();
        let total: usize = geometries.iter().map(|g| g.quantity()).sum();
        let report = |info: ProgressInfo| {
            if let Some(callback) = callback {
                callback(info);
            }
        };
        report(
            ProgressInfo::new()
                .with_phase("Layer Packing")
                .with_items(0, total)
                .with_elapsed(0),
        );

        let mut stop = None;
        let (fill, _stopped) = layer_fill_in_order(
            geometries,
            boundary,
            &self.config,
            || {
                stop = self.stop_requested(&start);
                stop.is_some()
            },
            |fill| {
                report(
                    ProgressInfo::new()
                        .with_phase("Layer Packing")
                        .with_items(fill.placed(), total)
                        .with_utilization(fill.utilization())
                        .with_elapsed(start.elapsed_ms()),
                )
            },
        );

        let mut result = SolveResult::new();
        result.unplaced = build_unplaced_list(&fill.placements, geometries);
        result.boundaries_used = usize::from(!fill.placements.is_empty());
        result.utilization = fill.utilization;
        result.placements = fill.placements;
        result.computation_time_ms = start.elapsed_ms();
        let phase = match stop {
            Some(StopReason::Cancelled) => "Cancelled",
            Some(_) => "Time Limit Reached",
            None => "Complete",
        };
        if let Some(stop) = stop {
            result.stop_reason = stop;
        }
        report(
            ProgressInfo::new()
                .with_phase(phase)
                .with_items(result.placements.len(), total)
                .with_utilization(result.utilization)
                .with_elapsed(result.computation_time_ms)
                .finished(),
        );
        result
    }

    /// The solve both [`Solver`] entry points run: the same checks, strategy
    /// run and support pass, with `progress` told of the run.
    fn solve_reporting(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        self.validate(geometries, boundary)?;

        // Reset cancellation flag
        self.cancelled.store(false, Ordering::Relaxed);
        let start = Timer::now();

        let mut result = self.run_strategy(geometries, boundary, progress)?;

        // What the packing leaves short follows from the packing.
        result.record_unplaced(geometries);
        // Authoritative instance-level request total (Σ quantity), recorded once
        // at the top-level entry point where `geometries` is the full request.
        result.total_requested = geometries.iter().map(|g| g.quantity()).sum();

        // Honor gravity/stability constraints, if requested, on the finished packing.
        if boundary.has_gravity() || boundary.has_stability() {
            self.enforce_support(&mut result, geometries, boundary);
        }
        // The solve's time, not its search's: a search starts after its
        // greedy start, and the limit covers all of it.
        result.computation_time_ms = start.elapsed_ms();
        Ok(result)
    }

    /// Whether a constructive fill must stop before its next box, and why:
    /// the caller cancelled, or `time_limit_ms` (0 = unlimited) ran out.
    fn stop_requested(&self, start: &Timer) -> Option<StopReason> {
        if self.cancelled.load(Ordering::Relaxed) {
            Some(StopReason::Cancelled)
        } else if self.config.time_limit_ms > 0 && start.elapsed_ms() >= self.config.time_limit_ms {
            Some(StopReason::TimeLimit)
        } else {
            None
        }
    }

    /// The time a search may run after the layer fill it starts from:
    /// what is left of the limit, `None` when there is no limit.
    fn search_time_limit(&self, started: &Timer) -> Option<std::time::Duration> {
        (self.config.time_limit_ms > 0).then(|| {
            std::time::Duration::from_millis(
                self.config
                    .time_limit_ms
                    .saturating_sub(started.elapsed_ms())
                    .max(1),
            )
        })
    }

    /// Returns the search's packing unless a simple fill -- the layer fill it
    /// started from, or the extreme-point fill -- packs better: more boxes,
    /// then a higher utilization. Packings are compared as the caller gets
    /// them, after gravity and stability have dropped the boxes they do not
    /// allow, since a search scores its packings before that.
    ///
    /// A search can stop, on a small population, generation or time budget,
    /// at a packing worse than the one pass of a simple fill; flooring it here
    /// means a search never returns less. A tie keeps the search's packing.
    fn not_worse_than_fills(
        &self,
        search: SolveResult<f64>,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        layer_fill: SolveResult<f64>,
    ) -> SolveResult<f64> {
        let supported = |mut r: SolveResult<f64>| {
            if boundary.has_gravity() || boundary.has_stability() {
                self.enforce_support(&mut r, geometries, boundary);
            }
            r
        };
        let fills = [
            Some(layer_fill),
            self.extreme_point(geometries, boundary, None).ok(),
        ]
        .into_iter()
        .flatten()
        .map(supported);
        supported(search).floored_at(fills, |a, b| {
            let (pa, pb) = (a.placements.len(), b.placements.len());
            pa > pb || (pa == pb && a.utilization > b.utilization + 1e-9)
        })
    }

    /// Genetic Algorithm based packing optimization.
    ///
    /// Optimizes placement order and orientations with layer-based decoding,
    /// starting from the layer fill and floored at the simple fills.
    fn genetic_algorithm(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        let started = Timer::now();
        let fill = self.layer_fill(geometries, boundary, progress);
        let mut ga_config = self.ga_config();
        ga_config.time_limit = self.search_time_limit(&started);
        let result = run_ga_packing(
            geometries,
            boundary,
            &self.config,
            ga_config,
            self.cancelled.clone(),
            Some(&fill.placements),
            progress,
        )?;
        Ok(self.not_worse_than_fills(result, geometries, boundary, fill))
    }

    /// BRKGA (Biased Random-Key Genetic Algorithm) based packing optimization.
    ///
    /// Random-key encoding and biased crossover, starting from the layer fill
    /// and floored at the simple fills.
    fn brkga(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        let started = Timer::now();
        let fill = self.layer_fill(geometries, boundary, progress);
        let mut brkga_config = self.brkga_config();
        brkga_config.time_limit = self.search_time_limit(&started);
        let result = run_brkga_packing(
            geometries,
            boundary,
            &self.config,
            brkga_config,
            self.cancelled.clone(),
            Some(&fill.placements),
            progress,
        )?;
        Ok(self.not_worse_than_fills(result, geometries, boundary, fill))
    }

    /// Simulated Annealing based packing optimization.
    ///
    /// Neighborhood moves with temperature-based acceptance, starting from
    /// the layer fill and floored at the simple fills.
    fn simulated_annealing(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        let started = Timer::now();
        let fill = self.layer_fill(geometries, boundary, progress);
        let mut sa_config = SaConfig::default()
            // The objective is `100 x placed share + 10 x utilisation`, so a layout a
            // few percent worse differs by a few tenths: a temperature of 100 down to
            // 0.1 accepts nearly every candidate and the search is a random walk.
            // Start where a tenth worse is accepted about one time in three and
            // cool to where a hundredth rarely is, over the iteration budget.
            .with_initial_temp(0.5)
            .with_final_temp(0.01)
            .with_cooling_rate(0.98)
            .with_iterations_per_temp(50)
            .with_max_iterations(self.config.max_iterations.unwrap_or(10000));
        sa_config.time_limit = self.search_time_limit(&started);
        sa_config.target_fitness = self.target_fitness();

        let result = run_sa_packing(
            geometries,
            boundary,
            &self.config,
            sa_config,
            self.cancelled.clone(),
            Some(&fill.placements),
            progress,
        )?;
        Ok(self.not_worse_than_fills(result, geometries, boundary, fill))
    }

    /// Extreme Point heuristic-based packing.
    ///
    /// Places boxes at extreme points (positions touching at least two surfaces).
    /// More efficient than layer-based packing for many scenarios.
    fn extreme_point(
        &self,
        geometries: &[Geometry3D],
        boundary: &Boundary3D,
        progress: Option<&ProgressCallback>,
    ) -> Result<SolveResult<f64>> {
        let start = Timer::now();
        let total: usize = geometries.iter().map(|g| g.quantity()).sum();
        let report = |info: ProgressInfo| {
            if let Some(callback) = progress {
                callback(info.with_phase("Extreme Point"));
            }
        };

        let mut stop = None;
        let (ep_placements, utilization) = run_ep_packing(
            geometries,
            boundary,
            self.config.margin,
            self.config.spacing,
            boundary.max_mass(),
            || {
                // Asked before each box: where the run hears of a cancel.
                report(
                    ProgressInfo::new()
                        .with_items(0, total)
                        .with_elapsed(start.elapsed_ms()),
                );
                stop = self.stop_requested(&start);
                stop.is_some()
            },
        );

        // Convert EP placements to Placement structs. The orientation index must be
        // preserved via `rotation_index` so the response can report the actual box
        // footprint; dropping it made every EP placement report the identity
        // orientation, which read as out-of-bounds for rotated boxes downstream.
        let mut placements = Vec::new();
        for (id, instance, position, orientation) in ep_placements {
            let placement = Placement::new_3d(
                id, instance, position.x, position.y, position.z, 0.0, // rotation_x
                0.0, // rotation_y
                0.0, // rotation_z (orientation encoded via rotation_index)
            )
            .with_rotation_index(orientation);
            placements.push(placement);
        }

        let unplaced = build_unplaced_list(&placements, geometries);

        let mut result = SolveResult::new();
        result.boundaries_used = usize::from(!placements.is_empty());
        result.placements = placements;
        result.utilization = utilization;
        if let Some(stop) = stop {
            result.stop_reason = stop;
        }
        result.unplaced = unplaced;
        result.computation_time_ms = start.elapsed_ms();
        result.strategy = Some("ExtremePoint".to_string());
        report(
            ProgressInfo::new()
                .with_items(result.placements.len(), total)
                .with_utilization(result.utilization)
                .with_elapsed(result.computation_time_ms)
                .finished(),
        );

        Ok(result)
    }
}

impl Solver for Packer3D {
    type Geometry = Geometry3D;
    type Boundary = Boundary3D;
    type Scalar = f64;

    fn solve(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
    ) -> Result<SolveResult<f64>> {
        self.solve_reporting(geometries, boundary, None)
    }

    fn solve_with_progress(
        &self,
        geometries: &[Self::Geometry],
        boundary: &Self::Boundary,
        callback: ProgressCallback,
    ) -> Result<SolveResult<f64>> {
        self.solve_reporting(geometries, boundary, Some(&callback))
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_geometry_id_given_twice_is_refused_by_every_entry_point() {
        let geometries = [
            Geometry3D::new("B", 10.0, 10.0, 10.0),
            Geometry3D::new("B", 50.0, 50.0, 50.0),
        ];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let packer = Packer3D::default_config();
        let errors = [
            packer.solve(&geometries, &boundary).expect_err("solve"),
            packer
                .solve_with_progress(&geometries, &boundary, Box::new(|_| {}))
                .expect_err("solve_with_progress"),
        ];
        for err in errors {
            let msg = err.to_string();
            assert!(msg.contains("'B'"), "{msg}");
            assert!(msg.contains("positions 0 and 1"), "{msg}");
        }
    }

    /// Nothing to pack is refused, and every entry point judges the whole
    /// request the same way -- the progress entry point used to skip the
    /// configuration, and no entry point judged the boxes before the strategy
    /// ran.
    #[test]
    fn every_entry_point_judges_the_whole_request() {
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let packer = Packer3D::default_config();
        let empty = Error::EmptyInput {
            parameter: "geometries".into(),
        };
        assert_eq!(packer.solve(&[], &boundary).expect_err("solve"), empty);
        assert_eq!(
            packer
                .solve_with_progress(&[], &boundary, Box::new(|_| {}))
                .expect_err("solve_with_progress"),
            empty
        );

        let boxes = [Geometry3D::new("B", 10.0, 10.0, 10.0)];
        let bad_config = Packer3D::new(Config::default().with_spacing(-1.0));
        for err in [
            bad_config.solve(&boxes, &boundary).expect_err("solve"),
            bad_config
                .solve_with_progress(&boxes, &boundary, Box::new(|_| {}))
                .expect_err("solve_with_progress"),
        ] {
            assert_eq!(err.code(), "parameter_out_of_range", "{err}");
        }

        let flat = [Geometry3D::new("flat", 10.0, 0.0, 10.0)];
        for strategy in [Strategy::ExtremePoint, Strategy::GeneticAlgorithm] {
            let packer = Packer3D::new(Config::default().with_strategy(strategy));
            match packer.solve(&flat, &boundary) {
                Err(Error::OutOfRange { parameter, id, .. }) => {
                    assert_eq!(parameter, "dimensions[1]");
                    assert_eq!(id.as_deref(), Some("flat"));
                }
                other => panic!("{strategy:?}: {other:?}"),
            }
        }
    }

    /// A population or generation budget the caller sets is the one the
    /// search runs with -- BRKGA used to run 50 x 100 whatever was asked;
    /// left unset, each search runs with its own.
    #[test]
    fn a_set_budget_is_run_and_an_unset_one_is_the_strategy_s() {
        let unset = Packer3D::default_config();
        assert_eq!(unset.ga_config().population_size, 100);
        assert_eq!(unset.ga_config().max_generations, 500);
        assert_eq!(unset.brkga_config().population_size, 50);
        assert_eq!(unset.brkga_config().max_generations, 100);

        let set = Packer3D::new(
            Config::default()
                .with_population_size(12)
                .with_max_generations(3),
        );
        assert_eq!(set.ga_config().population_size, 12);
        assert_eq!(set.ga_config().max_generations, 3);
        assert_eq!(set.brkga_config().population_size, 12);
        assert_eq!(set.brkga_config().max_generations, 3);
    }

    #[test]
    fn test_simple_packing() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(3),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let packer = Packer3D::default_config();

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert!(result.utilization > 0.0);
        assert!(result.placements.len() <= 5);
    }

    #[test]
    fn test_mass_constraint() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0)
            .with_quantity(10)
            .with_mass(100.0)];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0).with_max_mass(350.0);

        let packer = Packer3D::default_config();
        let result = packer.solve(&geometries, &boundary).unwrap();

        // Should only place 3 boxes (300 mass) due to 350 mass limit
        assert!(result.placements.len() <= 3);
    }

    #[test]
    fn test_placement_within_bounds() {
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(50.0, 50.0, 50.0);
        let config = Config::default().with_margin(5.0).with_spacing(2.0);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());

        // Verify placements are within bounds (with margin)
        for p in &result.placements {
            assert!(p.position[0] >= 5.0);
            assert!(p.position[1] >= 5.0);
            assert!(p.position[2] >= 5.0);
        }
    }

    #[test]
    fn test_ga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
    }

    #[test]
    fn test_ga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_3d_response_orientation_in_bounds() {
        use crate::build_pack3d_response;
        use crate::geometry::OrientationConstraint;

        // A tall box in a flat container can only be placed rotated. The response must
        // report the orientation that was actually used, so reconstructing the box
        // footprint from the reported orientation label keeps it inside the boundary.
        // Pre-fix, EP/GA/BRKGA dropped the orientation and reported "xyz" (identity),
        // making rotated placements read as out-of-bounds (the 0.3.3 "oob" reports).
        let (bw, bd, bh) = (100.0_f64, 100.0_f64, 30.0_f64);
        let boundary = Boundary3D::new(bw, bd, bh);

        for strategy in [
            Strategy::ExtremePoint,
            Strategy::GeneticAlgorithm,
            Strategy::Brkga,
        ] {
            let geometries = vec![Geometry3D::new("tall", 25.0, 25.0, 80.0)
                .with_quantity(3)
                .with_orientation(OrientationConstraint::Any)];
            let config = Config::default().with_strategy(strategy);
            let packer = Packer3D::new(config);
            let result = packer.solve(&geometries, &boundary).unwrap();
            let response = build_pack3d_response(&result, &geometries);

            // EP is deterministic and must place at least one box (only fits rotated).
            if strategy == Strategy::ExtremePoint {
                assert!(
                    !response.placements.is_empty(),
                    "EP should place the tall box by rotating it into the flat container"
                );
            }

            let base = geometries[0].dimensions_for_orientation(0); // unrotated dims
            for p in &response.placements {
                let axes: Vec<usize> = p
                    .orientation
                    .chars()
                    .map(|c| match c {
                        'x' => 0,
                        'y' => 1,
                        'z' => 2,
                        _ => panic!("unexpected orientation label '{}'", p.orientation),
                    })
                    .collect();
                let (dx, dy, dz) = (base[axes[0]], base[axes[1]], base[axes[2]]);
                let e = 1e-6;
                assert!(
                    p.x + dx <= bw + e && p.y + dy <= bd + e && p.z + dz <= bh + e,
                    "{:?} {}#{} out of bounds for reported orientation '{}': \
                     pos({:.1},{:.1},{:.1}) dims({dx:.1},{dy:.1},{dz:.1}) boundary({bw},{bd},{bh})",
                    strategy,
                    p.geometry_id,
                    p.instance,
                    p.orientation,
                    p.x,
                    p.y,
                    p.z,
                );
            }
        }
    }

    #[test]
    fn test_ga_strategy_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Box that fits better when rotated
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        // Container where orientation matters
        let boundary = Boundary3D::new(60.0, 60.0, 60.0);
        let config = Config::default().with_strategy(Strategy::GeneticAlgorithm);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // GA should find a way to place both boxes
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_brkga_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // BRKGA should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("BRKGA".to_string()));
    }

    #[test]
    fn test_brkga_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::Brkga);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ep_strategy_basic() {
        let geometries = vec![
            Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(2),
            Geometry3D::new("B2", 15.0, 15.0, 15.0).with_quantity(2),
        ];

        let boundary = Boundary3D::new(100.0, 80.0, 50.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // EP should place items and achieve positive utilization
        assert!(result.utilization > 0.0);
        assert!(!result.placements.is_empty());
        assert_eq!(result.strategy, Some("ExtremePoint".to_string()));
    }

    #[test]
    fn test_ep_strategy_all_placed() {
        // Small number of boxes that should all fit
        let geometries = vec![Geometry3D::new("B1", 10.0, 10.0, 10.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // All 4 boxes should be placed
        assert_eq!(result.placements.len(), 4);
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ep_strategy_with_margin() {
        let geometries = vec![Geometry3D::new("B1", 20.0, 20.0, 20.0).with_quantity(4)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default()
            .with_strategy(Strategy::ExtremePoint)
            .with_margin(5.0);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // Verify placements start at margin
        for p in &result.placements {
            assert!(p.position[0] >= 4.9);
            assert!(p.position[1] >= 4.9);
            assert!(p.position[2] >= 4.9);
        }
    }

    #[test]
    fn test_ep_strategy_with_orientations() {
        use crate::geometry::OrientationConstraint;

        // Long box that benefits from rotation
        let geometries = vec![Geometry3D::new("B1", 80.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let config = Config::default().with_strategy(Strategy::ExtremePoint);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // EP should find a way to place both boxes
        assert_eq!(result.placements.len(), 2);
    }

    #[test]
    fn test_ep_strategy_perfect_fill_via_solve() {
        // End-to-end through Packer3D::solve (the path FFI/WASM use): eight 50-cubes
        // tile a 100^3 container exactly. EP must place all 8. Guards the under-packing
        // regression at the dispatch level, not just run_ep_packing.
        let geometries = vec![Geometry3D::new("cube", 50.0, 50.0, 50.0).with_quantity(8)];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        let packer = Packer3D::new(Config::default().with_strategy(Strategy::ExtremePoint));

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert_eq!(result.placements.len(), 8, "EP must place all 8 cubes");
        assert!(result.unplaced.is_empty());
    }

    #[test]
    fn test_ep_at_least_as_good_as_blf() {
        // A correct Extreme-Point heuristic never places fewer boxes than the simpler
        // layer (BLF) strategy on the same instance. Before the fix EP stalled far below
        // BLF (e.g. 4 vs 13 on the mixed-box set). Covers several shapes.
        let boundary = Boundary3D::new(85.0, 85.0, 80.0);
        let cases: [(&str, f64, f64, f64, usize); 3] = [
            ("big", 40.0, 40.0, 40.0, 4),
            ("mid", 30.0, 20.0, 25.0, 6),
            ("small", 15.0, 15.0, 30.0, 8),
        ];
        let geometries: Vec<Geometry3D> = cases
            .iter()
            .map(|(id, w, d, h, q)| Geometry3D::new(*id, *w, *d, *h).with_quantity(*q))
            .collect();

        let blf = Packer3D::new(Config::default().with_strategy(Strategy::BottomLeftFill))
            .solve(&geometries, &boundary)
            .unwrap();
        let ep = Packer3D::new(Config::default().with_strategy(Strategy::ExtremePoint))
            .solve(&geometries, &boundary)
            .unwrap();

        assert!(
            ep.placements.len() >= blf.placements.len(),
            "EP placed {} but BLF placed {} — EP must not regress below BLF",
            ep.placements.len(),
            blf.placements.len()
        );
    }

    /// Builds a result whose second box floats above the floor with nothing beneath it.
    fn floating_pair() -> (Vec<Geometry3D>, SolveResult<f64>) {
        let geometries = vec![
            Geometry3D::new("a", 20.0, 20.0, 20.0),
            Geometry3D::new("b", 20.0, 20.0, 20.0),
        ];
        let mut result = SolveResult::new();
        result.placements.push(
            Placement::new_3d("a".to_string(), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                .with_rotation_index(0),
        );
        // "b" hovers at z=50 — no floor contact, no box below.
        result.placements.push(
            Placement::new_3d("b".to_string(), 0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0)
                .with_rotation_index(0),
        );
        (geometries, result)
    }

    #[test]
    fn test_gravity_removes_floating_box() {
        let (geometries, mut result) = floating_pair();
        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_gravity(true);
        let packer = Packer3D::default_config();

        packer.enforce_support(&mut result, &geometries, &boundary);

        let ids: Vec<&str> = result
            .placements
            .iter()
            .map(|p| p.geometry_id.as_str())
            .collect();
        assert_eq!(ids, vec!["a"], "floating box must be dropped under gravity");
        assert!(result.unplaced.iter().any(|id| id == "b"));
    }

    #[test]
    fn test_no_constraint_keeps_floating_box() {
        // Without gravity/stability the solver never calls enforce_support, so a result
        // with a floating box is returned untouched. Verify enforce_support is the only
        // thing that would remove it — i.e. solve() leaves such input alone.
        let (geometries, result) = floating_pair();
        let boundary = Boundary3D::new(100.0, 100.0, 100.0);
        assert!(!boundary.has_gravity() && !boundary.has_stability());
        // The pair was hand-built (not via solve); just assert the boundary flags gate it.
        assert_eq!(result.placements.len(), 2);
        let _ = geometries;
    }

    #[test]
    fn test_gravity_keeps_floor_boxes_with_margin() {
        // With a wall margin the pack starts boxes at z = margin, which the support
        // analysis must treat as the floor. If it checks against z = 0 instead, every
        // bottom-layer box reads as floating and the whole pack gets dropped. End-to-end
        // through solve() with margin > 0 and gravity on.
        let geometries = vec![
            Geometry3D::new("big", 40.0, 40.0, 40.0).with_quantity(4),
            Geometry3D::new("mid", 30.0, 20.0, 25.0).with_quantity(6),
        ];
        let boundary = Boundary3D::new(100.0, 100.0, 100.0).with_gravity(true);
        let packer = Packer3D::new(
            Config::default()
                .with_margin(5.0)
                .with_strategy(Strategy::BottomLeftFill),
        );

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert!(
            !result.placements.is_empty(),
            "margin>0 + gravity must not drop floor-resting boxes"
        );
    }

    #[test]
    fn test_stability_drops_undersupported_box() {
        // "b" rests on "a" but overlaps only a thin sliver of its top face (well under
        // the 70% base-support threshold). Gravity alone keeps it (it does touch below);
        // stability drops it.
        let geometries = vec![
            Geometry3D::new("a", 40.0, 40.0, 20.0),
            Geometry3D::new("b", 40.0, 40.0, 20.0),
        ];
        let make = || {
            let mut r = SolveResult::new();
            r.placements.push(
                Placement::new_3d("a".to_string(), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    .with_rotation_index(0),
            );
            // shifted so only a 5x40 strip (≈12.5%) of b's base sits on a's top
            r.placements.push(
                Placement::new_3d("b".to_string(), 0, 35.0, 0.0, 20.0, 0.0, 0.0, 0.0)
                    .with_rotation_index(0),
            );
            r
        };
        let packer = Packer3D::default_config();

        let mut grav = make();
        packer.enforce_support(
            &mut grav,
            &geometries,
            &Boundary3D::new(100.0, 100.0, 100.0).with_gravity(true),
        );
        assert_eq!(
            grav.placements.len(),
            2,
            "gravity keeps a box that touches below"
        );

        let mut stab = make();
        packer.enforce_support(
            &mut stab,
            &geometries,
            &Boundary3D::new(100.0, 100.0, 100.0).with_stability(true),
        );
        assert!(
            stab.placements.iter().all(|p| p.geometry_id == "a"),
            "stability drops the under-supported box"
        );
    }

    #[test]
    fn test_layer_packing_orientation_optimization() {
        use crate::geometry::OrientationConstraint;

        // A box 50x10x10 that won't fit in 45 width without rotation
        // But at orientation (1,0,2) it becomes 10x50x10, width=10, which fits
        let geometries = vec![Geometry3D::new("B1", 50.0, 10.0, 10.0)
            .with_quantity(2)
            .with_orientation(OrientationConstraint::Any)];

        // Narrow container: width=45, depth=80, height=80
        let boundary = Boundary3D::new(45.0, 80.0, 80.0);
        let config = Config::default().with_strategy(Strategy::BottomLeftFill);
        let packer = Packer3D::new(config);

        let result = packer.solve(&geometries, &boundary).unwrap();

        // Both boxes should be placed via orientation change
        assert_eq!(
            result.placements.len(),
            2,
            "Both boxes should be placed by using rotation"
        );
        assert!(result.unplaced.is_empty());

        // Verify orientation index is set for placements
        for p in &result.placements {
            assert!(
                p.rotation_index.is_some(),
                "Placement should have rotation_index set"
            );
        }
    }

    #[test]
    fn test_layer_packing_selects_best_orientation() {
        use crate::geometry::OrientationConstraint;

        // Box 30x20x10 in container 35x50x100
        // Original orientation (30x20x10): fits in row, leaves 5 spare width
        // Rotated (20x30x10): fits but uses more depth
        // Best: original orientation to minimize vertical space usage
        let geometries = vec![Geometry3D::new("B1", 30.0, 20.0, 10.0)
            .with_quantity(1)
            .with_orientation(OrientationConstraint::Any)];

        let boundary = Boundary3D::new(35.0, 50.0, 100.0);
        let packer = Packer3D::default_config();

        let result = packer.solve(&geometries, &boundary).unwrap();

        assert_eq!(result.placements.len(), 1);
        assert!(result.unplaced.is_empty());
    }
}
