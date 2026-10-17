//! Benchmark runner for ESICUP datasets.

use crate::dataset::{Dataset, ExpandedItem, Shape};
use crate::result::{BenchmarkResult, RunResult};
use instant::Instant;
use u_nesting_core::{Config, Strategy};
use u_nesting_d2::check::{check_layout, LayoutRules};
use u_nesting_d2::{Boundary2D, Geometry2D, Nester2D};

/// Configuration for benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Strategies to benchmark.
    pub strategies: Vec<Strategy>,
    /// Time limit per run in milliseconds.
    pub time_limit_ms: u64,
    /// Number of runs per configuration (for averaging).
    pub runs_per_config: usize,
    /// Whether to show progress.
    pub show_progress: bool,
    /// GA/BRKGA population size; `None` leaves it to each strategy.
    pub population_size: Option<usize>,
    /// GA/BRKGA generation budget; `None` leaves it to each strategy.
    pub max_generations: Option<u32>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                Strategy::BottomLeftFill,
                Strategy::NfpGuided,
                Strategy::GeneticAlgorithm,
            ],
            time_limit_ms: 60_000, // 60 seconds
            runs_per_config: 1,
            show_progress: true,
            population_size: None,
            max_generations: None,
        }
    }
}

impl BenchmarkConfig {
    /// Creates a new benchmark configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the strategies to benchmark.
    pub fn with_strategies(mut self, strategies: Vec<Strategy>) -> Self {
        self.strategies = strategies;
        self
    }

    /// Sets the time limit per run.
    pub fn with_time_limit(mut self, ms: u64) -> Self {
        self.time_limit_ms = ms;
        self
    }

    /// Sets the number of runs per configuration.
    pub fn with_runs_per_config(mut self, n: usize) -> Self {
        self.runs_per_config = n;
        self
    }

    /// Quick preset for fast benchmarking.
    pub fn quick() -> Self {
        Self {
            strategies: vec![Strategy::BottomLeftFill, Strategy::NfpGuided],
            time_limit_ms: 5_000,
            runs_per_config: 1,
            show_progress: true,
            population_size: None,
            max_generations: None,
        }
    }

    /// Standard preset for thorough benchmarking.
    pub fn standard() -> Self {
        Self {
            strategies: vec![
                Strategy::BottomLeftFill,
                Strategy::NfpGuided,
                Strategy::GeneticAlgorithm,
                Strategy::Brkga,
                Strategy::SimulatedAnnealing,
            ],
            time_limit_ms: 60_000,
            runs_per_config: 3,
            show_progress: true,
            population_size: None,
            max_generations: None,
        }
    }
}

/// Benchmark runner.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Creates a new benchmark runner.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Runs benchmarks on a single dataset.
    pub fn run_dataset(&self, dataset: &Dataset) -> BenchmarkResult {
        let mut results = BenchmarkResult::new();

        if self.config.show_progress {
            println!("\nBenchmarking dataset: {}", dataset.name);
            println!("  Items: {}", dataset.items.len());
            println!("  Total pieces: {}", dataset.expand_items().len());
            println!("  Strip height: {}", dataset.strip_height);
        }

        // Convert dataset to geometries
        let expanded = dataset.expand_items();
        let geometries = self.convert_to_geometries(&expanded);

        // Use provided strip_width or estimate if not given
        let initial_width = dataset
            .strip_width
            .unwrap_or_else(|| self.estimate_initial_width(&geometries, dataset.strip_height));
        let boundary = Boundary2D::rectangle(initial_width, dataset.strip_height);

        for strategy in &self.config.strategies {
            if self.config.show_progress {
                println!("  Running {:?}...", strategy);
            }

            for run_idx in 0..self.config.runs_per_config {
                let solver_config = Config::new()
                    .with_strategy(*strategy)
                    .with_time_limit(self.config.time_limit_ms)
                    .with_spacing(0.0)
                    .with_margin(0.0);

                // Override GA parameters
                let mut solver_config = solver_config;
                solver_config.population_size = self.config.population_size;
                solver_config.max_generations = self.config.max_generations;

                let nester = Nester2D::new(solver_config);

                let start = Instant::now();
                // Use multi-strip solve to automatically handle overflow
                let result = nester.solve_multi_strip(&geometries, &boundary);
                let elapsed = start.elapsed().as_millis() as u64;

                match result {
                    Ok(solve_result) => {
                        // The layout is judged by the public checker before it is
                        // measured: a length means nothing for a layout that
                        // overlaps, leaves its sheet or drops a piece.
                        // The solve lays sheets side by side; the checker reads
                        // each sheet in its own frame.
                        let mut local = solve_result.clone();
                        local.to_boundary_local(initial_width);
                        let found = check_layout(
                            &geometries,
                            &boundary,
                            &LayoutRules {
                                spacing: 0.0,
                                margin: 0.0,
                                multi_sheet: true,
                            },
                            &local.placements,
                            &local.unplaced,
                        );
                        let violations = found.len();
                        if self.config.show_progress {
                            for v in found.iter().take(3) {
                                println!("    violation: {v:?}");
                            }
                        }
                        let strip_length = self.calculate_strip_length(&solve_result, &geometries);
                        // ESICUP density: piece area over the strip it occupies.
                        let density = if strip_length > 0.0 && dataset.strip_height > 0.0 {
                            dataset.total_area() / (strip_length * dataset.strip_height)
                        } else {
                            0.0
                        };

                        // Convert placements to PlacementInfo
                        // Note: Do NOT normalize positions - the algorithms return positions
                        // where the geometry's origin should be placed. For rotated geometries,
                        // the origin position accounts for the AABB offset so that all vertices
                        // stay within bounds. Normalizing would break this.
                        let placement_infos: Vec<crate::result::PlacementInfo> = solve_result
                            .placements
                            .iter()
                            .map(|p| crate::result::PlacementInfo {
                                geometry_id: p.geometry_id.clone(),
                                position: [p.position[0], p.position[1]],
                                rotation: p.rotation.first().copied().unwrap_or(0.0),
                                strip_index: p.boundary_index,
                            })
                            .collect();

                        let mut run_result = RunResult::new(
                            dataset.name.clone(),
                            format!("run_{}", run_idx + 1),
                            *strategy,
                            strip_length,
                            dataset.strip_height,
                            solve_result.placements.len(),
                            geometries.len(),
                            elapsed,
                        )
                        .with_utilization(density)
                        .with_violations(violations)
                        .with_placements(placement_infos);

                        // A gap is the distance to the literature for a valid,
                        // complete layout on one strip only: a second strip
                        // means the estimated width was too short to measure a
                        // length by.
                        if let (Some(best), 0, true, 1) = (
                            dataset.best_known,
                            violations,
                            solve_result.unplaced.is_empty(),
                            solve_result.boundaries_used,
                        ) {
                            run_result = run_result.with_best_known(best);
                        }

                        if self.config.show_progress {
                            println!(
                                "    Run {}: length={:.2}, placed={}/{}, strips={}, violations={}, time={}ms",
                                run_idx + 1,
                                strip_length,
                                solve_result.placements.len(),
                                geometries.len(),
                                solve_result.boundaries_used,
                                violations,
                                elapsed
                            );
                        }

                        results.add_run(run_result);
                    }
                    Err(e) => {
                        if self.config.show_progress {
                            println!("    Run {} failed: {}", run_idx + 1, e);
                        }
                    }
                }
            }
        }

        results
    }

    /// Runs benchmarks on multiple datasets.
    pub fn run_datasets(&self, datasets: &[Dataset]) -> BenchmarkResult {
        let mut combined = BenchmarkResult::new();

        for dataset in datasets {
            let result = self.run_dataset(dataset);
            for run in result.runs {
                combined.add_run(run);
            }
        }

        combined
    }

    /// Converts expanded items to Geometry2D.
    fn convert_to_geometries(&self, items: &[ExpandedItem]) -> Vec<Geometry2D> {
        items
            .iter()
            .map(|item| {
                let geom_id = format!("piece_{}", item.piece_id);
                let geom = match &item.shape {
                    Shape::SimplePolygon(points) => {
                        let vertices: Vec<(f64, f64)> =
                            points.iter().map(|[x, y]| (*x, *y)).collect();
                        Geometry2D::new(geom_id).with_polygon(vertices)
                    }
                    Shape::PolygonWithHoles { outer, holes } => {
                        let exterior: Vec<(f64, f64)> =
                            outer.iter().map(|[x, y]| (*x, *y)).collect();
                        let mut geom = Geometry2D::new(geom_id).with_polygon(exterior);
                        for hole in holes {
                            let hole_verts: Vec<(f64, f64)> =
                                hole.iter().map(|[x, y]| (*x, *y)).collect();
                            geom = geom.with_hole(hole_verts);
                        }
                        geom
                    }
                    Shape::MultiPolygon(polygons) => {
                        // For multi-polygon, just use the first polygon
                        // (proper handling would require multiple geometries)
                        if let Some(first) = polygons.first() {
                            let vertices: Vec<(f64, f64)> =
                                first.iter().map(|[x, y]| (*x, *y)).collect();
                            Geometry2D::new(geom_id).with_polygon(vertices)
                        } else {
                            Geometry2D::new(geom_id)
                        }
                    }
                };

                // Set allowed rotations
                if !item.allowed_orientations.is_empty() {
                    geom.with_rotations_deg(item.allowed_orientations.clone())
                } else {
                    geom
                }
            })
            .collect()
    }

    /// Estimates initial width based on total area.
    fn estimate_initial_width(&self, geometries: &[Geometry2D], strip_height: f64) -> f64 {
        use u_nesting_core::geometry::Geometry;

        let total_area: f64 = geometries.iter().map(|g| g.measure()).sum();
        // Use 2x the theoretical minimum as initial width
        let theoretical_min = total_area / strip_height;
        theoretical_min * 3.0
    }

    /// The strip length: the furthest x any vertex of any placed piece reaches
    /// (a strip starts at x = 0), from the pieces' real outlines at their
    /// placed rotation. Sheets are local, so with more than one sheet this is
    /// the longest of them.
    fn calculate_strip_length(
        &self,
        result: &u_nesting_core::SolveResult<f64>,
        geometries: &[Geometry2D],
    ) -> f64 {
        use std::collections::HashMap;
        use u_nesting_core::geometry::Geometry;

        let by_id: HashMap<&str, &Geometry2D> =
            geometries.iter().map(|g| (g.id().as_str(), g)).collect();

        result
            .placements
            .iter()
            .filter_map(|p| {
                let geom = by_id.get(p.geometry_id.as_str())?;
                let rotation = p.rotation.first().copied().unwrap_or(0.0);
                geom.transformed_exterior(p.position[0], p.position[1], rotation)
                    .into_iter()
                    .map(|(x, _)| x)
                    .reduce(f64::max)
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::DatasetParser;

    #[test]
    fn test_convert_simple_polygon() {
        let json = r#"{
            "name": "test",
            "items": [
                {
                    "id": 0,
                    "demand": 1,
                    "allowed_orientations": [0.0],
                    "shape": {
                        "type": "simple_polygon",
                        "data": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
                    }
                }
            ],
            "strip_height": 100.0
        }"#;

        let parser = DatasetParser::new();
        let dataset = parser.parse_json(json).unwrap();
        let expanded = dataset.expand_items();

        let config = BenchmarkConfig::quick();
        let runner = BenchmarkRunner::new(config);
        let geometries = runner.convert_to_geometries(&expanded);

        assert_eq!(geometries.len(), 1);
    }
}
