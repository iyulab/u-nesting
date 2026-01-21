//! 3D benchmark runner for bin packing problems.

use crate::dataset3d::{Dataset3D, Item3D, OrientationConstraint as DatasetOrientation};
use crate::result::{BenchmarkResult, RunResult};
use instant::Instant;
use u_nesting_core::{Config, Solver, Strategy};
use u_nesting_d3::geometry::OrientationConstraint;
use u_nesting_d3::{Boundary3D, Geometry3D, Packer3D};

/// Configuration for 3D benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig3D {
    /// Strategies to benchmark.
    pub strategies: Vec<Strategy>,
    /// Time limit per run in milliseconds.
    pub time_limit_ms: u64,
    /// Number of runs per configuration (for averaging).
    pub runs_per_config: usize,
    /// Whether to show progress.
    pub show_progress: bool,
    /// GA population size.
    pub population_size: usize,
    /// GA max generations.
    pub max_generations: u32,
}

impl Default for BenchmarkConfig3D {
    fn default() -> Self {
        Self {
            strategies: vec![
                Strategy::BottomLeftFill,
                Strategy::ExtremePoint,
                Strategy::GeneticAlgorithm,
            ],
            time_limit_ms: 60_000,
            runs_per_config: 1,
            show_progress: true,
            population_size: 100,
            max_generations: 500,
        }
    }
}

impl BenchmarkConfig3D {
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
            strategies: vec![Strategy::BottomLeftFill, Strategy::ExtremePoint],
            time_limit_ms: 5_000,
            runs_per_config: 1,
            show_progress: true,
            population_size: 50,
            max_generations: 100,
        }
    }

    /// Standard preset for thorough benchmarking.
    pub fn standard() -> Self {
        Self {
            strategies: vec![
                Strategy::BottomLeftFill,
                Strategy::ExtremePoint,
                Strategy::GeneticAlgorithm,
                Strategy::Brkga,
                Strategy::SimulatedAnnealing,
            ],
            time_limit_ms: 60_000,
            runs_per_config: 3,
            show_progress: true,
            population_size: 100,
            max_generations: 500,
        }
    }
}

/// 3D benchmark runner.
pub struct BenchmarkRunner3D {
    config: BenchmarkConfig3D,
}

impl BenchmarkRunner3D {
    /// Creates a new benchmark runner.
    pub fn new(config: BenchmarkConfig3D) -> Self {
        Self { config }
    }

    /// Runs benchmarks on a single 3D dataset.
    pub fn run_dataset(&self, dataset: &Dataset3D) -> BenchmarkResult {
        let mut results = BenchmarkResult::new();

        if self.config.show_progress {
            println!("\nBenchmarking 3D dataset: {}", dataset.name);
            println!("  Instance class: {:?}", dataset.instance_class);
            println!("  Items: {}", dataset.items.len());
            println!("  Bin dimensions: {:?}", dataset.bin_dimensions);
            println!("  Total item volume: {:.2}", dataset.total_item_volume());
            println!("  Volume lower bound: {} bins", dataset.volume_lower_bound());
        }

        // Convert dataset to geometries
        let geometries = self.convert_to_geometries(&dataset.items);
        let boundary = Boundary3D::new(
            dataset.bin_dimensions[0],
            dataset.bin_dimensions[1],
            dataset.bin_dimensions[2],
        );

        for strategy in &self.config.strategies {
            if self.config.show_progress {
                println!("  Running {:?}...", strategy);
            }

            for run_idx in 0..self.config.runs_per_config {
                let mut solver_config = Config::new()
                    .with_strategy(*strategy)
                    .with_time_limit(self.config.time_limit_ms)
                    .with_spacing(0.0)
                    .with_margin(0.0);

                solver_config.population_size = self.config.population_size;
                solver_config.max_generations = self.config.max_generations;

                let packer = Packer3D::new(solver_config);

                let start = Instant::now();
                let result = packer.solve(&geometries, &boundary);
                let elapsed = start.elapsed().as_millis() as u64;

                match result {
                    Ok(solve_result) => {
                        let bins_used = solve_result.boundaries_used;
                        let utilization = solve_result.utilization;

                        // For 3D bin packing, use bins_used as the "strip_length" equivalent
                        let mut run_result = RunResult::new(
                            dataset.name.clone(),
                            format!("run_{}", run_idx + 1),
                            *strategy,
                            bins_used as f64, // bins used as primary metric
                            0.0, // no strip_height for 3D
                            solve_result.placements.len(),
                            geometries.len(),
                            elapsed,
                        );

                        // Store actual utilization separately
                        run_result.utilization = utilization;

                        if let Some(best) = dataset.best_known {
                            run_result = run_result.with_best_known(best as f64);
                        }

                        if self.config.show_progress {
                            println!(
                                "    Run {}: bins={}, placed={}/{}, utilization={:.2}%, time={}ms",
                                run_idx + 1,
                                bins_used,
                                solve_result.placements.len(),
                                geometries.len(),
                                utilization * 100.0,
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

    /// Runs benchmarks on multiple 3D datasets.
    pub fn run_datasets(&self, datasets: &[Dataset3D]) -> BenchmarkResult {
        let mut combined = BenchmarkResult::new();

        for dataset in datasets {
            let result = self.run_dataset(dataset);
            for run in result.runs {
                combined.add_run(run);
            }
        }

        combined
    }

    /// Converts Item3D to Geometry3D.
    fn convert_to_geometries(&self, items: &[Item3D]) -> Vec<Geometry3D> {
        let mut geometries = Vec::new();

        for item in items {
            for q in 0..item.quantity {
                let id = if item.quantity > 1 {
                    format!("box_{}_{}", item.id, q)
                } else {
                    format!("box_{}", item.id)
                };

                let mut geom =
                    Geometry3D::new(id, item.width(), item.height(), item.depth());

                // Set orientation constraint
                geom = match item.allowed_orientations {
                    DatasetOrientation::Any => geom.with_orientation(OrientationConstraint::Any),
                    DatasetOrientation::Upright => {
                        geom.with_orientation(OrientationConstraint::Upright)
                    }
                    DatasetOrientation::Fixed => {
                        geom.with_orientation(OrientationConstraint::Fixed)
                    }
                };

                geometries.push(geom);
            }
        }

        geometries
    }
}

/// Summary statistics for 3D benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkSummary3D {
    /// Dataset name
    pub dataset_name: String,
    /// Strategy used
    pub strategy: String,
    /// Average bins used
    pub avg_bins: f64,
    /// Best bins achieved
    pub best_bins: usize,
    /// Average utilization
    pub avg_utilization: f64,
    /// Average computation time (ms)
    pub avg_time_ms: f64,
    /// Number of runs
    pub num_runs: usize,
}

impl BenchmarkResult {
    /// Computes summary statistics for 3D benchmarks.
    pub fn summary_3d(&self) -> Vec<BenchmarkSummary3D> {
        use std::collections::HashMap;

        let mut grouped: HashMap<(String, String), Vec<&RunResult>> = HashMap::new();

        for run in &self.runs {
            grouped
                .entry((run.dataset.clone(), run.strategy.clone()))
                .or_default()
                .push(run);
        }

        let mut summaries: Vec<BenchmarkSummary3D> = grouped
            .into_iter()
            .map(|((name, strategy), runs)| {
                let num_runs = runs.len();
                let avg_bins =
                    runs.iter().map(|r| r.strip_length).sum::<f64>() / num_runs as f64;
                let best_bins = runs
                    .iter()
                    .map(|r| r.strip_length as usize)
                    .min()
                    .unwrap_or(0);
                let avg_utilization =
                    runs.iter().map(|r| r.utilization).sum::<f64>() / num_runs as f64;
                let avg_time_ms =
                    runs.iter().map(|r| r.time_ms as f64).sum::<f64>() / num_runs as f64;

                BenchmarkSummary3D {
                    dataset_name: name,
                    strategy,
                    avg_bins,
                    best_bins,
                    avg_utilization,
                    avg_time_ms,
                    num_runs,
                }
            })
            .collect();

        summaries.sort_by(|a, b| {
            a.dataset_name
                .cmp(&b.dataset_name)
                .then(a.strategy.cmp(&b.strategy))
        });

        summaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset3d::{InstanceClass, InstanceGenerator};

    #[test]
    fn test_run_small_dataset() {
        let gen = InstanceGenerator::new(100.0).with_seed(42);
        let dataset = gen.generate(InstanceClass::MPV5, 5);

        let config = BenchmarkConfig3D::quick()
            .with_strategies(vec![Strategy::BottomLeftFill])
            .with_runs_per_config(1);

        let mut runner = BenchmarkRunner3D::new(config);
        runner.config.show_progress = false;

        let result = runner.run_dataset(&dataset);

        assert!(!result.runs.is_empty());
        assert!(result.runs[0].pieces_placed > 0);
    }

    #[test]
    fn test_convert_geometries() {
        let items = vec![
            Item3D::new(0, 10.0, 20.0, 30.0),
            Item3D::new(1, 15.0, 25.0, 35.0).with_quantity(2),
        ];

        let config = BenchmarkConfig3D::default();
        let runner = BenchmarkRunner3D::new(config);
        let geometries = runner.convert_to_geometries(&items);

        // 1 + 2 = 3 geometries
        assert_eq!(geometries.len(), 3);
    }
}
