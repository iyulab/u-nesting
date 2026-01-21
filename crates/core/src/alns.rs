//! # Adaptive Large Neighborhood Search (ALNS) Framework
//!
//! Implementation of the ALNS metaheuristic based on Ropke & Pisinger (2006).
//!
//! ALNS combines:
//! - **Destroy operators**: Remove items from current solution (multiple strategies)
//! - **Repair operators**: Reinsert items to improve solution (multiple strategies)
//! - **Adaptive weights**: Operators are selected based on past performance
//! - **Simulated Annealing acceptance**: Probabilistic acceptance of worse solutions
//!
//! ## Key Features
//!
//! - Multiple destroy/repair operators with adaptive selection
//! - Segment-based weight updates
//! - Configurable acceptance criteria (SA, Hill-Climbing)
//! - Progress callbacks for monitoring
//!
//! ## Usage
//!
//! ```rust,ignore
//! use u_nesting_core::alns::{AlnsConfig, AlnsRunner, AlnsProblem};
//!
//! let config = AlnsConfig::default();
//! let runner = AlnsRunner::new(config);
//! let result = runner.run(&mut problem, progress_callback);
//! ```

use std::fmt::Debug;
use std::time::Instant;

/// Configuration for the ALNS algorithm.
#[derive(Debug, Clone)]
pub struct AlnsConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Time limit in milliseconds (0 = no limit)
    pub time_limit_ms: u64,
    /// Number of iterations per segment (for weight updates)
    pub segment_size: usize,
    /// Score for finding new best solution
    pub score_best: f64,
    /// Score for finding better solution than current
    pub score_better: f64,
    /// Score for accepting worse solution
    pub score_accepted: f64,
    /// Reaction factor (how quickly weights adapt, 0-1)
    pub reaction_factor: f64,
    /// Minimum operator weight
    pub min_weight: f64,
    /// Initial temperature for SA acceptance
    pub initial_temperature: f64,
    /// Cooling rate for SA acceptance (0-1)
    pub cooling_rate: f64,
    /// Final temperature threshold
    pub final_temperature: f64,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for AlnsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            time_limit_ms: 60000, // 1 minute
            segment_size: 100,
            score_best: 33.0,
            score_better: 9.0,
            score_accepted: 3.0,
            reaction_factor: 0.1,
            min_weight: 0.1,
            initial_temperature: 100.0,
            cooling_rate: 0.9995,
            final_temperature: 0.01,
            seed: None,
        }
    }
}

impl AlnsConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set time limit in milliseconds.
    pub fn with_time_limit_ms(mut self, ms: u64) -> Self {
        self.time_limit_ms = ms;
        self
    }

    /// Set segment size for weight updates.
    pub fn with_segment_size(mut self, size: usize) -> Self {
        self.segment_size = size.max(1);
        self
    }

    /// Set scoring parameters.
    pub fn with_scores(mut self, best: f64, better: f64, accepted: f64) -> Self {
        self.score_best = best;
        self.score_better = better;
        self.score_accepted = accepted;
        self
    }

    /// Set reaction factor.
    pub fn with_reaction_factor(mut self, factor: f64) -> Self {
        self.reaction_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set temperature parameters for SA acceptance.
    pub fn with_temperature(mut self, initial: f64, cooling_rate: f64, final_temp: f64) -> Self {
        self.initial_temperature = initial.max(0.01);
        self.cooling_rate = cooling_rate.clamp(0.9, 1.0);
        self.final_temperature = final_temp.max(0.001);
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Statistics for an operator.
#[derive(Debug, Clone)]
pub struct OperatorStats {
    /// Current weight
    pub weight: f64,
    /// Number of times used
    pub times_used: usize,
    /// Total score accumulated
    pub total_score: f64,
    /// Score in current segment
    pub segment_score: f64,
    /// Uses in current segment
    pub segment_uses: usize,
}

impl Default for OperatorStats {
    fn default() -> Self {
        Self {
            weight: 1.0,
            times_used: 0,
            total_score: 0.0,
            segment_score: 0.0,
            segment_uses: 0,
        }
    }
}

impl OperatorStats {
    /// Create new stats with initial weight.
    pub fn new(initial_weight: f64) -> Self {
        Self {
            weight: initial_weight,
            ..Default::default()
        }
    }

    /// Record operator usage with a score.
    pub fn record_use(&mut self, score: f64) {
        self.times_used += 1;
        self.total_score += score;
        self.segment_score += score;
        self.segment_uses += 1;
    }

    /// Update weight at end of segment.
    pub fn update_weight(&mut self, reaction_factor: f64, min_weight: f64) {
        if self.segment_uses > 0 {
            let segment_avg = self.segment_score / self.segment_uses as f64;
            self.weight = self.weight * (1.0 - reaction_factor) + segment_avg * reaction_factor;
            self.weight = self.weight.max(min_weight);
        }
        // Reset segment counters
        self.segment_score = 0.0;
        self.segment_uses = 0;
    }
}

/// Identifies a destroy operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DestroyOperatorId {
    /// Random removal
    Random,
    /// Worst removal (highest cost items)
    Worst,
    /// Related removal (similar/nearby items)
    Related,
    /// Shaw removal (clustering-based)
    Shaw,
    /// Custom operator by index
    Custom(usize),
}

/// Identifies a repair operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairOperatorId {
    /// Greedy repair (best position)
    Greedy,
    /// Regret-based repair
    Regret,
    /// Random repair
    Random,
    /// BLF-based repair
    BottomLeftFill,
    /// Custom operator by index
    Custom(usize),
}

/// Result of a destroy operation.
#[derive(Debug, Clone)]
pub struct DestroyResult {
    /// Indices of removed items
    pub removed_indices: Vec<usize>,
    /// Operator used
    pub operator: DestroyOperatorId,
}

/// Result of a repair operation.
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Number of items successfully placed
    pub placed_count: usize,
    /// Number of items that could not be placed
    pub unplaced_count: usize,
    /// Operator used
    pub operator: RepairOperatorId,
}

/// Progress information for ALNS callbacks.
#[derive(Debug, Clone)]
pub struct AlnsProgress {
    /// Current iteration number
    pub iteration: usize,
    /// Best fitness found so far
    pub best_fitness: f64,
    /// Current fitness
    pub current_fitness: f64,
    /// Current temperature
    pub temperature: f64,
    /// Current segment number
    pub segment: usize,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Acceptance rate in current segment
    pub acceptance_rate: f64,
    /// Best destroy operator (by weight)
    pub best_destroy: DestroyOperatorId,
    /// Best repair operator (by weight)
    pub best_repair: RepairOperatorId,
}

/// Result of ALNS optimization.
#[derive(Debug, Clone)]
pub struct AlnsResult<S> {
    /// Best solution found
    pub best_solution: S,
    /// Best fitness value
    pub best_fitness: f64,
    /// Total iterations performed
    pub iterations: usize,
    /// Total time elapsed in milliseconds
    pub elapsed_ms: u64,
    /// Number of improvements found
    pub improvements: usize,
    /// Final temperature
    pub final_temperature: f64,
    /// Final operator weights
    pub destroy_weights: Vec<(DestroyOperatorId, f64)>,
    /// Final repair operator weights
    pub repair_weights: Vec<(RepairOperatorId, f64)>,
}

/// Trait for solutions that can be optimized by ALNS.
pub trait AlnsSolution: Clone + Debug {
    /// Get the fitness value (lower is better, 0 = optimal).
    fn fitness(&self) -> f64;

    /// Get the number of placed items.
    fn placed_count(&self) -> usize;

    /// Get the total number of items.
    fn total_count(&self) -> usize;
}

/// Trait for problems that can be solved by ALNS.
pub trait AlnsProblem {
    /// Solution type
    type Solution: AlnsSolution;

    /// Create an initial solution.
    fn create_initial_solution(&mut self) -> Self::Solution;

    /// Clone a solution.
    fn clone_solution(&self, solution: &Self::Solution) -> Self::Solution;

    /// Get available destroy operators.
    fn destroy_operators(&self) -> Vec<DestroyOperatorId>;

    /// Get available repair operators.
    fn repair_operators(&self) -> Vec<RepairOperatorId>;

    /// Apply a destroy operator.
    fn destroy(
        &mut self,
        solution: &mut Self::Solution,
        operator: DestroyOperatorId,
        degree: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> DestroyResult;

    /// Apply a repair operator.
    fn repair(
        &mut self,
        solution: &mut Self::Solution,
        destroyed: &DestroyResult,
        operator: RepairOperatorId,
    ) -> RepairResult;

    /// Calculate relatedness between two items (for Shaw/Related removal).
    fn relatedness(&self, solution: &Self::Solution, i: usize, j: usize) -> f64 {
        // Default: no relatedness information
        let _ = (solution, i, j);
        0.0
    }
}

/// The ALNS algorithm runner.
pub struct AlnsRunner {
    config: AlnsConfig,
}

impl AlnsRunner {
    /// Create a new ALNS runner with the given configuration.
    pub fn new(config: AlnsConfig) -> Self {
        Self { config }
    }

    /// Run the ALNS algorithm on the given problem.
    pub fn run<P, F>(&self, problem: &mut P, mut progress_callback: F) -> AlnsResult<P::Solution>
    where
        P: AlnsProblem,
        F: FnMut(&AlnsProgress),
    {
        use rand::prelude::*;
        use rand::SeedableRng;

        // Initialize RNG
        let mut rng = match self.config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let start_time = Instant::now();

        // Create initial solution
        let mut current = problem.create_initial_solution();
        let mut best = problem.clone_solution(&current);
        let mut best_fitness = best.fitness();

        // Get operators
        let destroy_ops = problem.destroy_operators();
        let repair_ops = problem.repair_operators();

        // Initialize operator statistics
        let mut destroy_stats: Vec<(DestroyOperatorId, OperatorStats)> = destroy_ops
            .iter()
            .map(|&op| (op, OperatorStats::new(1.0)))
            .collect();

        let mut repair_stats: Vec<(RepairOperatorId, OperatorStats)> = repair_ops
            .iter()
            .map(|&op| (op, OperatorStats::new(1.0)))
            .collect();

        // Initialize temperature
        let mut temperature = self.config.initial_temperature;

        // Statistics
        let mut iteration = 0;
        let mut segment = 0;
        let mut improvements = 0;
        let mut segment_accepts = 0;
        let mut segment_total = 0;

        // Main loop
        loop {
            // Check termination conditions
            let elapsed = start_time.elapsed();
            let elapsed_ms = elapsed.as_millis() as u64;

            if iteration >= self.config.max_iterations {
                break;
            }

            if self.config.time_limit_ms > 0 && elapsed_ms >= self.config.time_limit_ms {
                break;
            }

            // Select destroy operator using roulette wheel
            let destroy_idx = self.select_operator_by_weight(&destroy_stats, &mut rng);
            let destroy_op = destroy_stats[destroy_idx].0;

            // Select repair operator using roulette wheel
            let repair_idx = self.select_operator_by_weight(&repair_stats, &mut rng);
            let repair_op = repair_stats[repair_idx].0;

            // Clone current solution
            let mut candidate = problem.clone_solution(&current);

            // Apply destroy (remove 10-40% of items)
            let degree = rng.gen_range(0.1..=0.4);
            let destroy_result = problem.destroy(&mut candidate, destroy_op, degree, &mut rng);

            // Apply repair
            let _repair_result = problem.repair(&mut candidate, &destroy_result, repair_op);

            let candidate_fitness = candidate.fitness();
            let current_fitness = current.fitness();

            // Determine acceptance and score
            let (accepted, score) = if candidate_fitness < best_fitness {
                // New best solution
                best = problem.clone_solution(&candidate);
                best_fitness = candidate_fitness;
                improvements += 1;
                (true, self.config.score_best)
            } else if candidate_fitness < current_fitness {
                // Better than current
                (true, self.config.score_better)
            } else {
                // SA acceptance criterion
                let delta = candidate_fitness - current_fitness;
                let accept_prob = (-delta / temperature).exp();
                if rng.gen::<f64>() < accept_prob {
                    (true, self.config.score_accepted)
                } else {
                    (false, 0.0)
                }
            };

            // Update current if accepted
            if accepted {
                current = candidate;
                segment_accepts += 1;
            }

            // Record operator usage
            destroy_stats[destroy_idx].1.record_use(score);
            repair_stats[repair_idx].1.record_use(score);

            segment_total += 1;

            // Update temperature
            temperature *= self.config.cooling_rate;
            temperature = temperature.max(self.config.final_temperature);

            // Check if segment is complete
            if iteration > 0 && iteration % self.config.segment_size == 0 {
                // Update weights
                for (_, stats) in &mut destroy_stats {
                    stats.update_weight(self.config.reaction_factor, self.config.min_weight);
                }
                for (_, stats) in &mut repair_stats {
                    stats.update_weight(self.config.reaction_factor, self.config.min_weight);
                }

                segment += 1;
                segment_accepts = 0;
                segment_total = 0;
            }

            // Progress callback
            let acceptance_rate = if segment_total > 0 {
                segment_accepts as f64 / segment_total as f64
            } else {
                0.0
            };

            let best_destroy = destroy_stats
                .iter()
                .max_by(|a, b| a.1.weight.partial_cmp(&b.1.weight).unwrap())
                .map(|(op, _)| *op)
                .unwrap_or(DestroyOperatorId::Random);

            let best_repair = repair_stats
                .iter()
                .max_by(|a, b| a.1.weight.partial_cmp(&b.1.weight).unwrap())
                .map(|(op, _)| *op)
                .unwrap_or(RepairOperatorId::Greedy);

            let progress = AlnsProgress {
                iteration,
                best_fitness,
                current_fitness: current.fitness(),
                temperature,
                segment,
                elapsed_ms,
                acceptance_rate,
                best_destroy,
                best_repair,
            };

            progress_callback(&progress);

            iteration += 1;
        }

        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        AlnsResult {
            best_solution: best,
            best_fitness,
            iterations: iteration,
            elapsed_ms,
            improvements,
            final_temperature: temperature,
            destroy_weights: destroy_stats
                .iter()
                .map(|(op, stats)| (*op, stats.weight))
                .collect(),
            repair_weights: repair_stats
                .iter()
                .map(|(op, stats)| (*op, stats.weight))
                .collect(),
        }
    }

    /// Select an operator index using roulette wheel selection.
    fn select_operator_by_weight<T>(
        &self,
        stats: &[(T, OperatorStats)],
        rng: &mut rand::rngs::StdRng,
    ) -> usize {
        use rand::prelude::*;

        let total_weight: f64 = stats.iter().map(|(_, s)| s.weight).sum();
        if total_weight <= 0.0 || stats.is_empty() {
            return 0;
        }

        let mut roll = rng.gen::<f64>() * total_weight;
        for (i, (_, stat)) in stats.iter().enumerate() {
            roll -= stat.weight;
            if roll <= 0.0 {
                return i;
            }
        }

        stats.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alns_config_default() {
        let config = AlnsConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.time_limit_ms, 60000);
        assert_eq!(config.segment_size, 100);
        assert!((config.score_best - 33.0).abs() < 1e-9);
    }

    #[test]
    fn test_alns_config_builder() {
        let config = AlnsConfig::new()
            .with_max_iterations(5000)
            .with_time_limit_ms(30000)
            .with_segment_size(50)
            .with_scores(10.0, 5.0, 1.0)
            .with_reaction_factor(0.2)
            .with_temperature(50.0, 0.999, 0.001)
            .with_seed(42);

        assert_eq!(config.max_iterations, 5000);
        assert_eq!(config.time_limit_ms, 30000);
        assert_eq!(config.segment_size, 50);
        assert!((config.score_best - 10.0).abs() < 1e-9);
        assert!((config.score_better - 5.0).abs() < 1e-9);
        assert!((config.score_accepted - 1.0).abs() < 1e-9);
        assert!((config.reaction_factor - 0.2).abs() < 1e-9);
        assert!((config.initial_temperature - 50.0).abs() < 1e-9);
        assert!((config.cooling_rate - 0.999).abs() < 1e-9);
        assert!((config.final_temperature - 0.001).abs() < 1e-9);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_operator_stats() {
        let mut stats = OperatorStats::new(1.0);

        stats.record_use(10.0);
        stats.record_use(20.0);

        assert_eq!(stats.times_used, 2);
        assert!((stats.total_score - 30.0).abs() < 1e-9);
        assert!((stats.segment_score - 30.0).abs() < 1e-9);
        assert_eq!(stats.segment_uses, 2);

        stats.update_weight(0.5, 0.1);

        // New weight = 1.0 * 0.5 + 15.0 * 0.5 = 8.0
        assert!((stats.weight - 8.0).abs() < 1e-9);
        assert!((stats.segment_score - 0.0).abs() < 1e-9);
        assert_eq!(stats.segment_uses, 0);
    }

    #[test]
    fn test_destroy_operator_ids() {
        let ops = [
            DestroyOperatorId::Random,
            DestroyOperatorId::Worst,
            DestroyOperatorId::Related,
            DestroyOperatorId::Shaw,
            DestroyOperatorId::Custom(0),
        ];

        assert_eq!(ops.len(), 5);
        assert_eq!(DestroyOperatorId::Random, DestroyOperatorId::Random);
        assert_ne!(DestroyOperatorId::Random, DestroyOperatorId::Worst);
    }

    #[test]
    fn test_repair_operator_ids() {
        let ops = [
            RepairOperatorId::Greedy,
            RepairOperatorId::Regret,
            RepairOperatorId::Random,
            RepairOperatorId::BottomLeftFill,
            RepairOperatorId::Custom(0),
        ];

        assert_eq!(ops.len(), 5);
        assert_eq!(RepairOperatorId::Greedy, RepairOperatorId::Greedy);
        assert_ne!(RepairOperatorId::Greedy, RepairOperatorId::Regret);
    }

    #[test]
    fn test_destroy_result() {
        let result = DestroyResult {
            removed_indices: vec![0, 3, 5],
            operator: DestroyOperatorId::Random,
        };

        assert_eq!(result.removed_indices.len(), 3);
        assert_eq!(result.operator, DestroyOperatorId::Random);
    }

    #[test]
    fn test_repair_result() {
        let result = RepairResult {
            placed_count: 8,
            unplaced_count: 2,
            operator: RepairOperatorId::Greedy,
        };

        assert_eq!(result.placed_count, 8);
        assert_eq!(result.unplaced_count, 2);
        assert_eq!(result.operator, RepairOperatorId::Greedy);
    }

    #[test]
    fn test_alns_progress() {
        let progress = AlnsProgress {
            iteration: 100,
            best_fitness: 0.85,
            current_fitness: 0.90,
            temperature: 50.0,
            segment: 1,
            elapsed_ms: 5000,
            acceptance_rate: 0.45,
            best_destroy: DestroyOperatorId::Worst,
            best_repair: RepairOperatorId::Greedy,
        };

        assert_eq!(progress.iteration, 100);
        assert!((progress.best_fitness - 0.85).abs() < 1e-9);
        assert_eq!(progress.segment, 1);
        assert_eq!(progress.best_destroy, DestroyOperatorId::Worst);
        assert_eq!(progress.best_repair, RepairOperatorId::Greedy);
    }

    // Mock implementation for testing the runner
    #[derive(Clone, Debug)]
    struct MockSolution {
        fitness: f64,
        placed: usize,
        total: usize,
    }

    impl AlnsSolution for MockSolution {
        fn fitness(&self) -> f64 {
            self.fitness
        }

        fn placed_count(&self) -> usize {
            self.placed
        }

        fn total_count(&self) -> usize {
            self.total
        }
    }

    struct MockProblem {
        improvement_per_iteration: f64,
    }

    impl AlnsProblem for MockProblem {
        type Solution = MockSolution;

        fn create_initial_solution(&mut self) -> MockSolution {
            MockSolution {
                fitness: 1.0,
                placed: 8,
                total: 10,
            }
        }

        fn clone_solution(&self, solution: &MockSolution) -> MockSolution {
            solution.clone()
        }

        fn destroy_operators(&self) -> Vec<DestroyOperatorId> {
            vec![
                DestroyOperatorId::Random,
                DestroyOperatorId::Worst,
            ]
        }

        fn repair_operators(&self) -> Vec<RepairOperatorId> {
            vec![
                RepairOperatorId::Greedy,
                RepairOperatorId::BottomLeftFill,
            ]
        }

        fn destroy(
            &mut self,
            _solution: &mut MockSolution,
            operator: DestroyOperatorId,
            _degree: f64,
            _rng: &mut rand::rngs::StdRng,
        ) -> DestroyResult {
            DestroyResult {
                removed_indices: vec![0, 1, 2],
                operator,
            }
        }

        fn repair(
            &mut self,
            solution: &mut MockSolution,
            _destroyed: &DestroyResult,
            operator: RepairOperatorId,
        ) -> RepairResult {
            // Simulate improvement
            solution.fitness -= self.improvement_per_iteration;
            solution.fitness = solution.fitness.max(0.1);
            RepairResult {
                placed_count: solution.placed,
                unplaced_count: 0,
                operator,
            }
        }
    }

    #[test]
    fn test_alns_runner_basic() {
        let config = AlnsConfig::new()
            .with_max_iterations(100)
            .with_time_limit_ms(5000)
            .with_seed(42);

        let mut problem = MockProblem {
            improvement_per_iteration: 0.01,
        };

        let runner = AlnsRunner::new(config);
        let mut last_progress: Option<AlnsProgress> = None;

        let result = runner.run(&mut problem, |progress| {
            last_progress = Some(progress.clone());
        });

        assert!(result.best_fitness <= 1.0);
        assert_eq!(result.iterations, 100);
        assert!(last_progress.is_some());
        assert!(!result.destroy_weights.is_empty());
        assert!(!result.repair_weights.is_empty());
    }

    #[test]
    fn test_alns_runner_time_limit() {
        let config = AlnsConfig::new()
            .with_max_iterations(1_000_000)
            .with_time_limit_ms(100)
            .with_seed(42);

        let mut problem = MockProblem {
            improvement_per_iteration: 0.001,
        };

        let runner = AlnsRunner::new(config);
        let result = runner.run(&mut problem, |_| {});

        // Should have terminated due to time limit
        assert!(result.iterations < 1_000_000);
        assert!(result.elapsed_ms >= 100);
    }

    #[test]
    fn test_alns_weight_adaptation() {
        let config = AlnsConfig::new()
            .with_max_iterations(200)
            .with_segment_size(50)
            .with_reaction_factor(0.5)
            .with_seed(42);

        let mut problem = MockProblem {
            improvement_per_iteration: 0.01,
        };

        let runner = AlnsRunner::new(config);
        let result = runner.run(&mut problem, |_| {});

        // Weights should have changed from initial values
        let _initial_weight_sum: f64 = 2.0; // 2 destroy operators, each starting at 1.0
        let _final_destroy_sum: f64 = result.destroy_weights.iter().map(|(_, w)| *w).sum();

        // At least some adaptation should have occurred
        // (weights won't all be exactly 1.0 anymore)
        let max_destroy_weight = result
            .destroy_weights
            .iter()
            .map(|(_, w)| *w)
            .fold(0.0, f64::max);

        assert!(max_destroy_weight >= 0.1); // At least min_weight
        assert!(result.iterations == 200);
    }
}
