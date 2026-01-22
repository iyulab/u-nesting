//! # Goal-Driven Ruin and Recreate (GDRR) Framework
//!
//! Implementation of the GDRR metaheuristic based on Gardeyn & Wauters (EJOR 2022).
//!
//! GDRR combines:
//! - **Ruin operators**: Remove items from current solution (Random, Cluster, Worst)
//! - **Recreate operators**: Reinsert items to improve solution (BestFit, BLF, NFP)
//! - **Goal-driven mechanism**: Decreasing bin area/volume limit to guide search
//! - **Late Acceptance Hill-Climbing (LAHC)**: Accept criterion for diversification
//!
//! ## Usage
//!
//! ```rust,ignore
//! use u_nesting_core::gdrr::{GdrrConfig, GdrrRunner, GdrrProblem};
//!
//! let config = GdrrConfig::default();
//! let runner = GdrrRunner::new(config);
//! let result = runner.run(&mut problem, progress_callback);
//! ```

use std::fmt::Debug;
use std::time::Instant;

/// Configuration for the GDRR algorithm.
#[derive(Debug, Clone)]
pub struct GdrrConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Time limit in milliseconds (0 = no limit)
    pub time_limit_ms: u64,
    /// Minimum ruin percentage (fraction of items to remove)
    pub min_ruin_ratio: f64,
    /// Maximum ruin percentage
    pub max_ruin_ratio: f64,
    /// LAHC list length for acceptance criterion
    pub lahc_list_length: usize,
    /// Initial goal ratio (starting bin size as fraction of optimal)
    pub initial_goal_ratio: f64,
    /// Goal decrease rate per iteration without improvement
    pub goal_decrease_rate: f64,
    /// Minimum goal ratio (prevents goal from becoming too small)
    pub min_goal_ratio: f64,
    /// Iterations without improvement before goal decrease
    pub stagnation_threshold: usize,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Weight for random ruin operator
    pub random_ruin_weight: f64,
    /// Weight for cluster ruin operator
    pub cluster_ruin_weight: f64,
    /// Weight for worst ruin operator
    pub worst_ruin_weight: f64,
}

impl Default for GdrrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            time_limit_ms: 60000, // 1 minute
            min_ruin_ratio: 0.1,
            max_ruin_ratio: 0.4,
            lahc_list_length: 50,
            initial_goal_ratio: 1.2,
            goal_decrease_rate: 0.995,
            min_goal_ratio: 0.8,
            stagnation_threshold: 100,
            seed: None,
            random_ruin_weight: 1.0,
            cluster_ruin_weight: 1.0,
            worst_ruin_weight: 1.0,
        }
    }
}

impl GdrrConfig {
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

    /// Set ruin ratio range.
    pub fn with_ruin_ratio(mut self, min: f64, max: f64) -> Self {
        self.min_ruin_ratio = min.clamp(0.0, 1.0);
        self.max_ruin_ratio = max.clamp(self.min_ruin_ratio, 1.0);
        self
    }

    /// Set LAHC list length.
    pub fn with_lahc_list_length(mut self, length: usize) -> Self {
        self.lahc_list_length = length.max(1);
        self
    }

    /// Set goal-driven parameters.
    pub fn with_goal_params(
        mut self,
        initial_ratio: f64,
        decrease_rate: f64,
        min_ratio: f64,
    ) -> Self {
        self.initial_goal_ratio = initial_ratio.max(1.0);
        self.goal_decrease_rate = decrease_rate.clamp(0.9, 1.0);
        self.min_goal_ratio = min_ratio.clamp(0.5, 1.0);
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set ruin operator weights.
    pub fn with_ruin_weights(mut self, random: f64, cluster: f64, worst: f64) -> Self {
        self.random_ruin_weight = random.max(0.0);
        self.cluster_ruin_weight = cluster.max(0.0);
        self.worst_ruin_weight = worst.max(0.0);
        self
    }
}

/// Types of ruin operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuinType {
    /// Remove random items
    Random,
    /// Remove spatially clustered items
    Cluster,
    /// Remove items with worst placement scores
    Worst,
}

/// Types of recreate operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecreateType {
    /// Best-fit decreasing by area/volume
    BestFit,
    /// Bottom-left fill heuristic
    BottomLeftFill,
    /// NFP-guided placement (2D only)
    NfpGuided,
}

/// A removed item during ruin phase.
#[derive(Debug, Clone)]
pub struct RuinedItem {
    /// Index of the item in the original solution
    pub index: usize,
    /// ID of the geometry
    pub geometry_id: String,
    /// Position before removal (x, y) for 2D or (x, y, z) for 3D
    pub position: Vec<f64>,
    /// Rotation before removal
    pub rotation: f64,
    /// Placement score (for worst ruin)
    pub score: f64,
}

/// Result of a ruin operation.
#[derive(Debug, Clone)]
pub struct RuinResult {
    /// Items that were removed
    pub removed_items: Vec<RuinedItem>,
    /// Type of ruin operator used
    pub ruin_type: RuinType,
}

/// Result of a recreate operation.
#[derive(Debug, Clone)]
pub struct RecreateResult {
    /// Number of items successfully placed
    pub placed_count: usize,
    /// Number of items that could not be placed
    pub unplaced_count: usize,
    /// Type of recreate operator used
    pub recreate_type: RecreateType,
}

/// Progress information for GDRR callbacks.
#[derive(Debug, Clone)]
pub struct GdrrProgress {
    /// Current iteration number
    pub iteration: usize,
    /// Best fitness found so far
    pub best_fitness: f64,
    /// Current fitness
    pub current_fitness: f64,
    /// Current goal (bin size limit)
    pub current_goal: f64,
    /// Iterations since last improvement
    pub stagnation_count: usize,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Acceptance rate in recent iterations
    pub acceptance_rate: f64,
}

/// Result of GDRR optimization.
#[derive(Debug, Clone)]
pub struct GdrrResult<S> {
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
    /// Final goal value
    pub final_goal: f64,
}

/// Trait for solutions that can be optimized by GDRR.
pub trait GdrrSolution: Clone + Debug {
    /// Get the fitness value (lower is better, 0 = optimal).
    fn fitness(&self) -> f64;

    /// Get the number of placed items.
    fn placed_count(&self) -> usize;

    /// Get the total number of items.
    fn total_count(&self) -> usize;

    /// Get the utilization ratio (0.0 to 1.0).
    fn utilization(&self) -> f64;

    /// Check if the solution fits within the goal.
    fn fits_goal(&self, goal: f64) -> bool;
}

/// Trait for problems that can be solved by GDRR.
pub trait GdrrProblem {
    /// Solution type
    type Solution: GdrrSolution;

    /// Create an initial solution.
    fn create_initial_solution(&mut self) -> Self::Solution;

    /// Clone a solution.
    fn clone_solution(&self, solution: &Self::Solution) -> Self::Solution;

    /// Apply random ruin operator.
    fn ruin_random(
        &mut self,
        solution: &mut Self::Solution,
        ratio: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> RuinResult;

    /// Apply cluster ruin operator.
    fn ruin_cluster(
        &mut self,
        solution: &mut Self::Solution,
        ratio: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> RuinResult;

    /// Apply worst ruin operator.
    fn ruin_worst(
        &mut self,
        solution: &mut Self::Solution,
        ratio: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> RuinResult;

    /// Apply best-fit recreate operator.
    fn recreate_best_fit(
        &mut self,
        solution: &mut Self::Solution,
        ruined: &RuinResult,
    ) -> RecreateResult;

    /// Apply BLF recreate operator.
    fn recreate_blf(
        &mut self,
        solution: &mut Self::Solution,
        ruined: &RuinResult,
    ) -> RecreateResult;

    /// Apply NFP-guided recreate operator (2D only).
    fn recreate_nfp(
        &mut self,
        solution: &mut Self::Solution,
        ruined: &RuinResult,
    ) -> RecreateResult;

    /// Calculate placement score for an item (for worst ruin).
    fn placement_score(&self, solution: &Self::Solution, item_index: usize) -> f64;

    /// Get spatial neighbors of an item (for cluster ruin).
    fn get_neighbors(
        &self,
        solution: &Self::Solution,
        item_index: usize,
        radius: f64,
    ) -> Vec<usize>;
}

/// The GDRR algorithm runner.
pub struct GdrrRunner {
    config: GdrrConfig,
}

impl GdrrRunner {
    /// Create a new GDRR runner with the given configuration.
    pub fn new(config: GdrrConfig) -> Self {
        Self { config }
    }

    /// Run the GDRR algorithm on the given problem.
    pub fn run<P, F>(&self, problem: &mut P, mut progress_callback: F) -> GdrrResult<P::Solution>
    where
        P: GdrrProblem,
        F: FnMut(&GdrrProgress),
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

        // Initialize LAHC list
        let mut lahc_list: Vec<f64> = vec![best_fitness; self.config.lahc_list_length];
        let mut lahc_index = 0;

        // Initialize goal
        let optimal_estimate = best_fitness;
        let mut current_goal = optimal_estimate * self.config.initial_goal_ratio;

        // Statistics
        let mut iteration = 0;
        let mut stagnation_count = 0;
        let mut improvements = 0;
        let mut _accepted_count = 0;
        let acceptance_window = 100;
        let mut recent_accepts: Vec<bool> = Vec::with_capacity(acceptance_window);

        // Calculate total ruin weight
        let total_ruin_weight = self.config.random_ruin_weight
            + self.config.cluster_ruin_weight
            + self.config.worst_ruin_weight;

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

            // Select ruin ratio
            let ruin_ratio = rng.gen_range(self.config.min_ruin_ratio..=self.config.max_ruin_ratio);

            // Select ruin operator using roulette wheel
            let ruin_roll: f64 = rng.gen::<f64>() * total_ruin_weight;
            let ruin_type = if ruin_roll < self.config.random_ruin_weight {
                RuinType::Random
            } else if ruin_roll < self.config.random_ruin_weight + self.config.cluster_ruin_weight {
                RuinType::Cluster
            } else {
                RuinType::Worst
            };

            // Clone current solution for modification
            let mut candidate = problem.clone_solution(&current);

            // Apply ruin
            let ruin_result = match ruin_type {
                RuinType::Random => problem.ruin_random(&mut candidate, ruin_ratio, &mut rng),
                RuinType::Cluster => problem.ruin_cluster(&mut candidate, ruin_ratio, &mut rng),
                RuinType::Worst => problem.ruin_worst(&mut candidate, ruin_ratio, &mut rng),
            };

            // Select recreate operator (cycle through them for diversity)
            let recreate_type = match iteration % 3 {
                0 => RecreateType::BestFit,
                1 => RecreateType::BottomLeftFill,
                _ => RecreateType::NfpGuided,
            };

            // Apply recreate
            let _recreate_result = match recreate_type {
                RecreateType::BestFit => problem.recreate_best_fit(&mut candidate, &ruin_result),
                RecreateType::BottomLeftFill => problem.recreate_blf(&mut candidate, &ruin_result),
                RecreateType::NfpGuided => problem.recreate_nfp(&mut candidate, &ruin_result),
            };

            let candidate_fitness = candidate.fitness();

            // LAHC acceptance criterion
            let lahc_threshold = lahc_list[lahc_index % self.config.lahc_list_length];
            let current_fitness = current.fitness();

            let accepted = candidate_fitness <= current_fitness
                || candidate_fitness <= lahc_threshold
                || (candidate.fits_goal(current_goal) && !current.fits_goal(current_goal));

            if accepted {
                current = candidate;
                lahc_list[lahc_index % self.config.lahc_list_length] = current.fitness();
                _accepted_count += 1;

                // Update best if improved
                if current.fitness() < best_fitness {
                    best = problem.clone_solution(&current);
                    best_fitness = best.fitness();
                    stagnation_count = 0;
                    improvements += 1;
                } else {
                    stagnation_count += 1;
                }
            } else {
                stagnation_count += 1;
            }

            // Track recent accepts for acceptance rate
            if recent_accepts.len() >= acceptance_window {
                recent_accepts.remove(0);
            }
            recent_accepts.push(accepted);

            // Goal-driven mechanism: decrease goal after stagnation
            if stagnation_count >= self.config.stagnation_threshold {
                current_goal *= self.config.goal_decrease_rate;
                current_goal = current_goal.max(optimal_estimate * self.config.min_goal_ratio);
            }

            // Progress callback
            let acceptance_rate = if recent_accepts.is_empty() {
                0.0
            } else {
                recent_accepts.iter().filter(|&&a| a).count() as f64 / recent_accepts.len() as f64
            };

            let progress = GdrrProgress {
                iteration,
                best_fitness,
                current_fitness: current.fitness(),
                current_goal,
                stagnation_count,
                elapsed_ms,
                acceptance_rate,
            };

            progress_callback(&progress);

            iteration += 1;
            lahc_index += 1;
        }

        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        GdrrResult {
            best_solution: best,
            best_fitness,
            iterations: iteration,
            elapsed_ms,
            improvements,
            final_goal: current_goal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdrr_config_default() {
        let config = GdrrConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.time_limit_ms, 60000);
        assert!((config.min_ruin_ratio - 0.1).abs() < 1e-9);
        assert!((config.max_ruin_ratio - 0.4).abs() < 1e-9);
        assert_eq!(config.lahc_list_length, 50);
    }

    #[test]
    fn test_gdrr_config_builder() {
        let config = GdrrConfig::new()
            .with_max_iterations(5000)
            .with_time_limit_ms(30000)
            .with_ruin_ratio(0.2, 0.5)
            .with_lahc_list_length(100)
            .with_goal_params(1.5, 0.99, 0.7)
            .with_seed(42)
            .with_ruin_weights(2.0, 1.5, 1.0);

        assert_eq!(config.max_iterations, 5000);
        assert_eq!(config.time_limit_ms, 30000);
        assert!((config.min_ruin_ratio - 0.2).abs() < 1e-9);
        assert!((config.max_ruin_ratio - 0.5).abs() < 1e-9);
        assert_eq!(config.lahc_list_length, 100);
        assert!((config.initial_goal_ratio - 1.5).abs() < 1e-9);
        assert!((config.goal_decrease_rate - 0.99).abs() < 1e-9);
        assert!((config.min_goal_ratio - 0.7).abs() < 1e-9);
        assert_eq!(config.seed, Some(42));
        assert!((config.random_ruin_weight - 2.0).abs() < 1e-9);
        assert!((config.cluster_ruin_weight - 1.5).abs() < 1e-9);
        assert!((config.worst_ruin_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ruin_type_variants() {
        let types = [RuinType::Random, RuinType::Cluster, RuinType::Worst];
        assert_eq!(types.len(), 3);
        assert_eq!(RuinType::Random, RuinType::Random);
        assert_ne!(RuinType::Random, RuinType::Cluster);
    }

    #[test]
    fn test_recreate_type_variants() {
        let types = [
            RecreateType::BestFit,
            RecreateType::BottomLeftFill,
            RecreateType::NfpGuided,
        ];
        assert_eq!(types.len(), 3);
        assert_eq!(RecreateType::BestFit, RecreateType::BestFit);
        assert_ne!(RecreateType::BestFit, RecreateType::BottomLeftFill);
    }

    #[test]
    fn test_ruined_item() {
        let item = RuinedItem {
            index: 5,
            geometry_id: "rect1".to_string(),
            position: vec![100.0, 200.0],
            rotation: 90.0,
            score: 0.75,
        };

        assert_eq!(item.index, 5);
        assert_eq!(item.geometry_id, "rect1");
        assert_eq!(item.position, vec![100.0, 200.0]);
        assert!((item.rotation - 90.0).abs() < 1e-9);
        assert!((item.score - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_ruin_result() {
        let result = RuinResult {
            removed_items: vec![
                RuinedItem {
                    index: 0,
                    geometry_id: "a".to_string(),
                    position: vec![0.0, 0.0],
                    rotation: 0.0,
                    score: 0.5,
                },
                RuinedItem {
                    index: 1,
                    geometry_id: "b".to_string(),
                    position: vec![10.0, 10.0],
                    rotation: 45.0,
                    score: 0.3,
                },
            ],
            ruin_type: RuinType::Random,
        };

        assert_eq!(result.removed_items.len(), 2);
        assert_eq!(result.ruin_type, RuinType::Random);
    }

    #[test]
    fn test_recreate_result() {
        let result = RecreateResult {
            placed_count: 8,
            unplaced_count: 2,
            recreate_type: RecreateType::BestFit,
        };

        assert_eq!(result.placed_count, 8);
        assert_eq!(result.unplaced_count, 2);
        assert_eq!(result.recreate_type, RecreateType::BestFit);
    }

    #[test]
    fn test_gdrr_progress() {
        let progress = GdrrProgress {
            iteration: 100,
            best_fitness: 0.85,
            current_fitness: 0.90,
            current_goal: 1.1,
            stagnation_count: 25,
            elapsed_ms: 5000,
            acceptance_rate: 0.45,
        };

        assert_eq!(progress.iteration, 100);
        assert!((progress.best_fitness - 0.85).abs() < 1e-9);
        assert!((progress.current_fitness - 0.90).abs() < 1e-9);
        assert!((progress.current_goal - 1.1).abs() < 1e-9);
        assert_eq!(progress.stagnation_count, 25);
        assert_eq!(progress.elapsed_ms, 5000);
        assert!((progress.acceptance_rate - 0.45).abs() < 1e-9);
    }

    // Mock implementation for testing the runner
    #[derive(Clone, Debug)]
    struct MockSolution {
        fitness: f64,
        placed: usize,
        total: usize,
    }

    impl GdrrSolution for MockSolution {
        fn fitness(&self) -> f64 {
            self.fitness
        }

        fn placed_count(&self) -> usize {
            self.placed
        }

        fn total_count(&self) -> usize {
            self.total
        }

        fn utilization(&self) -> f64 {
            self.placed as f64 / self.total as f64
        }

        fn fits_goal(&self, goal: f64) -> bool {
            self.fitness <= goal
        }
    }

    struct MockProblem {
        improvement_per_iteration: f64,
    }

    impl GdrrProblem for MockProblem {
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

        fn ruin_random(
            &mut self,
            _solution: &mut MockSolution,
            _ratio: f64,
            _rng: &mut rand::rngs::StdRng,
        ) -> RuinResult {
            RuinResult {
                removed_items: vec![],
                ruin_type: RuinType::Random,
            }
        }

        fn ruin_cluster(
            &mut self,
            _solution: &mut MockSolution,
            _ratio: f64,
            _rng: &mut rand::rngs::StdRng,
        ) -> RuinResult {
            RuinResult {
                removed_items: vec![],
                ruin_type: RuinType::Cluster,
            }
        }

        fn ruin_worst(
            &mut self,
            _solution: &mut MockSolution,
            _ratio: f64,
            _rng: &mut rand::rngs::StdRng,
        ) -> RuinResult {
            RuinResult {
                removed_items: vec![],
                ruin_type: RuinType::Worst,
            }
        }

        fn recreate_best_fit(
            &mut self,
            solution: &mut MockSolution,
            _ruined: &RuinResult,
        ) -> RecreateResult {
            // Simulate improvement
            solution.fitness -= self.improvement_per_iteration;
            solution.fitness = solution.fitness.max(0.1);
            RecreateResult {
                placed_count: solution.placed,
                unplaced_count: 0,
                recreate_type: RecreateType::BestFit,
            }
        }

        fn recreate_blf(
            &mut self,
            solution: &mut MockSolution,
            _ruined: &RuinResult,
        ) -> RecreateResult {
            solution.fitness -= self.improvement_per_iteration * 0.5;
            solution.fitness = solution.fitness.max(0.1);
            RecreateResult {
                placed_count: solution.placed,
                unplaced_count: 0,
                recreate_type: RecreateType::BottomLeftFill,
            }
        }

        fn recreate_nfp(
            &mut self,
            solution: &mut MockSolution,
            _ruined: &RuinResult,
        ) -> RecreateResult {
            solution.fitness -= self.improvement_per_iteration * 0.8;
            solution.fitness = solution.fitness.max(0.1);
            RecreateResult {
                placed_count: solution.placed,
                unplaced_count: 0,
                recreate_type: RecreateType::NfpGuided,
            }
        }

        fn placement_score(&self, _solution: &MockSolution, _item_index: usize) -> f64 {
            0.5
        }

        fn get_neighbors(
            &self,
            _solution: &MockSolution,
            _item_index: usize,
            _radius: f64,
        ) -> Vec<usize> {
            vec![]
        }
    }

    #[test]
    fn test_gdrr_runner_basic() {
        let config = GdrrConfig::new()
            .with_max_iterations(100)
            .with_time_limit_ms(5000)
            .with_seed(42);

        let mut problem = MockProblem {
            improvement_per_iteration: 0.01,
        };

        let runner = GdrrRunner::new(config);
        let mut last_progress: Option<GdrrProgress> = None;

        let result = runner.run(&mut problem, |progress| {
            last_progress = Some(progress.clone());
        });

        assert!(result.best_fitness <= 1.0);
        assert_eq!(result.iterations, 100);
        // elapsed_ms may be 0 for very fast iterations
        assert!(last_progress.is_some());
    }

    #[test]
    fn test_gdrr_runner_time_limit() {
        let config = GdrrConfig::new()
            .with_max_iterations(1_000_000) // Very high
            .with_time_limit_ms(100) // Very short
            .with_seed(42);

        let mut problem = MockProblem {
            improvement_per_iteration: 0.001,
        };

        let runner = GdrrRunner::new(config);
        let result = runner.run(&mut problem, |_| {});

        // Should have terminated due to time limit, not iteration limit
        assert!(result.iterations < 1_000_000);
        assert!(result.elapsed_ms >= 100);
    }
}
