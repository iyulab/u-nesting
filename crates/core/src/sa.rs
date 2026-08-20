//! Simulated Annealing framework for optimization.
//!
//! # Architecture
//!
//! This module maintains its own SA loop rather than delegating to u-metaheur's
//! SA runner. u-metaheur uses `cost(&self, &Solution) -> f64` (immutable),
//! while u-nesting uses `evaluate(&self, &mut Solution)` (mutable) for
//! consistency with the GA/BRKGA evaluation pattern.
//!
//! Additionally, this module provides features not available in u-metaheur SA:
//! - `NeighborhoodOperator` enum (5 operators) with `available_operators()`
//! - `PermutationSolution` with built-in swap/relocate/inversion/rotation/chain
//! - Reheating with configurable threshold and factor
//! - Adaptive cooling schedule
//! - Parallel multi-restart via `run_parallel()`
//!
//! The rand 0.9 API is shared with u-metaheur for ecosystem compatibility.

use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::timing::Timer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Cooling schedule types for Simulated Annealing.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CoolingSchedule {
    /// Geometric cooling: T_new = T * alpha (alpha typically 0.95-0.99).
    #[default]
    Geometric,
    /// Linear cooling: T_new = T - delta.
    Linear,
    /// Adaptive cooling: adjusts based on acceptance rate.
    Adaptive,
    /// Lundy-Mees: T_new = T / (1 + beta * T).
    LundyMees,
}

/// Configuration for Simulated Annealing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SaConfig {
    /// Initial temperature.
    pub initial_temp: f64,
    /// Final (minimum) temperature.
    pub final_temp: f64,
    /// Cooling rate (alpha for Geometric, delta for Linear, beta for LundyMees).
    pub cooling_rate: f64,
    /// Number of iterations at each temperature level.
    pub iterations_per_temp: usize,
    /// Maximum total iterations (None = temperature-based stopping only).
    pub max_iterations: Option<u64>,
    /// Cooling schedule type.
    pub cooling_schedule: CoolingSchedule,
    /// Maximum time limit (None = unlimited).
    pub time_limit: Option<Duration>,
    /// Target fitness to stop early (None = run until temperature limit).
    pub target_fitness: Option<f64>,
    /// Enable reheating when stagnation detected.
    pub enable_reheating: bool,
    /// Stagnation threshold for reheating.
    pub reheat_threshold: u64,
    /// Reheat factor (multiplier for current temperature).
    pub reheat_factor: f64,
}

impl Default for SaConfig {
    fn default() -> Self {
        Self {
            initial_temp: 1000.0,
            final_temp: 0.001,
            cooling_rate: 0.95,
            iterations_per_temp: 100,
            max_iterations: Some(100_000),
            cooling_schedule: CoolingSchedule::Geometric,
            time_limit: None,
            target_fitness: None,
            enable_reheating: false,
            reheat_threshold: 1000,
            reheat_factor: 2.0,
        }
    }
}

impl SaConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial temperature.
    pub fn with_initial_temp(mut self, temp: f64) -> Self {
        self.initial_temp = temp.max(0.001);
        self
    }

    /// Sets the final temperature.
    pub fn with_final_temp(mut self, temp: f64) -> Self {
        self.final_temp = temp.max(0.0001);
        self
    }

    /// Sets the cooling rate.
    pub fn with_cooling_rate(mut self, rate: f64) -> Self {
        self.cooling_rate = rate.clamp(0.001, 0.9999);
        self
    }

    /// Sets the iterations per temperature level.
    pub fn with_iterations_per_temp(mut self, iterations: usize) -> Self {
        self.iterations_per_temp = iterations.max(1);
        self
    }

    /// Sets the maximum iterations.
    pub fn with_max_iterations(mut self, iterations: u64) -> Self {
        self.max_iterations = Some(iterations);
        self
    }

    /// Sets the cooling schedule.
    pub fn with_cooling_schedule(mut self, schedule: CoolingSchedule) -> Self {
        self.cooling_schedule = schedule;
        self
    }

    /// Sets the time limit.
    pub fn with_time_limit(mut self, duration: Duration) -> Self {
        self.time_limit = Some(duration);
        self
    }

    /// Sets the target fitness.
    pub fn with_target_fitness(mut self, fitness: f64) -> Self {
        self.target_fitness = Some(fitness);
        self
    }

    /// Enables reheating.
    pub fn with_reheating(mut self, threshold: u64, factor: f64) -> Self {
        self.enable_reheating = true;
        self.reheat_threshold = threshold;
        self.reheat_factor = factor.max(1.1);
        self
    }
}

/// Trait for solutions in Simulated Annealing.
pub trait SaSolution: Clone + Send + Sync {
    /// Returns the objective value (fitness) of this solution.
    /// Higher values are better (maximization).
    fn objective(&self) -> f64;

    /// Sets the objective value.
    fn set_objective(&mut self, value: f64);
}

/// Neighborhood operator types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborhoodOperator {
    /// Swap two elements.
    Swap,
    /// Relocate an element to a new position.
    Relocate,
    /// Reverse a segment (2-opt style).
    Inversion,
    /// Rotate/change orientation of an element.
    Rotation,
    /// Chain swap (3-opt style).
    Chain,
    /// Flip an element's mirror flag (`allow_flip` support, 2D nesting only
    /// — a solution's `mirrors` vector is harmless-if-unused by problems
    /// with no mirror concept, same as an all-`false` `rotations` vector is
    /// for a problem with a single rotation option).
    MirrorFlip,
}

/// Trait for problem-specific SA operations.
pub trait SaProblem: Send + Sync {
    /// The solution type for this problem.
    type Solution: SaSolution;

    /// Creates an initial solution.
    fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution;

    /// Generates a neighbor solution using the specified operator.
    fn neighbor<R: Rng>(
        &self,
        solution: &Self::Solution,
        operator: NeighborhoodOperator,
        rng: &mut R,
    ) -> Self::Solution;

    /// Evaluates the objective of a solution.
    fn evaluate(&self, solution: &mut Self::Solution);

    /// Returns available neighborhood operators for this problem.
    fn available_operators(&self) -> Vec<NeighborhoodOperator> {
        vec![
            NeighborhoodOperator::Swap,
            NeighborhoodOperator::Relocate,
            NeighborhoodOperator::Inversion,
        ]
    }

    /// Called after each temperature level (for progress reporting).
    fn on_temperature_change(
        &self,
        _temperature: f64,
        _iteration: u64,
        _best: &Self::Solution,
        _current: &Self::Solution,
    ) {
        // Default: do nothing
    }
}

/// Progress information during SA execution.
#[derive(Debug, Clone)]
pub struct SaProgress {
    /// Current temperature.
    pub temperature: f64,
    /// Current iteration number.
    pub iteration: u64,
    /// Best fitness so far.
    pub best_fitness: f64,
    /// Current fitness.
    pub current_fitness: f64,
    /// Acceptance rate at current temperature.
    pub acceptance_rate: f64,
    /// Elapsed time since start.
    pub elapsed: Duration,
    /// Whether the algorithm is still running.
    pub running: bool,
}

/// Result of a SA run.
#[derive(Debug, Clone)]
pub struct SaResult<S: SaSolution> {
    /// The best solution found.
    pub best: S,
    /// Final temperature reached.
    pub final_temperature: f64,
    /// Total iterations performed.
    pub iterations: u64,
    /// Total elapsed time.
    pub elapsed: Duration,
    /// Whether the target fitness was reached.
    pub target_reached: bool,
    /// Number of reheats performed.
    pub reheat_count: u32,
    /// Fitness history (sampled at temperature changes).
    pub history: Vec<f64>,
}

/// Simulated Annealing runner.
pub struct SaRunner<P: SaProblem> {
    config: SaConfig,
    problem: P,
    cancelled: Arc<AtomicBool>,
}

impl<P: SaProblem> SaRunner<P> {
    /// Creates a new SA runner.
    pub fn new(config: SaConfig, problem: P) -> Self {
        Self {
            config,
            problem,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns a handle to cancel the algorithm.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    /// Runs the Simulated Annealing algorithm.
    pub fn run(&self) -> SaResult<P::Solution> {
        self.run_with_rng(&mut rand::rng())
    }

    /// Runs the Simulated Annealing algorithm with a specific RNG.
    pub fn run_with_rng<R: Rng>(&self, rng: &mut R) -> SaResult<P::Solution> {
        let start = Timer::now();
        let mut history = Vec::new();

        // Initialize
        let mut current = self.problem.initial_solution(rng);
        self.problem.evaluate(&mut current);
        let mut best = current.clone();
        let mut best_fitness = best.objective();

        let mut temperature = self.config.initial_temp;
        let mut iteration = 0u64;
        let mut target_reached = false;
        let mut reheat_count = 0u32;
        let mut stagnation_count = 0u64;

        let operators = self.problem.available_operators();
        let temp_delta = if matches!(self.config.cooling_schedule, CoolingSchedule::Linear) {
            (self.config.initial_temp - self.config.final_temp)
                / (self.config.max_iterations.unwrap_or(10000) as f64
                    / self.config.iterations_per_temp as f64)
        } else {
            0.0
        };

        // For adaptive cooling
        let mut accepted_count = 0usize;
        let mut total_count = 0usize;

        while temperature > self.config.final_temp {
            // Check cancellation
            if self.cancelled.load(Ordering::Relaxed) {
                break;
            }

            // Check time limit
            if let Some(limit) = self.config.time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }

            // Check max iterations
            if let Some(max) = self.config.max_iterations {
                if iteration >= max {
                    break;
                }
            }

            // Check target fitness
            if let Some(target) = self.config.target_fitness {
                if best_fitness >= target {
                    target_reached = true;
                    break;
                }
            }

            // Iterations at this temperature
            for _ in 0..self.config.iterations_per_temp {
                iteration += 1;
                total_count += 1;

                // Select random operator
                let operator = operators[rng.random_range(0..operators.len())];

                // Generate neighbor
                let mut neighbor = self.problem.neighbor(&current, operator, rng);
                self.problem.evaluate(&mut neighbor);

                let current_obj = current.objective();
                let neighbor_obj = neighbor.objective();
                let delta = neighbor_obj - current_obj;

                // Accept or reject
                let accept = if delta >= 0.0 {
                    // Better solution - always accept
                    true
                } else {
                    // Worse solution - accept with probability exp(delta/T)
                    let probability = (delta / temperature).exp();
                    rng.random::<f64>() < probability
                };

                if accept {
                    accepted_count += 1;
                    current = neighbor;

                    // Update best
                    if current.objective() > best_fitness {
                        best = current.clone();
                        best_fitness = best.objective();
                        stagnation_count = 0;
                    } else {
                        stagnation_count += 1;
                    }
                } else {
                    stagnation_count += 1;
                }

                // Check max iterations inside inner loop
                if let Some(max) = self.config.max_iterations {
                    if iteration >= max {
                        break;
                    }
                }
            }

            // Record history
            history.push(best_fitness);

            // Callback
            self.problem
                .on_temperature_change(temperature, iteration, &best, &current);

            // Reheating check
            if self.config.enable_reheating && stagnation_count >= self.config.reheat_threshold {
                temperature *= self.config.reheat_factor;
                temperature = temperature.min(self.config.initial_temp);
                stagnation_count = 0;
                reheat_count += 1;
            }

            // Cool down
            temperature = self.cool_down(temperature, temp_delta, accepted_count, total_count);

            // Reset adaptive counters
            accepted_count = 0;
            total_count = 0;
        }

        // Final history entry
        history.push(best_fitness);

        SaResult {
            best,
            final_temperature: temperature,
            iterations: iteration,
            elapsed: start.elapsed(),
            target_reached,
            reheat_count,
            history,
        }
    }

    /// Apply cooling schedule.
    fn cool_down(&self, current_temp: f64, delta: f64, accepted: usize, total: usize) -> f64 {
        match self.config.cooling_schedule {
            CoolingSchedule::Geometric => current_temp * self.config.cooling_rate,
            CoolingSchedule::Linear => (current_temp - delta).max(self.config.final_temp),
            CoolingSchedule::Adaptive => {
                // Adjust cooling rate based on acceptance rate
                let acceptance_rate = if total > 0 {
                    accepted as f64 / total as f64
                } else {
                    0.5
                };

                // If acceptance rate is high, cool faster; if low, cool slower
                let adjusted_rate = if acceptance_rate > 0.5 {
                    self.config.cooling_rate * 0.95 // Cool faster
                } else if acceptance_rate < 0.1 {
                    self.config.cooling_rate.powf(0.5) // Cool slower (sqrt)
                } else {
                    self.config.cooling_rate
                };

                current_temp * adjusted_rate
            }
            CoolingSchedule::LundyMees => {
                // T_new = T / (1 + beta * T)
                current_temp / (1.0 + self.config.cooling_rate * current_temp)
            }
        }
    }

    /// Runs multiple SA instances in parallel and returns the best result.
    ///
    /// This is useful for escaping local optima by exploring different regions
    /// of the solution space simultaneously.
    ///
    /// # Arguments
    /// * `num_restarts` - Number of parallel SA runs to perform
    ///
    /// # Returns
    /// The best result among all parallel runs
    ///
    /// Requires the `parallel` feature to be enabled.
    #[cfg(feature = "parallel")]
    pub fn run_parallel(&self, num_restarts: usize) -> SaResult<P::Solution>
    where
        P: Clone,
    {
        let num_restarts = num_restarts.max(1);

        // Run SA instances in parallel
        let results: Vec<SaResult<P::Solution>> = (0..num_restarts)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                self.run_with_rng(&mut rng)
            })
            .collect();

        // Find the best result
        results
            .into_iter()
            .max_by(|a, b| {
                a.best
                    .objective()
                    .partial_cmp(&b.best.objective())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("At least one result should exist")
    }
}

/// Permutation-based solution for SA.
#[derive(Debug, Clone)]
pub struct PermutationSolution {
    /// The permutation (indices).
    pub sequence: Vec<usize>,
    /// Additional rotation/orientation values.
    pub rotations: Vec<usize>,
    /// Number of rotation options per item.
    pub rotation_options: usize,
    /// Mirror flag per item (`allow_flip` support, 2D nesting only — see
    /// [`NeighborhoodOperator::MirrorFlip`]).
    pub mirrors: Vec<bool>,
    /// Cached objective value.
    objective: f64,
}

impl PermutationSolution {
    /// Creates a new permutation solution.
    pub fn new(size: usize, rotation_options: usize) -> Self {
        Self {
            sequence: (0..size).collect(),
            rotations: vec![0; size],
            rotation_options,
            mirrors: vec![false; size],
            objective: f64::NEG_INFINITY,
        }
    }

    /// Creates a random permutation solution.
    pub fn random<R: Rng>(size: usize, rotation_options: usize, rng: &mut R) -> Self {
        let mut sequence: Vec<usize> = (0..size).collect();
        sequence.shuffle(rng);

        let rotations: Vec<usize> = (0..size)
            .map(|_| rng.random_range(0..rotation_options.max(1)))
            .collect();

        let mirrors: Vec<bool> = (0..size).map(|_| rng.random()).collect();

        Self {
            sequence,
            rotations,
            rotation_options,
            mirrors,
            objective: f64::NEG_INFINITY,
        }
    }

    /// Returns the length of the sequence.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Applies swap operator: swaps two elements in sequence.
    pub fn apply_swap<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        if result.sequence.len() < 2 {
            return result;
        }

        let i = rng.random_range(0..result.sequence.len());
        let j = rng.random_range(0..result.sequence.len());
        result.sequence.swap(i, j);
        result.objective = f64::NEG_INFINITY;
        result
    }

    /// Applies relocate operator: moves an element to a new position.
    pub fn apply_relocate<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        if result.sequence.len() < 2 {
            return result;
        }

        let from = rng.random_range(0..result.sequence.len());
        let to = rng.random_range(0..result.sequence.len());

        if from != to {
            let elem = result.sequence.remove(from);
            let insert_pos = if to > from { to - 1 } else { to };
            result
                .sequence
                .insert(insert_pos.min(result.sequence.len()), elem);
        }

        result.objective = f64::NEG_INFINITY;
        result
    }

    /// Applies inversion operator: reverses a segment.
    pub fn apply_inversion<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        let n = result.sequence.len();
        if n < 2 {
            return result;
        }

        let (mut p1, mut p2) = (rng.random_range(0..n), rng.random_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        result.sequence[p1..=p2].reverse();
        result.objective = f64::NEG_INFINITY;
        result
    }

    /// Applies rotation operator: changes rotation of one element.
    pub fn apply_rotation<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        if result.rotations.is_empty() || result.rotation_options <= 1 {
            return result;
        }

        let idx = rng.random_range(0..result.rotations.len());
        result.rotations[idx] = rng.random_range(0..result.rotation_options);
        result.objective = f64::NEG_INFINITY;
        result
    }

    /// Applies mirror-flip operator: flips one element's mirror bit
    /// (`allow_flip` support). Same shape as `apply_rotation` — unconditional
    /// on the gene, the consuming problem masks it against its own
    /// per-item mirror-eligibility (e.g. `Geometry2D::allow_flip()`).
    pub fn apply_mirror_flip<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        if result.mirrors.is_empty() {
            return result;
        }

        let idx = rng.random_range(0..result.mirrors.len());
        result.mirrors[idx] = !result.mirrors[idx];
        result.objective = f64::NEG_INFINITY;
        result
    }

    /// Applies chain operator: 3-opt style move.
    pub fn apply_chain<R: Rng>(&self, rng: &mut R) -> Self {
        let mut result = self.clone();
        let n = result.sequence.len();
        if n < 4 {
            // Fall back to swap for small sequences
            return self.apply_swap(rng);
        }

        // Select three distinct positions
        let mut positions: Vec<usize> = (0..n).collect();
        positions.shuffle(rng);
        let mut selected: Vec<usize> = positions.into_iter().take(3).collect();
        selected.sort();

        let (p1, p2, p3) = (selected[0], selected[1], selected[2]);

        // Rotate segments: [0..p1] [p1..p2] [p2..p3] [p3..n]
        // New order: [0..p1] [p2..p3] [p1..p2] [p3..n]
        let seg1: Vec<usize> = result.sequence[..p1].to_vec();
        let seg2: Vec<usize> = result.sequence[p1..p2].to_vec();
        let seg3: Vec<usize> = result.sequence[p2..p3].to_vec();
        let seg4: Vec<usize> = result.sequence[p3..].to_vec();

        result.sequence = [seg1, seg3, seg2, seg4].concat();
        result.objective = f64::NEG_INFINITY;
        result
    }
}

impl SaSolution for PermutationSolution {
    fn objective(&self) -> f64 {
        self.objective
    }

    fn set_objective(&mut self, value: f64) {
        self.objective = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleMaxProblem {
        size: usize,
    }

    impl SaProblem for SimpleMaxProblem {
        type Solution = PermutationSolution;

        fn initial_solution<R: Rng>(&self, rng: &mut R) -> Self::Solution {
            PermutationSolution::random(self.size, 1, rng)
        }

        fn neighbor<R: Rng>(
            &self,
            solution: &Self::Solution,
            operator: NeighborhoodOperator,
            rng: &mut R,
        ) -> Self::Solution {
            match operator {
                NeighborhoodOperator::Swap => solution.apply_swap(rng),
                NeighborhoodOperator::Relocate => solution.apply_relocate(rng),
                NeighborhoodOperator::Inversion => solution.apply_inversion(rng),
                NeighborhoodOperator::Rotation => solution.apply_rotation(rng),
                NeighborhoodOperator::Chain => solution.apply_chain(rng),
                NeighborhoodOperator::MirrorFlip => solution.apply_mirror_flip(rng),
            }
        }

        fn evaluate(&self, solution: &mut Self::Solution) {
            // Maximize: sequence should be in ascending order
            // Fitness = negative sum of inversions
            let mut inversions = 0i64;
            for i in 0..solution.sequence.len() {
                for j in (i + 1)..solution.sequence.len() {
                    if solution.sequence[i] > solution.sequence[j] {
                        inversions += 1;
                    }
                }
            }
            solution.set_objective(-inversions as f64);
        }
    }

    #[test]
    fn test_sa_basic() {
        let config = SaConfig::default()
            .with_initial_temp(100.0)
            .with_final_temp(0.1)
            .with_cooling_rate(0.9)
            .with_iterations_per_temp(50)
            .with_max_iterations(5000);

        let problem = SimpleMaxProblem { size: 10 };
        let runner = SaRunner::new(config, problem);
        let result = runner.run();

        // Should find something reasonably good (fewer inversions)
        assert!(result.best.objective() > -20.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_cooling_schedules() {
        let problem = SimpleMaxProblem { size: 5 };

        for schedule in [
            CoolingSchedule::Geometric,
            CoolingSchedule::Linear,
            CoolingSchedule::Adaptive,
            CoolingSchedule::LundyMees,
        ] {
            let config = SaConfig::default()
                .with_cooling_schedule(schedule)
                .with_max_iterations(1000);

            let runner = SaRunner::new(config, problem.clone());
            let result = runner.run();

            // Should complete without panic
            assert!(result.iterations > 0);
        }
    }

    #[test]
    fn test_neighborhood_operators() {
        let mut rng = rand::rng();
        let solution = PermutationSolution::random(10, 4, &mut rng);

        // Test all operators produce valid permutations
        let swap = solution.apply_swap(&mut rng);
        let relocate = solution.apply_relocate(&mut rng);
        let inversion = solution.apply_inversion(&mut rng);
        let rotation = solution.apply_rotation(&mut rng);
        let chain = solution.apply_chain(&mut rng);

        for sol in [&swap, &relocate, &inversion, &rotation, &chain] {
            let mut sorted = sol.sequence.clone();
            sorted.sort();
            assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_reheating() {
        let config = SaConfig::default()
            .with_initial_temp(10.0)
            .with_final_temp(0.1)
            .with_max_iterations(500)
            .with_reheating(50, 1.5);

        let problem = SimpleMaxProblem { size: 8 };
        let runner = SaRunner::new(config, problem);
        let result = runner.run();

        // Should complete (reheating may or may not trigger)
        assert!(result.iterations > 0);
    }

    impl Clone for SimpleMaxProblem {
        fn clone(&self) -> Self {
            Self { size: self.size }
        }
    }
}
