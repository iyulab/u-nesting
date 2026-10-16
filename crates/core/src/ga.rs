//! Genetic Algorithm framework for optimization.
//!
//! This module provides the GA abstraction layer for u-nesting. It defines
//! domain-specific traits ([`Individual`], [`GaProblem`]) that support
//! mutable evaluation — the key difference from u-metaheur's immutable pattern.
//!
//! # Architecture
//!
//! u-nesting uses a **mutable evaluation** pattern: `evaluate(&mut Individual)`
//! sets both fitness and auxiliary state (e.g., `placed_count`, `total_count`).
//! This differs from u-metaheur's `evaluate(&Individual) -> Fitness` pattern,
//! which only returns a fitness value without modifying the individual.
//!
//! Because of this fundamental difference, u-nesting maintains its own
//! evolutionary loop while sharing the rand/rayon ecosystem with u-metaheur.
//! Crossover and mutation operators are defined on [`Individual`] (u-nesting
//! convention), not on [`GaProblem`] (u-metaheur convention).

use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::error::{check_at_least, check_range, Result};
use crate::result::StopReason;
use crate::timing::{evaluate_within, expired, Timer};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the genetic algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GaConfig {
    /// Population size.
    pub population_size: usize,
    /// Maximum number of generations.
    pub max_generations: u32,
    /// Crossover rate (0.0 - 1.0).
    pub crossover_rate: f64,
    /// Mutation rate (0.0 - 1.0).
    pub mutation_rate: f64,
    /// Number of elite individuals to preserve each generation.
    pub elite_count: usize,
    /// Tournament size for selection.
    pub tournament_size: usize,
    /// Maximum time limit (None = unlimited).
    pub time_limit: Option<Duration>,
    /// Target fitness to stop early (None = run all generations).
    pub target_fitness: Option<f64>,
    /// Stagnation generations before early stop.
    pub stagnation_limit: Option<u32>,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 500,
            crossover_rate: 0.85,
            mutation_rate: 0.05,
            elite_count: 5,
            tournament_size: 3,
            time_limit: None,
            target_fitness: None,
            stagnation_limit: Some(50),
        }
    }
}

impl GaConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the population size.
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Sets the maximum generations.
    pub fn with_max_generations(mut self, gen: u32) -> Self {
        self.max_generations = gen;
        self
    }

    /// Sets the crossover rate.
    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Sets the mutation rate.
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Sets the elite count.
    pub fn with_elite_count(mut self, count: usize) -> Self {
        self.elite_count = count;
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

    /// Checks every field against its range: the builders store what they are
    /// given, and this is the one place a value is judged.
    ///
    /// # Errors
    /// `ConfigError` naming the first field out of range: a crossover or
    /// mutation rate outside `[0, 1]`, a population below 2 or a tournament of 0.
    pub fn validate(&self) -> Result<()> {
        check_range("crossover_rate", self.crossover_rate, 0.0, 1.0)?;
        check_range("mutation_rate", self.mutation_rate, 0.0, 1.0)?;
        check_at_least("population_size", self.population_size as f64, 2.0)?;
        check_at_least("tournament_size", self.tournament_size as f64, 1.0)?;
        Ok(())
    }
}

/// Trait for individuals in the genetic algorithm.
///
/// In u-nesting's convention, crossover and mutation are defined on the
/// individual itself (unlike u-metaheur which places them on GaProblem).
pub trait Individual: Clone + Send + Sync {
    /// The fitness type (usually f64).
    type Fitness: PartialOrd + Copy + Send;

    /// Returns the fitness of this individual.
    fn fitness(&self) -> Self::Fitness;

    /// Creates a random individual.
    fn random<R: Rng>(rng: &mut R) -> Self;

    /// Performs crossover with another individual.
    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self;

    /// Mutates this individual in place.
    fn mutate<R: Rng>(&mut self, rng: &mut R);
}

/// Trait for problem-specific GA operations.
///
/// Uses mutable evaluation: `evaluate(&mut Individual)` can set both
/// fitness and auxiliary state on the individual.
pub trait GaProblem: Send + Sync {
    /// The individual type for this problem.
    type Individual: Individual;

    /// Evaluates the fitness of an individual (mutable — can set auxiliary state).
    fn evaluate(&self, individual: &mut Self::Individual);

    /// Evaluates multiple individuals in parallel.
    /// Default implementation uses rayon when the `parallel` feature is enabled.
    fn evaluate_parallel(&self, individuals: &mut [Self::Individual]) {
        #[cfg(feature = "parallel")]
        individuals.par_iter_mut().for_each(|ind| {
            self.evaluate(ind);
        });
        #[cfg(not(feature = "parallel"))]
        for ind in individuals.iter_mut() {
            self.evaluate(ind);
        }
    }

    /// Creates an initial population.
    fn initialize_population<R: Rng>(&self, size: usize, rng: &mut R) -> Vec<Self::Individual> {
        (0..size).map(|_| Self::Individual::random(rng)).collect()
    }

    /// Called after each generation (for progress reporting).
    fn on_generation(
        &self,
        _generation: u32,
        _best: &Self::Individual,
        _population: &[Self::Individual],
    ) {
        // Default: do nothing
    }
}

/// Progress information during GA execution.
#[derive(Debug, Clone)]
pub struct GaProgress<F> {
    /// Current generation number.
    pub generation: u32,
    /// Maximum generations configured.
    pub max_generations: u32,
    /// Best fitness so far.
    pub best_fitness: F,
    /// Average fitness of current population.
    pub avg_fitness: f64,
    /// Elapsed time since start.
    pub elapsed: Duration,
    /// Whether the algorithm is still running.
    pub running: bool,
}

/// Result of a GA run.
#[derive(Debug, Clone)]
pub struct GaResult<I: Individual> {
    /// The best individual found.
    pub best: I,
    /// Final generation reached.
    pub generations: u32,
    /// Total elapsed time.
    pub elapsed: Duration,
    /// Why the run stopped.
    pub stop_reason: StopReason,
    /// Fitness history (best fitness per generation).
    pub history: Vec<f64>,
}

/// Genetic algorithm runner.
///
/// Runs the evolutionary loop with mutable evaluation, tournament selection,
/// elitism, and configurable stopping conditions.
pub struct GaRunner<P: GaProblem> {
    config: GaConfig,
    problem: P,
    cancelled: Arc<AtomicBool>,
}

impl<P: GaProblem> GaRunner<P>
where
    <P::Individual as Individual>::Fitness: Into<f64>,
{
    /// Creates a new GA runner.
    ///
    /// # Errors
    /// As [`GaConfig::validate`].
    pub fn new(config: GaConfig, problem: P) -> Result<Self> {
        Self::with_cancellation(config, problem, Arc::new(AtomicBool::new(false)))
    }

    /// Creates a runner that stops when `cancelled` is set — the caller's own
    /// flag, observed directly, so no thread is needed to forward it.
    ///
    /// # Errors
    /// As [`GaConfig::validate`].
    pub fn with_cancellation(
        config: GaConfig,
        problem: P,
        cancelled: Arc<AtomicBool>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            problem,
            cancelled,
        })
    }

    /// Returns a handle to cancel the algorithm.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    /// Runs the genetic algorithm.
    pub fn run(&self) -> GaResult<P::Individual> {
        self.run_with_rng(&mut rand::rng())
    }

    /// Runs the genetic algorithm with a progress callback.
    pub fn run_with_progress<F>(&self, progress_callback: F) -> GaResult<P::Individual>
    where
        F: Fn(GaProgress<<P::Individual as Individual>::Fitness>),
    {
        self.run_with_rng_and_progress(&mut rand::rng(), Some(progress_callback))
    }

    /// Runs the genetic algorithm with a specific RNG.
    pub fn run_with_rng<R: Rng>(&self, rng: &mut R) -> GaResult<P::Individual> {
        self.run_with_rng_and_progress::<R, fn(GaProgress<<P::Individual as Individual>::Fitness>)>(
            rng, None,
        )
    }

    /// Runs the genetic algorithm with a specific RNG and optional progress callback.
    pub fn run_with_rng_and_progress<R: Rng, F>(
        &self,
        rng: &mut R,
        progress_callback: Option<F>,
    ) -> GaResult<P::Individual>
    where
        F: Fn(GaProgress<<P::Individual as Individual>::Fitness>),
    {
        let start = Timer::now();
        let mut history = Vec::new();

        // A report after every evaluated batch, not only every generation: one
        // generation of a large problem can take seconds, and a caller that
        // cancels from its callback has to be heard within it. Returns the
        // best fitness seen, counting the batch.
        let heartbeat = |generation: u32,
                         best: Option<<P::Individual as Individual>::Fitness>,
                         batch: &[P::Individual]| {
            let best = batch
                .iter()
                .map(Individual::fitness)
                .fold(best, |b, f| match b {
                    Some(b) if b >= f => Some(b),
                    _ => Some(f),
                });
            if let (Some(callback), Some(best_fitness)) = (&progress_callback, best) {
                let avg_fitness = batch.iter().map(|i| i.fitness().into()).sum::<f64>()
                    / batch.len().max(1) as f64;
                callback(GaProgress {
                    generation,
                    max_generations: self.config.max_generations,
                    best_fitness,
                    avg_fitness,
                    elapsed: start.elapsed(),
                    running: true,
                });
            }
            best
        };

        // Initialize population
        let mut population = self
            .problem
            .initialize_population(self.config.population_size, rng);

        // Evaluate the initial population — up to the time limit
        let mut best_so_far = None;
        evaluate_within(&mut population, &start, self.config.time_limit, |batch| {
            self.problem.evaluate_parallel(batch);
            best_so_far = heartbeat(0, best_so_far, batch);
        });

        // Sort by fitness (descending - higher is better in u-nesting convention)
        population.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut best = population[0].clone();
        let mut best_fitness: f64 = best.fitness().into();
        let mut stagnation_count = 0u32;
        let mut generation = 0u32;
        let reached = |fitness: f64| self.config.target_fitness.is_some_and(|t| fitness >= t);

        // Each exit names its reason. The target is judged before the
        // generation budget, so a target met by the last generation is
        // reported as met.
        let stop_reason = loop {
            if reached(best_fitness) {
                break StopReason::TargetReached;
            }
            if generation >= self.config.max_generations {
                break StopReason::IterationLimit;
            }
            if self.cancelled.load(Ordering::Relaxed) {
                break StopReason::Cancelled;
            }
            if expired(&start, self.config.time_limit) {
                break StopReason::TimeLimit;
            }

            // Record history
            history.push(best_fitness);

            // Create new generation
            let mut new_population = Vec::with_capacity(self.config.population_size);

            // Elitism: keep the best individuals
            for individual in population
                .iter()
                .take(self.config.elite_count.min(population.len()))
            {
                new_population.push(individual.clone());
            }

            // Fill the rest with crossover and mutation
            let mut children: Vec<P::Individual> =
                Vec::with_capacity(self.config.population_size - new_population.len());

            while children.len() < self.config.population_size - new_population.len() {
                // Tournament selection
                let parent1 = self.tournament_select(&population, rng);
                let parent2 = self.tournament_select(&population, rng);

                // Crossover (on Individual, u-nesting convention)
                let mut child = if rng.random::<f64>() < self.config.crossover_rate {
                    parent1.crossover(parent2, rng)
                } else {
                    parent1.clone()
                };

                // Mutation (on Individual, u-nesting convention)
                if rng.random::<f64>() < self.config.mutation_rate {
                    child.mutate(rng);
                }

                children.push(child);
            }

            // Evaluate the children — up to the time limit; the loop ends on the
            // next check if it was reached
            let best_now = best.fitness();
            evaluate_within(&mut children, &start, self.config.time_limit, |batch| {
                self.problem.evaluate_parallel(batch);
                heartbeat(generation, Some(best_now), batch);
            });

            // Add evaluated children to new population
            new_population.extend(children);

            // Sort new population
            new_population.sort_by(|a, b| {
                b.fitness()
                    .partial_cmp(&a.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update best
            let new_best_fitness: f64 = new_population[0].fitness().into();
            if new_best_fitness > best_fitness {
                best = new_population[0].clone();
                best_fitness = new_best_fitness;
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }

            // Check stagnation
            if let Some(limit) = self.config.stagnation_limit {
                if stagnation_count >= limit {
                    break StopReason::NoImprovement;
                }
            }

            // Callback to GaProblem
            self.problem
                .on_generation(generation, &best, &new_population);

            // Progress callback
            if let Some(ref callback) = progress_callback {
                let avg_fitness = new_population
                    .iter()
                    .map(|ind| ind.fitness().into())
                    .sum::<f64>()
                    / new_population.len() as f64;

                callback(GaProgress {
                    generation,
                    max_generations: self.config.max_generations,
                    best_fitness: best.fitness(),
                    avg_fitness,
                    elapsed: start.elapsed(),
                    running: true,
                });
            }

            population = new_population;
            generation += 1;
        };

        // Final history entry
        history.push(best_fitness);

        // Final progress callback indicating completion
        if let Some(ref callback) = progress_callback {
            let avg_fitness = population
                .iter()
                .map(|ind| ind.fitness().into())
                .sum::<f64>()
                / population.len().max(1) as f64;

            callback(GaProgress {
                generation,
                max_generations: self.config.max_generations,
                best_fitness: best.fitness(),
                avg_fitness,
                elapsed: start.elapsed(),
                running: false,
            });
        }

        GaResult {
            best,
            generations: generation,
            elapsed: start.elapsed(),
            stop_reason,
            history,
        }
    }

    /// Tournament selection (maximization — higher fitness wins).
    fn tournament_select<'a, R: Rng>(
        &self,
        population: &'a [P::Individual],
        rng: &mut R,
    ) -> &'a P::Individual {
        let mut best_idx = rng.random_range(0..population.len());

        for _ in 1..self.config.tournament_size {
            let idx = rng.random_range(0..population.len());
            if population[idx].fitness() > population[best_idx].fitness() {
                best_idx = idx;
            }
        }

        &population[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct SimpleIndividual {
        value: f64,
    }

    impl Individual for SimpleIndividual {
        type Fitness = f64;

        fn fitness(&self) -> f64 {
            // Maximize: -(x^2), optimal at x=0
            -self.value * self.value
        }

        fn random<R: Rng>(rng: &mut R) -> Self {
            Self {
                value: rng.random_range(-100.0..100.0),
            }
        }

        fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
            Self {
                value: if rng.random() {
                    self.value
                } else {
                    other.value
                },
            }
        }

        fn mutate<R: Rng>(&mut self, rng: &mut R) {
            self.value += rng.random_range(-10.0..10.0);
        }
    }

    struct SimpleProblem;

    impl GaProblem for SimpleProblem {
        type Individual = SimpleIndividual;

        fn evaluate(&self, _individual: &mut Self::Individual) {
            // Fitness is computed in Individual::fitness()
        }
    }

    #[test]
    fn test_ga_basic() {
        let config = GaConfig::default()
            .with_population_size(50)
            .with_max_generations(100)
            .with_target_fitness(-0.01);

        let runner = GaRunner::new(config, SimpleProblem).expect("valid config");
        let result = runner.run();

        // Should find something close to 0
        assert!(result.best.value.abs() < 5.0);
    }
}
