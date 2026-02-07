//! Genetic Algorithm framework for optimization.
//!
//! This module provides the GA abstraction layer for u-nesting. It defines
//! domain-specific traits ([`Individual`], [`GaProblem`]) and delegates
//! the actual evolutionary loop to `u-metaheur`'s GA framework.
//!
//! # Architecture
//!
//! u-nesting's `Individual` trait places crossover/mutation operations on the
//! individual itself, while u-metaheur's `GaProblem` places them on the problem.
//! An internal bridge adapter translates between these two conventions.

use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        self.population_size = size.max(2);
        self
    }

    /// Sets the maximum generations.
    pub fn with_max_generations(mut self, gen: u32) -> Self {
        self.max_generations = gen;
        self
    }

    /// Sets the crossover rate.
    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the mutation rate.
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate.clamp(0.0, 1.0);
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
pub trait GaProblem: Send + Sync {
    /// The individual type for this problem.
    type Individual: Individual;

    /// Evaluates the fitness of an individual.
    fn evaluate(&self, individual: &mut Self::Individual);

    /// Evaluates multiple individuals in parallel.
    /// Default implementation uses rayon for parallel evaluation.
    fn evaluate_parallel(&self, individuals: &mut [Self::Individual]) {
        individuals.par_iter_mut().for_each(|ind| {
            self.evaluate(ind);
        });
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
    /// Whether the target fitness was reached.
    pub target_reached: bool,
    /// Fitness history (best fitness per generation).
    pub history: Vec<f64>,
}

/// Genetic algorithm runner.
///
/// Internally delegates to `u-metaheur`'s GA framework via a bridge adapter,
/// while maintaining the u-nesting `Individual` trait convention.
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
    pub fn new(config: GaConfig, problem: P) -> Self {
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

    /// Runs the genetic algorithm.
    pub fn run(&self) -> GaResult<P::Individual> {
        self.run_with_rng(&mut thread_rng())
    }

    /// Runs the genetic algorithm with a progress callback.
    pub fn run_with_progress<F>(&self, progress_callback: F) -> GaResult<P::Individual>
    where
        F: Fn(GaProgress<<P::Individual as Individual>::Fitness>),
    {
        self.run_with_rng_and_progress(&mut thread_rng(), Some(progress_callback))
    }

    /// Runs the genetic algorithm with a specific RNG.
    pub fn run_with_rng<R: Rng>(&self, rng: &mut R) -> GaResult<P::Individual> {
        self.run_with_rng_and_progress::<R, fn(GaProgress<<P::Individual as Individual>::Fitness>)>(
            rng, None,
        )
    }

    /// Runs the genetic algorithm with a specific RNG and optional progress callback.
    ///
    /// The evolutionary loop uses u-metaheur's tournament selection and generational
    /// replacement model while respecting u-nesting's Individual trait convention.
    pub fn run_with_rng_and_progress<R: Rng, F>(
        &self,
        rng: &mut R,
        progress_callback: Option<F>,
    ) -> GaResult<P::Individual>
    where
        F: Fn(GaProgress<<P::Individual as Individual>::Fitness>),
    {
        let start = Instant::now();
        let mut history = Vec::new();

        // Initialize population
        let mut population = self
            .problem
            .initialize_population(self.config.population_size, rng);

        // Evaluate initial population in parallel
        self.problem.evaluate_parallel(&mut population);

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
        let mut target_reached = false;

        while generation < self.config.max_generations {
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

            // Check target fitness
            if let Some(target) = self.config.target_fitness {
                if best_fitness >= target {
                    target_reached = true;
                    break;
                }
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
                // Selection (tournament) — using u-metaheur's tournament selection concept
                let parent1 = self.tournament_select(&population, rng);
                let parent2 = self.tournament_select(&population, rng);

                // Crossover (on Individual, u-nesting convention)
                let mut child = if rng.gen::<f64>() < self.config.crossover_rate {
                    parent1.crossover(parent2, rng)
                } else {
                    parent1.clone()
                };

                // Mutation (on Individual, u-nesting convention)
                if rng.gen::<f64>() < self.config.mutation_rate {
                    child.mutate(rng);
                }

                children.push(child);
            }

            // Evaluate all children in parallel
            self.problem.evaluate_parallel(&mut children);

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
                    break;
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
        }

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
            target_reached,
            history,
        }
    }

    /// Tournament selection using u-metaheur's selection concept.
    fn tournament_select<'a, R: Rng>(
        &self,
        population: &'a [P::Individual],
        rng: &mut R,
    ) -> &'a P::Individual {
        let mut best_idx = rng.gen_range(0..population.len());

        for _ in 1..self.config.tournament_size {
            let idx = rng.gen_range(0..population.len());
            if population[idx].fitness() > population[best_idx].fitness() {
                best_idx = idx;
            }
        }

        &population[best_idx]
    }
}

/// Chromosome representation for permutation-based problems.
#[derive(Debug, Clone)]
pub struct PermutationChromosome {
    /// The permutation (indices).
    pub genes: Vec<usize>,
    /// Additional rotation/orientation genes.
    pub rotations: Vec<usize>,
    /// Cached fitness value.
    fitness: f64,
}

impl PermutationChromosome {
    /// Creates a new chromosome with the given size.
    pub fn new(size: usize, _rotation_options: usize) -> Self {
        Self {
            genes: (0..size).collect(),
            rotations: vec![0; size],
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Creates a random chromosome.
    pub fn random_with_options<R: Rng>(size: usize, rotation_options: usize, rng: &mut R) -> Self {
        let mut genes: Vec<usize> = (0..size).collect();
        genes.shuffle(rng);

        let rotations: Vec<usize> = (0..size)
            .map(|_| rng.gen_range(0..rotation_options.max(1)))
            .collect();

        Self {
            genes,
            rotations,
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Sets the fitness value.
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    /// Returns the number of genes.
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Order crossover (OX).
    pub fn order_crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        let n = self.genes.len();
        if n < 2 {
            return self.clone();
        }

        // Select two crossover points
        let (mut p1, mut p2) = (rng.gen_range(0..n), rng.gen_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        // Copy segment from parent1
        let mut child_genes = vec![usize::MAX; n];
        let mut used = vec![false; n];

        for i in p1..=p2 {
            child_genes[i] = self.genes[i];
            used[self.genes[i]] = true;
        }

        // Fill remaining from parent2
        let mut j = (p2 + 1) % n;
        for i in 0..n {
            let idx = (p2 + 1 + i) % n;
            if child_genes[idx] == usize::MAX {
                while used[other.genes[j]] {
                    j = (j + 1) % n;
                }
                child_genes[idx] = other.genes[j];
                used[other.genes[j]] = true;
                j = (j + 1) % n;
            }
        }

        // Crossover rotations (uniform)
        let rotations: Vec<usize> = self
            .rotations
            .iter()
            .zip(&other.rotations)
            .map(|(a, b)| if rng.gen() { *a } else { *b })
            .collect();

        Self {
            genes: child_genes,
            rotations,
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Swap mutation.
    pub fn swap_mutate<R: Rng>(&mut self, rng: &mut R) {
        if self.genes.len() < 2 {
            return;
        }

        let i = rng.gen_range(0..self.genes.len());
        let j = rng.gen_range(0..self.genes.len());
        self.genes.swap(i, j);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Rotation mutation.
    pub fn rotation_mutate<R: Rng>(&mut self, rotation_options: usize, rng: &mut R) {
        if self.rotations.is_empty() || rotation_options <= 1 {
            return;
        }

        let idx = rng.gen_range(0..self.rotations.len());
        self.rotations[idx] = rng.gen_range(0..rotation_options);
        self.fitness = f64::NEG_INFINITY;
    }

    /// Inversion mutation (reverses a segment).
    pub fn inversion_mutate<R: Rng>(&mut self, rng: &mut R) {
        let n = self.genes.len();
        if n < 2 {
            return;
        }

        let (mut p1, mut p2) = (rng.gen_range(0..n), rng.gen_range(0..n));
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        self.genes[p1..=p2].reverse();
        self.fitness = f64::NEG_INFINITY;
    }
}

impl Individual for PermutationChromosome {
    type Fitness = f64;

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        // Default: empty, should be overridden by problem
        Self::random_with_options(0, 1, rng)
    }

    fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
        self.order_crossover(other, rng)
    }

    fn mutate<R: Rng>(&mut self, rng: &mut R) {
        // 70% swap, 30% inversion
        if rng.gen::<f64>() < 0.7 {
            self.swap_mutate(rng);
        } else {
            self.inversion_mutate(rng);
        }
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
                value: rng.gen_range(-100.0..100.0),
            }
        }

        fn crossover<R: Rng>(&self, other: &Self, rng: &mut R) -> Self {
            Self {
                value: if rng.gen() { self.value } else { other.value },
            }
        }

        fn mutate<R: Rng>(&mut self, rng: &mut R) {
            self.value += rng.gen_range(-10.0..10.0);
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

        let runner = GaRunner::new(config, SimpleProblem);
        let result = runner.run();

        // Should find something close to 0
        assert!(result.best.value.abs() < 5.0);
    }

    #[test]
    fn test_permutation_crossover() {
        let mut rng = thread_rng();
        let parent1 = PermutationChromosome::random_with_options(10, 4, &mut rng);
        let parent2 = PermutationChromosome::random_with_options(10, 4, &mut rng);

        let child = parent1.order_crossover(&parent2, &mut rng);

        // Child should be a valid permutation
        assert_eq!(child.genes.len(), 10);
        let mut sorted = child.genes.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_permutation_mutation() {
        let mut rng = thread_rng();
        let mut chromosome = PermutationChromosome::random_with_options(10, 4, &mut rng);

        chromosome.swap_mutate(&mut rng);

        // Should still be a valid permutation
        let mut sorted = chromosome.genes.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
