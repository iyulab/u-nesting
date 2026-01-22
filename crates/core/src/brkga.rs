//! Biased Random-Key Genetic Algorithm (BRKGA) framework.
//!
//! BRKGA is a variant of genetic algorithms that uses random-key encoding
//! and biased crossover to favor elite parents.
//!
//! # Key Features
//!
//! - **Random-key encoding**: Each gene is a float in [0, 1)
//! - **Biased crossover**: Elite parent is favored during crossover
//! - **Population partitioning**: Elite, non-elite, and mutant subpopulations
//!
//! # References
//!
//! Gonçalves, J. F., & Resende, M. G. (2011). Biased random-key genetic algorithms
//! for combinatorial optimization. Journal of Heuristics, 17(5), 487-525.

use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for BRKGA.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrkgaConfig {
    /// Population size.
    pub population_size: usize,
    /// Maximum number of generations.
    pub max_generations: u32,
    /// Fraction of population that is elite (0.0 - 1.0).
    pub elite_fraction: f64,
    /// Fraction of population that are mutants (0.0 - 1.0).
    pub mutant_fraction: f64,
    /// Probability of inheriting gene from elite parent during crossover.
    pub elite_bias: f64,
    /// Maximum time limit (None = unlimited).
    pub time_limit: Option<Duration>,
    /// Target fitness to stop early (None = run all generations).
    pub target_fitness: Option<f64>,
    /// Stagnation generations before early stop.
    pub stagnation_limit: Option<u32>,
}

impl Default for BrkgaConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 500,
            elite_fraction: 0.2,   // 20% elite
            mutant_fraction: 0.15, // 15% mutants
            elite_bias: 0.7,       // 70% chance to inherit from elite
            time_limit: None,
            target_fitness: None,
            stagnation_limit: Some(50),
        }
    }
}

impl BrkgaConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the population size.
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size.max(4);
        self
    }

    /// Sets the maximum generations.
    pub fn with_max_generations(mut self, gen: u32) -> Self {
        self.max_generations = gen;
        self
    }

    /// Sets the elite fraction.
    pub fn with_elite_fraction(mut self, fraction: f64) -> Self {
        self.elite_fraction = fraction.clamp(0.01, 0.5);
        self
    }

    /// Sets the mutant fraction.
    pub fn with_mutant_fraction(mut self, fraction: f64) -> Self {
        self.mutant_fraction = fraction.clamp(0.0, 0.5);
        self
    }

    /// Sets the elite bias for crossover.
    pub fn with_elite_bias(mut self, bias: f64) -> Self {
        self.elite_bias = bias.clamp(0.5, 1.0);
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

    /// Sets the stagnation limit.
    pub fn with_stagnation_limit(mut self, limit: u32) -> Self {
        self.stagnation_limit = Some(limit);
        self
    }

    /// Returns the number of elite individuals.
    pub fn elite_count(&self) -> usize {
        ((self.population_size as f64) * self.elite_fraction).ceil() as usize
    }

    /// Returns the number of mutant individuals.
    pub fn mutant_count(&self) -> usize {
        ((self.population_size as f64) * self.mutant_fraction).ceil() as usize
    }
}

/// Random-key chromosome for BRKGA.
///
/// Each gene is a floating-point value in [0, 1) that represents a random key.
/// The decoder interprets these keys to construct a solution.
#[derive(Debug, Clone)]
pub struct RandomKeyChromosome {
    /// Random keys in [0, 1).
    pub keys: Vec<f64>,
    /// Cached fitness value.
    fitness: f64,
}

impl RandomKeyChromosome {
    /// Creates a new chromosome with the given number of keys.
    pub fn new(num_keys: usize) -> Self {
        Self {
            keys: vec![0.0; num_keys],
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Creates a random chromosome.
    pub fn random<R: Rng>(num_keys: usize, rng: &mut R) -> Self {
        let keys: Vec<f64> = (0..num_keys).map(|_| rng.gen::<f64>()).collect();
        Self {
            keys,
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Returns the number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Returns the fitness value.
    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    /// Sets the fitness value.
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    /// Biased crossover with another chromosome.
    ///
    /// For each gene, there's an `elite_bias` probability of inheriting
    /// from `self` (the elite parent) and `1 - elite_bias` from `other`.
    pub fn biased_crossover<R: Rng>(&self, other: &Self, elite_bias: f64, rng: &mut R) -> Self {
        let keys: Vec<f64> = self
            .keys
            .iter()
            .zip(&other.keys)
            .map(|(&elite_key, &non_elite_key)| {
                if rng.gen::<f64>() < elite_bias {
                    elite_key
                } else {
                    non_elite_key
                }
            })
            .collect();

        Self {
            keys,
            fitness: f64::NEG_INFINITY,
        }
    }

    /// Decodes the random keys into a permutation (sorted indices by key value).
    ///
    /// This is the most common decoding: sort items by their random keys
    /// to get a placement order.
    pub fn decode_as_permutation(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.keys.len()).collect();
        indices.sort_by(|&a, &b| {
            self.keys[a]
                .partial_cmp(&self.keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Decodes a subset of keys as discrete values in a range.
    ///
    /// For example, to decode orientation (0-5), use:
    /// `decode_as_discrete(key_idx, 6)` which maps [0, 1) to {0, 1, 2, 3, 4, 5}.
    pub fn decode_as_discrete(&self, key_idx: usize, num_options: usize) -> usize {
        if key_idx >= self.keys.len() || num_options == 0 {
            return 0;
        }
        let key = self.keys[key_idx].clamp(0.0, 0.9999999);
        (key * num_options as f64) as usize
    }
}

/// Trait for BRKGA problem-specific operations.
pub trait BrkgaProblem: Send + Sync {
    /// Returns the number of random keys needed for one solution.
    fn num_keys(&self) -> usize;

    /// Evaluates the fitness of a chromosome and updates its fitness value.
    fn evaluate(&self, chromosome: &mut RandomKeyChromosome);

    /// Evaluates multiple chromosomes in parallel.
    /// Default implementation uses rayon for parallel evaluation.
    fn evaluate_parallel(&self, chromosomes: &mut [RandomKeyChromosome]) {
        chromosomes.par_iter_mut().for_each(|c| {
            self.evaluate(c);
        });
    }

    /// Called after each generation (for progress reporting).
    fn on_generation(
        &self,
        _generation: u32,
        _best: &RandomKeyChromosome,
        _population: &[RandomKeyChromosome],
    ) {
        // Default: do nothing
    }
}

/// Result of a BRKGA run.
#[derive(Debug, Clone)]
pub struct BrkgaResult {
    /// The best chromosome found.
    pub best: RandomKeyChromosome,
    /// Final generation reached.
    pub generations: u32,
    /// Total elapsed time.
    pub elapsed: Duration,
    /// Whether the target fitness was reached.
    pub target_reached: bool,
    /// Fitness history (best fitness per generation).
    pub history: Vec<f64>,
}

/// Progress information during BRKGA execution.
#[derive(Debug, Clone)]
pub struct BrkgaProgress {
    /// Current generation number.
    pub generation: u32,
    /// Maximum generations configured.
    pub max_generations: u32,
    /// Best fitness so far.
    pub best_fitness: f64,
    /// Average fitness of current population.
    pub avg_fitness: f64,
    /// Elapsed time since start.
    pub elapsed: Duration,
    /// Whether the algorithm is still running.
    pub running: bool,
}

/// BRKGA runner.
pub struct BrkgaRunner<P: BrkgaProblem> {
    config: BrkgaConfig,
    problem: P,
    cancelled: Arc<AtomicBool>,
}

impl<P: BrkgaProblem> BrkgaRunner<P> {
    /// Creates a new BRKGA runner.
    pub fn new(config: BrkgaConfig, problem: P) -> Self {
        Self {
            config,
            problem,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a runner with a pre-existing cancellation handle.
    pub fn with_cancellation(config: BrkgaConfig, problem: P, cancelled: Arc<AtomicBool>) -> Self {
        Self {
            config,
            problem,
            cancelled,
        }
    }

    /// Returns a handle to cancel the algorithm.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    /// Runs the BRKGA algorithm.
    pub fn run(&self) -> BrkgaResult {
        self.run_with_rng(&mut thread_rng())
    }

    /// Runs the BRKGA algorithm with a progress callback.
    pub fn run_with_progress<F>(&self, progress_callback: F) -> BrkgaResult
    where
        F: Fn(BrkgaProgress),
    {
        self.run_with_rng_and_progress(&mut thread_rng(), Some(progress_callback))
    }

    /// Runs the BRKGA algorithm with a specific RNG.
    pub fn run_with_rng<R: Rng>(&self, rng: &mut R) -> BrkgaResult {
        self.run_with_rng_and_progress::<R, fn(BrkgaProgress)>(rng, None)
    }

    /// Runs the BRKGA algorithm with a specific RNG and optional progress callback.
    pub fn run_with_rng_and_progress<R: Rng, F>(
        &self,
        rng: &mut R,
        progress_callback: Option<F>,
    ) -> BrkgaResult
    where
        F: Fn(BrkgaProgress),
    {
        let start = Instant::now();
        let mut history = Vec::new();
        let num_keys = self.problem.num_keys();

        // Initialize population with random chromosomes
        let mut population: Vec<RandomKeyChromosome> = (0..self.config.population_size)
            .map(|_| RandomKeyChromosome::random(num_keys, rng))
            .collect();

        // Evaluate initial population in parallel
        self.problem.evaluate_parallel(&mut population);

        // Sort by fitness (descending - higher is better)
        population.sort_by(|a, b| {
            b.fitness()
                .partial_cmp(&a.fitness())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut best = population[0].clone();
        let mut best_fitness = best.fitness();
        let mut stagnation_count = 0u32;
        let mut generation = 0u32;
        let mut target_reached = false;

        let elite_count = self.config.elite_count();
        let mutant_count = self.config.mutant_count();

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

            // 1. Copy elite individuals directly
            for elite in population.iter().take(elite_count) {
                new_population.push(elite.clone());
            }

            // 2. Generate mutants (completely random new individuals)
            let mut mutants: Vec<RandomKeyChromosome> = (0..mutant_count)
                .map(|_| RandomKeyChromosome::random(num_keys, rng))
                .collect();

            // 3. Fill the rest with biased crossover
            let crossover_count = self.config.population_size - elite_count - mutant_count;
            let mut children: Vec<RandomKeyChromosome> = (0..crossover_count)
                .map(|_| {
                    // Select one elite parent
                    let elite_idx = rng.gen_range(0..elite_count);
                    let elite_parent = &population[elite_idx];

                    // Select one non-elite parent
                    let non_elite_idx = rng.gen_range(elite_count..population.len());
                    let non_elite_parent = &population[non_elite_idx];

                    // Biased crossover
                    elite_parent.biased_crossover(non_elite_parent, self.config.elite_bias, rng)
                })
                .collect();

            // Evaluate all new individuals (mutants + children) in parallel
            self.problem.evaluate_parallel(&mut mutants);
            self.problem.evaluate_parallel(&mut children);

            // Add to new population
            new_population.extend(mutants);
            new_population.extend(children);

            // Sort new population
            new_population.sort_by(|a, b| {
                b.fitness()
                    .partial_cmp(&a.fitness())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Update best
            let new_best_fitness = new_population[0].fitness();
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

            // Callback to BrkgaProblem
            self.problem
                .on_generation(generation, &best, &new_population);

            // Progress callback
            if let Some(ref callback) = progress_callback {
                let avg_fitness = new_population.iter().map(|c| c.fitness()).sum::<f64>()
                    / new_population.len() as f64;

                callback(BrkgaProgress {
                    generation,
                    max_generations: self.config.max_generations,
                    best_fitness,
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
            let avg_fitness = population.iter().map(|c| c.fitness()).sum::<f64>()
                / population.len().max(1) as f64;

            callback(BrkgaProgress {
                generation,
                max_generations: self.config.max_generations,
                best_fitness,
                avg_fitness,
                elapsed: start.elapsed(),
                running: false,
            });
        }

        BrkgaResult {
            best,
            generations: generation,
            elapsed: start.elapsed(),
            target_reached,
            history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple problem: maximize sum of keys (trivially, all keys should be close to 1)
    struct MaxSumProblem {
        num_keys: usize,
    }

    impl BrkgaProblem for MaxSumProblem {
        fn num_keys(&self) -> usize {
            self.num_keys
        }

        fn evaluate(&self, chromosome: &mut RandomKeyChromosome) {
            let sum: f64 = chromosome.keys.iter().sum();
            chromosome.set_fitness(sum);
        }
    }

    #[test]
    fn test_brkga_basic() {
        let config = BrkgaConfig::default()
            .with_population_size(50)
            .with_max_generations(50);

        let problem = MaxSumProblem { num_keys: 10 };
        let runner = BrkgaRunner::new(config, problem);
        let result = runner.run();

        // Best should have keys close to 1.0, sum close to 10.0
        assert!(result.best.fitness() > 5.0);
    }

    #[test]
    fn test_random_key_chromosome() {
        let mut rng = thread_rng();
        let chromosome = RandomKeyChromosome::random(10, &mut rng);

        assert_eq!(chromosome.len(), 10);
        for &key in &chromosome.keys {
            assert!(key >= 0.0 && key < 1.0);
        }
    }

    #[test]
    fn test_biased_crossover() {
        let mut rng = thread_rng();
        let elite = RandomKeyChromosome::random(10, &mut rng);
        let non_elite = RandomKeyChromosome::random(10, &mut rng);

        let child = elite.biased_crossover(&non_elite, 0.7, &mut rng);

        assert_eq!(child.len(), 10);
        for &key in &child.keys {
            assert!(key >= 0.0 && key < 1.0);
        }
    }

    #[test]
    fn test_decode_as_permutation() {
        let mut chromosome = RandomKeyChromosome::new(5);
        chromosome.keys = vec![0.3, 0.1, 0.9, 0.5, 0.2];

        let perm = chromosome.decode_as_permutation();

        // Sorted order: 0.1(1), 0.2(4), 0.3(0), 0.5(3), 0.9(2)
        assert_eq!(perm, vec![1, 4, 0, 3, 2]);
    }

    #[test]
    fn test_decode_as_discrete() {
        let mut chromosome = RandomKeyChromosome::new(3);
        chromosome.keys = vec![0.0, 0.5, 0.99];

        // 6 options: 0.0 -> 0, 0.5 -> 3, 0.99 -> 5
        assert_eq!(chromosome.decode_as_discrete(0, 6), 0);
        assert_eq!(chromosome.decode_as_discrete(1, 6), 3);
        assert_eq!(chromosome.decode_as_discrete(2, 6), 5);
    }
}
