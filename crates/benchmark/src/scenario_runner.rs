//! Scenario runner for executing test scenarios and collecting results.
//!
//! This module provides infrastructure for:
//! - Loading and parsing scenarios
//! - Executing benchmarks according to scenario definitions
//! - Collecting metrics and detecting defects
//! - Generating reports

use crate::dataset::Dataset;
use crate::download::DatasetManager;
use crate::parser::DatasetParser;
use crate::runner::{BenchmarkConfig, BenchmarkRunner};
use crate::scenario::{
    CriterionResult, DatasetRef, Defect, DefectCategory, DefectSeverity, Scenario,
    ScenarioCategory, ScenarioResult, ScenarioRunResult, ScenarioSet, SuccessCriteria,
};
use crate::SyntheticDatasets;
use instant::Instant;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use u_nesting_core::Strategy;

/// Configuration for the scenario runner.
#[derive(Debug, Clone)]
pub struct ScenarioRunnerConfig {
    /// Base directory for datasets
    pub datasets_dir: PathBuf,
    /// Output directory for results
    pub results_dir: PathBuf,
    /// Whether to show progress
    pub verbose: bool,
    /// GA population size
    pub population_size: usize,
    /// GA max generations
    pub max_generations: u32,
    /// Synthetic dataset seed
    pub synthetic_seed: u64,
}

impl Default for ScenarioRunnerConfig {
    fn default() -> Self {
        Self {
            datasets_dir: PathBuf::from("datasets"),
            results_dir: PathBuf::from("benchmark/results"),
            verbose: true,
            population_size: 100,
            max_generations: 500,
            synthetic_seed: 42,
        }
    }
}

/// Runner for executing test scenarios.
pub struct ScenarioRunner {
    config: ScenarioRunnerConfig,
    parser: DatasetParser,
    esicup_manager: DatasetManager,
    synthetic_datasets: HashMap<String, Dataset>,
}

impl ScenarioRunner {
    /// Create a new scenario runner.
    pub fn new(config: ScenarioRunnerConfig) -> Self {
        let esicup_dir = config.datasets_dir.join("2d/esicup");
        let esicup_manager = DatasetManager::new(&esicup_dir);

        // Pre-generate synthetic datasets
        let synthetic_list = SyntheticDatasets::all(config.synthetic_seed);
        let mut synthetic_datasets = HashMap::new();
        for ds in synthetic_list {
            synthetic_datasets.insert(ds.name.clone(), ds);
        }

        Self {
            config,
            parser: DatasetParser::new(),
            esicup_manager,
            synthetic_datasets,
        }
    }

    /// Run all scenarios in a set.
    pub fn run_all(&self, set: &ScenarioSet) -> Vec<ScenarioResult> {
        if self.config.verbose {
            println!("\n{}", "=".repeat(60));
            println!("Running Scenario Set: {}", set.name);
            println!("{}", "=".repeat(60));
            println!("Total scenarios: {}", set.scenarios.len());
        }

        let mut results = Vec::new();
        for scenario in &set.scenarios {
            let result = self.run_scenario(scenario);
            results.push(result);
        }

        results
    }

    /// Run scenarios filtered by category.
    pub fn run_by_category(
        &self,
        set: &ScenarioSet,
        category: ScenarioCategory,
    ) -> Vec<ScenarioResult> {
        let filtered: Vec<_> = set.filter_by_category(category);
        if self.config.verbose {
            println!(
                "\nRunning {:?} scenarios: {} total",
                category,
                filtered.len()
            );
        }

        let mut results = Vec::new();
        for scenario in filtered {
            let result = self.run_scenario(scenario);
            results.push(result);
        }

        results
    }

    /// Run a single scenario.
    pub fn run_scenario(&self, scenario: &Scenario) -> ScenarioResult {
        let start = Instant::now();

        if self.config.verbose {
            println!("\n{}", "-".repeat(50));
            println!("Scenario: {} - {}", scenario.id, scenario.name);
            println!("Purpose: {}", scenario.purpose);
            println!("Datasets: {}", scenario.datasets.len());
            println!("Strategies: {:?}", scenario.strategies);
            println!("{}", "-".repeat(50));
        }

        let mut runs = Vec::new();
        let mut defects = Vec::new();

        // Load and run each dataset
        for dataset_ref in &scenario.datasets {
            match self.load_dataset(dataset_ref) {
                Ok(dataset) => {
                    let dataset_runs = self.run_dataset_scenario(scenario, &dataset);
                    runs.extend(dataset_runs);
                }
                Err(e) => {
                    if self.config.verbose {
                        println!("  [WARN] Failed to load dataset {:?}: {}", dataset_ref, e);
                    }
                    defects.push(Defect {
                        id: format!("{}-LOAD-ERR", scenario.id),
                        category: DefectCategory::Bug,
                        severity: DefectSeverity::P1,
                        title: "Dataset load failure".to_string(),
                        description: format!("Failed to load dataset: {}", e),
                        scenario_id: scenario.id.clone(),
                        dataset: format!("{:?}", dataset_ref),
                        strategy: "N/A".to_string(),
                        evidence: e.to_string(),
                    });
                }
            }
        }

        // Evaluate criteria
        let criteria_results = self.evaluate_criteria(&scenario.criteria, &runs);
        let passed = criteria_results.values().all(|c| c.passed);

        // Detect defects from runs
        let detected_defects = self.detect_defects(scenario, &runs);
        defects.extend(detected_defects);

        let total_time_ms = start.elapsed().as_millis() as u64;

        if self.config.verbose {
            let status = if passed { "PASSED" } else { "FAILED" };
            println!(
                "  Result: {} ({} runs, {} defects, {}ms)",
                status,
                runs.len(),
                defects.len(),
                total_time_ms
            );
        }

        ScenarioResult {
            scenario_id: scenario.id.clone(),
            passed,
            runs,
            criteria_results,
            defects,
            total_time_ms,
        }
    }

    /// Load a dataset based on the reference type.
    fn load_dataset(&self, dataset_ref: &DatasetRef) -> Result<Dataset, String> {
        match dataset_ref {
            DatasetRef::Esicup { dataset, instance } => self
                .esicup_manager
                .load_cached(instance)
                .or_else(|_| self.esicup_manager.download(dataset, instance))
                .map_err(|e| e.to_string()),
            DatasetRef::File { path } => self
                .parser
                .parse_file(Path::new(path))
                .map_err(|e| e.to_string()),
            DatasetRef::Synthetic { synthetic } => self
                .synthetic_datasets
                .get(synthetic)
                .cloned()
                .ok_or_else(|| format!("Unknown synthetic dataset: {}", synthetic)),
            DatasetRef::Mpv { mpv_class } => {
                // Generate MPV instance on demand
                self.generate_mpv_instance(mpv_class)
            }
        }
    }

    /// Generate an MPV 3D instance (converted to 2D-like dataset for now).
    fn generate_mpv_instance(&self, mpv_class: &str) -> Result<Dataset, String> {
        // For now, we'll skip 3D scenarios or return a placeholder
        // Full 3D support would require extending the Dataset type
        Err(format!("3D MPV instances not yet supported: {}", mpv_class))
    }

    /// Run a scenario on a single dataset.
    fn run_dataset_scenario(
        &self,
        scenario: &Scenario,
        dataset: &Dataset,
    ) -> Vec<ScenarioRunResult> {
        let mut results = Vec::new();

        let strategies: Vec<Strategy> = scenario.strategies.iter().map(|s| (*s).into()).collect();

        let config = BenchmarkConfig::new()
            .with_strategies(strategies.clone())
            .with_time_limit(scenario.time_limit_secs * 1000)
            .with_runs_per_config(scenario.runs);

        let runner = BenchmarkRunner::new(BenchmarkConfig {
            population_size: self.config.population_size,
            max_generations: self.config.max_generations,
            show_progress: false,
            ..config
        });

        let benchmark_result = runner.run_dataset(dataset);

        for run in benchmark_result.runs {
            let gap_from_best = dataset.best_known.map(|best| {
                if best > 0.0 {
                    (run.strip_length - best) / best
                } else {
                    0.0
                }
            });

            let placement_ratio = if run.total_pieces > 0 {
                run.pieces_placed as f64 / run.total_pieces as f64
            } else {
                0.0
            };

            results.push(ScenarioRunResult {
                dataset: run.dataset.clone(),
                strategy: run.strategy.clone(),
                run_index: run.instance.parse().unwrap_or(0),
                placement_ratio,
                utilization: run.utilization,
                strip_length: run.strip_length,
                time_ms: run.time_ms,
                placed_count: run.pieces_placed,
                total_count: run.total_pieces,
                error: None,
                gap_from_best,
            });
        }

        results
    }

    /// Evaluate success criteria against run results.
    fn evaluate_criteria(
        &self,
        criteria: &SuccessCriteria,
        runs: &[ScenarioRunResult],
    ) -> HashMap<String, CriterionResult> {
        let mut results = HashMap::new();

        // Check must_not_crash
        let crash_count = runs.iter().filter(|r| r.error.is_some()).count();
        results.insert(
            "must_not_crash".to_string(),
            CriterionResult {
                criterion: "Must not crash".to_string(),
                passed: crash_count == 0 || !criteria.must_not_crash,
                expected: "No crashes".to_string(),
                actual: format!("{} crashes", crash_count),
            },
        );

        // Check min_placement_ratio
        if let Some(min_ratio) = criteria.min_placement_ratio {
            let avg_ratio: f64 = if runs.is_empty() {
                0.0
            } else {
                runs.iter().map(|r| r.placement_ratio).sum::<f64>() / runs.len() as f64
            };
            results.insert(
                "min_placement_ratio".to_string(),
                CriterionResult {
                    criterion: "Min placement ratio".to_string(),
                    passed: avg_ratio >= min_ratio,
                    expected: format!("{:.1}%", min_ratio * 100.0),
                    actual: format!("{:.1}%", avg_ratio * 100.0),
                },
            );
        }

        // Check min_utilization
        if let Some(min_util) = criteria.min_utilization {
            let avg_util: f64 = if runs.is_empty() {
                0.0
            } else {
                runs.iter().map(|r| r.utilization).sum::<f64>() / runs.len() as f64
            };
            results.insert(
                "min_utilization".to_string(),
                CriterionResult {
                    criterion: "Min utilization".to_string(),
                    passed: avg_util >= min_util,
                    expected: format!("{:.1}%", min_util * 100.0),
                    actual: format!("{:.1}%", avg_util * 100.0),
                },
            );
        }

        // Check max_gap_from_best
        if let Some(max_gap) = criteria.max_gap_from_best {
            let gaps: Vec<f64> = runs.iter().filter_map(|r| r.gap_from_best).collect();
            let avg_gap = if gaps.is_empty() {
                0.0
            } else {
                gaps.iter().sum::<f64>() / gaps.len() as f64
            };
            results.insert(
                "max_gap_from_best".to_string(),
                CriterionResult {
                    criterion: "Max gap from best known".to_string(),
                    passed: avg_gap <= max_gap,
                    expected: format!("<= {:.1}%", max_gap * 100.0),
                    actual: format!("{:.1}%", avg_gap * 100.0),
                },
            );
        }

        // Check max_time_ms
        if let Some(max_time) = criteria.max_time_ms {
            let max_observed = runs.iter().map(|r| r.time_ms).max().unwrap_or(0);
            results.insert(
                "max_time_ms".to_string(),
                CriterionResult {
                    criterion: "Max execution time".to_string(),
                    passed: max_observed <= max_time,
                    expected: format!("<= {}ms", max_time),
                    actual: format!("{}ms", max_observed),
                },
            );
        }

        // Check all_items_placed
        if criteria.all_items_placed {
            let all_placed = runs.iter().all(|r| r.placed_count == r.total_count);
            results.insert(
                "all_items_placed".to_string(),
                CriterionResult {
                    criterion: "All items placed".to_string(),
                    passed: all_placed,
                    expected: "100%".to_string(),
                    actual: if all_placed {
                        "100%".to_string()
                    } else {
                        format!(
                            "{}/{} avg",
                            runs.iter().map(|r| r.placed_count).sum::<usize>() / runs.len().max(1),
                            runs.iter().map(|r| r.total_count).sum::<usize>() / runs.len().max(1)
                        )
                    },
                },
            );
        }

        results
    }

    /// Detect defects from run results.
    fn detect_defects(&self, scenario: &Scenario, runs: &[ScenarioRunResult]) -> Vec<Defect> {
        let mut defects = Vec::new();

        for run in runs {
            // Detect crashes
            if let Some(ref error) = run.error {
                defects.push(Defect {
                    id: format!("{}-CRASH-{}", scenario.id, defects.len()),
                    category: DefectCategory::Bug,
                    severity: DefectSeverity::P0,
                    title: format!("Crash in {} on {}", run.strategy, run.dataset),
                    description: error.clone(),
                    scenario_id: scenario.id.clone(),
                    dataset: run.dataset.clone(),
                    strategy: run.strategy.clone(),
                    evidence: format!("Error: {}", error),
                });
            }

            // Detect zero placement
            if run.placed_count == 0 && run.total_count > 0 {
                defects.push(Defect {
                    id: format!("{}-ZERO-{}", scenario.id, defects.len()),
                    category: DefectCategory::Bug,
                    severity: DefectSeverity::P1,
                    title: format!("Zero items placed by {} on {}", run.strategy, run.dataset),
                    description: "Strategy failed to place any items".to_string(),
                    scenario_id: scenario.id.clone(),
                    dataset: run.dataset.clone(),
                    strategy: run.strategy.clone(),
                    evidence: format!("0/{} items placed", run.total_count),
                });
            }

            // Detect very low utilization (< 30%)
            if run.utilization < 0.30 && run.placement_ratio > 0.5 {
                defects.push(Defect {
                    id: format!("{}-LOWUTIL-{}", scenario.id, defects.len()),
                    category: DefectCategory::Quality,
                    severity: DefectSeverity::P2,
                    title: format!(
                        "Low utilization ({:.1}%) by {} on {}",
                        run.utilization * 100.0,
                        run.strategy,
                        run.dataset
                    ),
                    description: "Placement quality is significantly below expected".to_string(),
                    scenario_id: scenario.id.clone(),
                    dataset: run.dataset.clone(),
                    strategy: run.strategy.clone(),
                    evidence: format!(
                        "Utilization: {:.1}%, Placement: {:.1}%",
                        run.utilization * 100.0,
                        run.placement_ratio * 100.0
                    ),
                });
            }

            // Detect BLF outperforming NFP (quality anomaly found in Phase 0.1)
            // This would require cross-run comparison
        }

        // Cross-strategy comparison for quality anomalies
        let by_dataset: HashMap<&str, Vec<&ScenarioRunResult>> =
            runs.iter().fold(HashMap::new(), |mut acc, run| {
                acc.entry(run.dataset.as_str()).or_default().push(run);
                acc
            });

        for (dataset, dataset_runs) in by_dataset {
            // Find BLF and NFP runs
            let blf_runs: Vec<_> = dataset_runs
                .iter()
                .filter(|r| r.strategy.contains("BottomLeftFill"))
                .collect();
            let nfp_runs: Vec<_> = dataset_runs
                .iter()
                .filter(|r| r.strategy.contains("NfpGuided"))
                .collect();

            if !blf_runs.is_empty() && !nfp_runs.is_empty() {
                let blf_avg_len: f64 =
                    blf_runs.iter().map(|r| r.strip_length).sum::<f64>() / blf_runs.len() as f64;
                let nfp_avg_len: f64 =
                    nfp_runs.iter().map(|r| r.strip_length).sum::<f64>() / nfp_runs.len() as f64;

                // BLF shouldn't beat NFP by more than 5%
                if blf_avg_len < nfp_avg_len * 0.95 {
                    let blf_placed = blf_runs.iter().map(|r| r.placement_ratio).sum::<f64>()
                        / blf_runs.len() as f64;
                    let nfp_placed = nfp_runs.iter().map(|r| r.placement_ratio).sum::<f64>()
                        / nfp_runs.len() as f64;

                    // Only flag if both achieved similar placement
                    if (blf_placed - nfp_placed).abs() < 0.1 {
                        defects.push(Defect {
                            id: format!("{}-QUALITY-ANOMALY-{}", scenario.id, defects.len()),
                            category: DefectCategory::Quality,
                            severity: DefectSeverity::P1,
                            title: format!("BLF outperforms NFP on {}", dataset),
                            description: format!(
                                "BLF achieved shorter strip length ({:.2}) than NFP ({:.2}) with similar placement ratio",
                                blf_avg_len, nfp_avg_len
                            ),
                            scenario_id: scenario.id.clone(),
                            dataset: dataset.to_string(),
                            strategy: "BLF vs NFP".to_string(),
                            evidence: format!(
                                "BLF: {:.2} len, {:.1}% placed; NFP: {:.2} len, {:.1}% placed",
                                blf_avg_len,
                                blf_placed * 100.0,
                                nfp_avg_len,
                                nfp_placed * 100.0
                            ),
                        });
                    }
                }
            }
        }

        defects
    }

    /// Generate a summary report.
    pub fn generate_report(&self, results: &[ScenarioResult]) -> ScenarioReport {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        let all_defects: Vec<_> = results.iter().flat_map(|r| r.defects.clone()).collect();

        let defects_by_severity = {
            let mut map = HashMap::new();
            for d in &all_defects {
                *map.entry(d.severity).or_insert(0) += 1;
            }
            map
        };

        let defects_by_category = {
            let mut map = HashMap::new();
            for d in &all_defects {
                *map.entry(d.category).or_insert(0) += 1;
            }
            map
        };

        ScenarioReport {
            total_scenarios: total,
            passed_scenarios: passed,
            failed_scenarios: failed,
            total_defects: all_defects.len(),
            defects_by_severity,
            defects_by_category,
            defects: all_defects,
            results: results.to_vec(),
        }
    }
}

/// Summary report of scenario execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub total_scenarios: usize,
    pub passed_scenarios: usize,
    pub failed_scenarios: usize,
    pub total_defects: usize,
    pub defects_by_severity: HashMap<DefectSeverity, usize>,
    pub defects_by_category: HashMap<DefectCategory, usize>,
    pub defects: Vec<Defect>,
    pub results: Vec<ScenarioResult>,
}

impl ScenarioReport {
    /// Print a summary to stdout.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("SCENARIO EXECUTION REPORT");
        println!("{}", "=".repeat(60));

        println!(
            "\nScenarios: {}/{} passed ({:.1}%)",
            self.passed_scenarios,
            self.total_scenarios,
            self.passed_scenarios as f64 / self.total_scenarios.max(1) as f64 * 100.0
        );

        println!("\nDefects Found: {}", self.total_defects);
        println!("  By Severity:");
        for severity in [
            DefectSeverity::P0,
            DefectSeverity::P1,
            DefectSeverity::P2,
            DefectSeverity::P3,
        ] {
            let count = self.defects_by_severity.get(&severity).unwrap_or(&0);
            println!("    {:?}: {}", severity, count);
        }

        println!("  By Category:");
        for (cat, count) in &self.defects_by_category {
            println!("    {:?}: {}", cat, count);
        }

        if !self.defects.is_empty() {
            println!("\n{}", "-".repeat(40));
            println!("TOP DEFECTS:");
            for (i, defect) in self.defects.iter().take(10).enumerate() {
                println!(
                    "  {}. [{:?}] {:?} - {}",
                    i + 1,
                    defect.severity,
                    defect.category,
                    defect.title
                );
            }
        }

        println!("\n{}", "=".repeat(60));
    }

    /// Save report to JSON.
    pub fn save_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Generate markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# Scenario Execution Report\n\n");

        md.push_str("## Summary\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!(
            "| Scenarios Passed | {}/{} ({:.1}%) |\n",
            self.passed_scenarios,
            self.total_scenarios,
            self.passed_scenarios as f64 / self.total_scenarios.max(1) as f64 * 100.0
        ));
        md.push_str(&format!("| Total Defects | {} |\n", self.total_defects));

        md.push_str("\n## Defects by Severity\n\n");
        md.push_str("| Severity | Count |\n|----------|-------|\n");
        for severity in [
            DefectSeverity::P0,
            DefectSeverity::P1,
            DefectSeverity::P2,
            DefectSeverity::P3,
        ] {
            let count = self.defects_by_severity.get(&severity).unwrap_or(&0);
            md.push_str(&format!("| {:?} | {} |\n", severity, count));
        }

        if !self.defects.is_empty() {
            md.push_str("\n## Defect List\n\n");
            for defect in &self.defects {
                md.push_str(&format!(
                    "### {} [{:?}]\n\n**Severity**: {:?}\n**Category**: {:?}\n\n{}\n\n**Evidence**: {}\n\n---\n\n",
                    defect.title,
                    defect.id,
                    defect.severity,
                    defect.category,
                    defect.description,
                    defect.evidence
                ));
            }
        }

        md
    }
}

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_runner_config() {
        let config = ScenarioRunnerConfig::default();
        assert_eq!(config.population_size, 100);
        assert!(config.verbose);
    }

    #[test]
    fn test_evaluate_criteria_empty() {
        let config = ScenarioRunnerConfig::default();
        let runner = ScenarioRunner::new(config);

        let criteria = SuccessCriteria::default();
        let results = runner.evaluate_criteria(&criteria, &[]);

        assert!(results.get("must_not_crash").unwrap().passed);
    }
}
