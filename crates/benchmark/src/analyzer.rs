//! Benchmark result analysis and report generation.
//!
//! This module provides tools for analyzing benchmark results and generating
//! comparison reports in various formats (Markdown, JSON, CSV).

use crate::result::{BenchmarkResult, RunResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Analysis report for benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Report title
    pub title: String,
    /// Report timestamp
    pub timestamp: String,
    /// Overall statistics
    pub overall: OverallStats,
    /// Per-strategy analysis
    pub by_strategy: Vec<StrategyAnalysis>,
    /// Per-dataset analysis
    pub by_dataset: Vec<DatasetAnalysis>,
    /// Strategy comparison matrix
    pub comparison: StrategyComparison,
    /// Performance rankings
    pub rankings: Rankings,
}

/// Overall statistics across all runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverallStats {
    /// Total number of benchmark runs
    pub total_runs: usize,
    /// Number of datasets tested
    pub datasets_count: usize,
    /// Number of strategies tested
    pub strategies_count: usize,
    /// Average utilization across all runs
    pub avg_utilization: f64,
    /// Best utilization achieved
    pub best_utilization: f64,
    /// Worst utilization achieved
    pub worst_utilization: f64,
    /// Average computation time (ms)
    pub avg_time_ms: u64,
    /// Total computation time (ms)
    pub total_time_ms: u64,
    /// Number of runs with known best solutions
    pub runs_with_best_known: usize,
    /// Average gap from best known (%)
    pub avg_gap_percent: Option<f64>,
}

/// Detailed analysis for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAnalysis {
    /// Strategy name
    pub strategy: String,
    /// Number of runs
    pub run_count: usize,
    /// Average utilization
    pub avg_utilization: f64,
    /// Standard deviation of utilization
    pub std_utilization: f64,
    /// Minimum utilization
    pub min_utilization: f64,
    /// Maximum utilization
    pub max_utilization: f64,
    /// Average time (ms)
    pub avg_time_ms: u64,
    /// Minimum time (ms)
    pub min_time_ms: u64,
    /// Maximum time (ms)
    pub max_time_ms: u64,
    /// Average gap from best known (%)
    pub avg_gap_percent: Option<f64>,
    /// Number of times this strategy achieved the best result
    pub wins: usize,
}

/// Analysis for a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetAnalysis {
    /// Dataset name
    pub dataset: String,
    /// Number of instances
    pub instance_count: usize,
    /// Best strategy for this dataset
    pub best_strategy: String,
    /// Best utilization achieved
    pub best_utilization: f64,
    /// Best known solution (if available)
    pub best_known: Option<f64>,
    /// Results by strategy
    pub by_strategy: HashMap<String, DatasetStrategyResult>,
}

/// Result for a specific dataset-strategy combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStrategyResult {
    /// Average utilization
    pub avg_utilization: f64,
    /// Average time (ms)
    pub avg_time_ms: u64,
    /// Average gap (%)
    pub avg_gap_percent: Option<f64>,
}

/// Strategy comparison matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StrategyComparison {
    /// Strategies being compared
    pub strategies: Vec<String>,
    /// Win matrix: strategies\[i\] beats strategies\[j\] how many times
    pub win_matrix: Vec<Vec<usize>>,
    /// Average improvement percentages
    pub improvement_matrix: Vec<Vec<Option<f64>>>,
}

/// Performance rankings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Rankings {
    /// Strategies ranked by average utilization
    pub by_utilization: Vec<RankEntry>,
    /// Strategies ranked by average speed
    pub by_speed: Vec<RankEntry>,
    /// Strategies ranked by consistency (lower std is better)
    pub by_consistency: Vec<RankEntry>,
    /// Strategies ranked by win count
    pub by_wins: Vec<RankEntry>,
}

/// A ranking entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankEntry {
    /// Rank (1-based)
    pub rank: usize,
    /// Strategy name
    pub strategy: String,
    /// Score/value for this ranking
    pub value: f64,
}

/// Benchmark result analyzer.
#[derive(Debug, Default)]
pub struct Analyzer {
    results: Vec<BenchmarkResult>,
}

impl Analyzer {
    /// Creates a new analyzer.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Adds benchmark results to analyze.
    pub fn add_results(&mut self, results: BenchmarkResult) {
        self.results.push(results);
    }

    /// Loads results from a JSON file.
    pub fn load_json(&mut self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let content = fs::read_to_string(path)?;
        let results: BenchmarkResult = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.results.push(results);
        Ok(())
    }

    /// Returns all runs from all loaded results.
    fn all_runs(&self) -> Vec<&RunResult> {
        self.results.iter().flat_map(|r| r.runs.iter()).collect()
    }

    /// Generates an analysis report.
    pub fn analyze(&self) -> AnalysisReport {
        let all_runs = self.all_runs();
        let timestamp = self
            .results
            .first()
            .map(|r| r.timestamp.clone())
            .unwrap_or_default();

        let overall = self.compute_overall_stats(&all_runs);
        let by_strategy = self.analyze_by_strategy(&all_runs);
        let by_dataset = self.analyze_by_dataset(&all_runs);
        let comparison = self.compute_comparison(&all_runs, &by_strategy);
        let rankings = self.compute_rankings(&by_strategy);

        AnalysisReport {
            title: "U-Nesting Benchmark Analysis Report".to_string(),
            timestamp,
            overall,
            by_strategy,
            by_dataset,
            comparison,
            rankings,
        }
    }

    fn compute_overall_stats(&self, runs: &[&RunResult]) -> OverallStats {
        if runs.is_empty() {
            return OverallStats::default();
        }

        let total_runs = runs.len();
        let datasets: std::collections::HashSet<_> = runs.iter().map(|r| &r.dataset).collect();
        let strategies: std::collections::HashSet<_> = runs.iter().map(|r| &r.strategy).collect();

        let utilizations: Vec<f64> = runs.iter().map(|r| r.utilization).collect();
        let times: Vec<u64> = runs.iter().map(|r| r.time_ms).collect();
        let gaps: Vec<f64> = runs.iter().filter_map(|r| r.gap_percent).collect();

        let avg_utilization = utilizations.iter().sum::<f64>() / total_runs as f64;
        let best_utilization = utilizations.iter().cloned().fold(0.0, f64::max);
        let worst_utilization = utilizations.iter().cloned().fold(1.0, f64::min);
        let total_time_ms = times.iter().sum();
        let avg_time_ms = total_time_ms / total_runs as u64;

        let avg_gap_percent = if gaps.is_empty() {
            None
        } else {
            Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
        };

        OverallStats {
            total_runs,
            datasets_count: datasets.len(),
            strategies_count: strategies.len(),
            avg_utilization,
            best_utilization,
            worst_utilization,
            avg_time_ms,
            total_time_ms,
            runs_with_best_known: gaps.len(),
            avg_gap_percent,
        }
    }

    fn analyze_by_strategy(&self, runs: &[&RunResult]) -> Vec<StrategyAnalysis> {
        let mut by_strategy: HashMap<String, Vec<&RunResult>> = HashMap::new();
        for run in runs {
            by_strategy
                .entry(run.strategy.clone())
                .or_default()
                .push(run);
        }

        // Compute wins (best utilization per dataset-instance)
        let mut wins: HashMap<String, usize> = HashMap::new();
        let mut by_instance: HashMap<(String, String), Vec<&RunResult>> = HashMap::new();
        for run in runs {
            by_instance
                .entry((run.dataset.clone(), run.instance.clone()))
                .or_default()
                .push(run);
        }
        for (_, instance_runs) in by_instance.iter() {
            if let Some(best) = instance_runs.iter().max_by(|a, b| {
                a.utilization
                    .partial_cmp(&b.utilization)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                *wins.entry(best.strategy.clone()).or_default() += 1;
            }
        }

        by_strategy
            .into_iter()
            .map(|(strategy, strategy_runs)| {
                let n = strategy_runs.len();
                let utilizations: Vec<f64> = strategy_runs.iter().map(|r| r.utilization).collect();
                let times: Vec<u64> = strategy_runs.iter().map(|r| r.time_ms).collect();
                let gaps: Vec<f64> = strategy_runs.iter().filter_map(|r| r.gap_percent).collect();

                let avg_utilization = utilizations.iter().sum::<f64>() / n as f64;
                let min_utilization = utilizations.iter().cloned().fold(1.0, f64::min);
                let max_utilization = utilizations.iter().cloned().fold(0.0, f64::max);

                // Standard deviation
                let variance = utilizations
                    .iter()
                    .map(|u| (u - avg_utilization).powi(2))
                    .sum::<f64>()
                    / n as f64;
                let std_utilization = variance.sqrt();

                let avg_time_ms = times.iter().sum::<u64>() / n as u64;
                let min_time_ms = *times.iter().min().unwrap_or(&0);
                let max_time_ms = *times.iter().max().unwrap_or(&0);

                let avg_gap_percent = if gaps.is_empty() {
                    None
                } else {
                    Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
                };

                StrategyAnalysis {
                    strategy: strategy.clone(),
                    run_count: n,
                    avg_utilization,
                    std_utilization,
                    min_utilization,
                    max_utilization,
                    avg_time_ms,
                    min_time_ms,
                    max_time_ms,
                    avg_gap_percent,
                    wins: *wins.get(&strategy).unwrap_or(&0),
                }
            })
            .collect()
    }

    fn analyze_by_dataset(&self, runs: &[&RunResult]) -> Vec<DatasetAnalysis> {
        let mut by_dataset: HashMap<String, Vec<&RunResult>> = HashMap::new();
        for run in runs {
            by_dataset.entry(run.dataset.clone()).or_default().push(run);
        }

        by_dataset
            .into_iter()
            .map(|(dataset, dataset_runs)| {
                let instances: std::collections::HashSet<_> =
                    dataset_runs.iter().map(|r| &r.instance).collect();

                // Group by strategy
                let mut by_strategy: HashMap<String, Vec<&RunResult>> = HashMap::new();
                for run in &dataset_runs {
                    by_strategy
                        .entry(run.strategy.clone())
                        .or_default()
                        .push(run);
                }

                let strategy_results: HashMap<String, DatasetStrategyResult> = by_strategy
                    .iter()
                    .map(|(strategy, strat_runs)| {
                        let n = strat_runs.len();
                        let avg_utilization =
                            strat_runs.iter().map(|r| r.utilization).sum::<f64>() / n as f64;
                        let avg_time_ms =
                            strat_runs.iter().map(|r| r.time_ms).sum::<u64>() / n as u64;
                        let gaps: Vec<f64> =
                            strat_runs.iter().filter_map(|r| r.gap_percent).collect();
                        let avg_gap_percent = if gaps.is_empty() {
                            None
                        } else {
                            Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
                        };

                        (
                            strategy.clone(),
                            DatasetStrategyResult {
                                avg_utilization,
                                avg_time_ms,
                                avg_gap_percent,
                            },
                        )
                    })
                    .collect();

                // Find best strategy
                let (best_strategy, best_util) = strategy_results
                    .iter()
                    .max_by(|a, b| {
                        a.1.avg_utilization
                            .partial_cmp(&b.1.avg_utilization)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(s, r)| (s.clone(), r.avg_utilization))
                    .unwrap_or_default();

                // Best known
                let best_known = dataset_runs.iter().filter_map(|r| r.best_known).next();

                DatasetAnalysis {
                    dataset,
                    instance_count: instances.len(),
                    best_strategy,
                    best_utilization: best_util,
                    best_known,
                    by_strategy: strategy_results,
                }
            })
            .collect()
    }

    fn compute_comparison(
        &self,
        runs: &[&RunResult],
        by_strategy: &[StrategyAnalysis],
    ) -> StrategyComparison {
        let strategies: Vec<String> = by_strategy.iter().map(|s| s.strategy.clone()).collect();
        let n = strategies.len();

        if n == 0 {
            return StrategyComparison::default();
        }

        // Group runs by instance
        let mut by_instance: HashMap<(String, String), HashMap<String, &RunResult>> =
            HashMap::new();
        for run in runs {
            by_instance
                .entry((run.dataset.clone(), run.instance.clone()))
                .or_default()
                .insert(run.strategy.clone(), run);
        }

        let mut win_matrix = vec![vec![0usize; n]; n];
        let mut improvement_sums = vec![vec![0.0f64; n]; n];
        let mut improvement_counts = vec![vec![0usize; n]; n];

        for (_, instance_runs) in by_instance.iter() {
            for (i, si) in strategies.iter().enumerate() {
                for (j, sj) in strategies.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    if let (Some(ri), Some(rj)) = (instance_runs.get(si), instance_runs.get(sj)) {
                        if ri.utilization > rj.utilization {
                            win_matrix[i][j] += 1;
                        }
                        if rj.utilization > 0.0 {
                            let improvement =
                                (ri.utilization - rj.utilization) / rj.utilization * 100.0;
                            improvement_sums[i][j] += improvement;
                            improvement_counts[i][j] += 1;
                        }
                    }
                }
            }
        }

        let improvement_matrix: Vec<Vec<Option<f64>>> = improvement_sums
            .iter()
            .zip(improvement_counts.iter())
            .map(|(sums, counts)| {
                sums.iter()
                    .zip(counts.iter())
                    .map(|(&sum, &count)| {
                        if count > 0 {
                            Some(sum / count as f64)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        StrategyComparison {
            strategies,
            win_matrix,
            improvement_matrix,
        }
    }

    fn compute_rankings(&self, by_strategy: &[StrategyAnalysis]) -> Rankings {
        let mut by_utilization: Vec<_> = by_strategy
            .iter()
            .map(|s| (s.strategy.clone(), s.avg_utilization))
            .collect();
        by_utilization.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut by_speed: Vec<_> = by_strategy
            .iter()
            .map(|s| (s.strategy.clone(), s.avg_time_ms as f64))
            .collect();
        by_speed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut by_consistency: Vec<_> = by_strategy
            .iter()
            .map(|s| (s.strategy.clone(), s.std_utilization))
            .collect();
        by_consistency.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut by_wins: Vec<_> = by_strategy
            .iter()
            .map(|s| (s.strategy.clone(), s.wins as f64))
            .collect();
        by_wins.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_rank_entries = |items: Vec<(String, f64)>| -> Vec<RankEntry> {
            items
                .into_iter()
                .enumerate()
                .map(|(i, (strategy, value))| RankEntry {
                    rank: i + 1,
                    strategy,
                    value,
                })
                .collect()
        };

        Rankings {
            by_utilization: to_rank_entries(by_utilization),
            by_speed: to_rank_entries(by_speed),
            by_consistency: to_rank_entries(by_consistency),
            by_wins: to_rank_entries(by_wins),
        }
    }
}

/// Report generator for various output formats.
pub struct ReportGenerator;

impl ReportGenerator {
    /// Generates a Markdown report.
    pub fn to_markdown(report: &AnalysisReport) -> String {
        let mut md = String::new();

        // Title
        md.push_str(&format!("# {}\n\n", report.title));
        md.push_str(&format!("*Generated: {}*\n\n", report.timestamp));

        // Overall Statistics
        md.push_str("## Overall Statistics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total Runs | {} |\n", report.overall.total_runs));
        md.push_str(&format!(
            "| Datasets | {} |\n",
            report.overall.datasets_count
        ));
        md.push_str(&format!(
            "| Strategies | {} |\n",
            report.overall.strategies_count
        ));
        md.push_str(&format!(
            "| Avg Utilization | {:.1}% |\n",
            report.overall.avg_utilization * 100.0
        ));
        md.push_str(&format!(
            "| Best Utilization | {:.1}% |\n",
            report.overall.best_utilization * 100.0
        ));
        md.push_str(&format!(
            "| Avg Time | {} ms |\n",
            report.overall.avg_time_ms
        ));
        if let Some(gap) = report.overall.avg_gap_percent {
            md.push_str(&format!("| Avg Gap from Best Known | {:.2}% |\n", gap));
        }
        md.push('\n');

        // Strategy Analysis
        md.push_str("## Strategy Analysis\n\n");
        md.push_str("| Strategy | Runs | Avg Util | Std | Min | Max | Avg Time | Wins |\n");
        md.push_str("|----------|------|----------|-----|-----|-----|----------|------|\n");
        for s in &report.by_strategy {
            md.push_str(&format!(
                "| {} | {} | {:.1}% | {:.2}% | {:.1}% | {:.1}% | {} ms | {} |\n",
                s.strategy,
                s.run_count,
                s.avg_utilization * 100.0,
                s.std_utilization * 100.0,
                s.min_utilization * 100.0,
                s.max_utilization * 100.0,
                s.avg_time_ms,
                s.wins
            ));
        }
        md.push('\n');

        // Rankings
        md.push_str("## Rankings\n\n");

        md.push_str("### By Utilization\n\n");
        md.push_str("| Rank | Strategy | Avg Utilization |\n");
        md.push_str("|------|----------|----------------|\n");
        for r in &report.rankings.by_utilization {
            md.push_str(&format!(
                "| {} | {} | {:.1}% |\n",
                r.rank,
                r.strategy,
                r.value * 100.0
            ));
        }
        md.push('\n');

        md.push_str("### By Speed\n\n");
        md.push_str("| Rank | Strategy | Avg Time (ms) |\n");
        md.push_str("|------|----------|---------------|\n");
        for r in &report.rankings.by_speed {
            md.push_str(&format!(
                "| {} | {} | {:.0} |\n",
                r.rank, r.strategy, r.value
            ));
        }
        md.push('\n');

        md.push_str("### By Consistency\n\n");
        md.push_str("| Rank | Strategy | Std Deviation |\n");
        md.push_str("|------|----------|---------------|\n");
        for r in &report.rankings.by_consistency {
            md.push_str(&format!(
                "| {} | {} | {:.2}% |\n",
                r.rank,
                r.strategy,
                r.value * 100.0
            ));
        }
        md.push('\n');

        // Dataset Analysis
        md.push_str("## Dataset Analysis\n\n");
        md.push_str("| Dataset | Instances | Best Strategy | Best Util |\n");
        md.push_str("|---------|-----------|---------------|----------|\n");
        for d in &report.by_dataset {
            md.push_str(&format!(
                "| {} | {} | {} | {:.1}% |\n",
                d.dataset,
                d.instance_count,
                d.best_strategy,
                d.best_utilization * 100.0
            ));
        }
        md.push('\n');

        // Strategy Comparison
        if !report.comparison.strategies.is_empty() {
            md.push_str("## Strategy Comparison (Win Matrix)\n\n");
            md.push_str("*Cell [i,j] shows how many times strategy i beats strategy j*\n\n");

            md.push_str("| vs |");
            for s in &report.comparison.strategies {
                md.push_str(&format!(" {} |", s));
            }
            md.push_str("\n|");
            for _ in 0..=report.comparison.strategies.len() {
                md.push_str("----|");
            }
            md.push('\n');

            for (i, si) in report.comparison.strategies.iter().enumerate() {
                md.push_str(&format!("| {} |", si));
                for j in 0..report.comparison.strategies.len() {
                    if i == j {
                        md.push_str(" - |");
                    } else {
                        md.push_str(&format!(" {} |", report.comparison.win_matrix[i][j]));
                    }
                }
                md.push('\n');
            }
        }

        md
    }

    /// Saves the report as JSON.
    pub fn save_json(report: &AnalysisReport, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Saves the report as Markdown.
    pub fn save_markdown(report: &AnalysisReport, path: impl AsRef<Path>) -> std::io::Result<()> {
        let md = Self::to_markdown(report);
        fs::write(path, md)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::RunResult;
    use u_nesting_core::Strategy;

    fn create_test_results() -> BenchmarkResult {
        let mut result = BenchmarkResult::new();
        result.add_run(RunResult::new(
            "SHAPES".into(),
            "shapes0".into(),
            Strategy::BottomLeftFill,
            100.0,
            40.0, // strip_height
            40,
            43,
            500,
        ));
        result.add_run(RunResult::new(
            "SHAPES".into(),
            "shapes0".into(),
            Strategy::NfpGuided,
            95.0,
            40.0, // strip_height
            43,
            43,
            1500,
        ));
        result.add_run(RunResult::new(
            "SHAPES".into(),
            "shapes0".into(),
            Strategy::GeneticAlgorithm,
            90.0,
            40.0, // strip_height
            43,
            43,
            5000,
        ));
        result
    }

    #[test]
    fn test_analyzer_overall_stats() {
        let mut analyzer = Analyzer::new();
        analyzer.add_results(create_test_results());

        let report = analyzer.analyze();

        assert_eq!(report.overall.total_runs, 3);
        assert_eq!(report.overall.strategies_count, 3);
        assert_eq!(report.overall.datasets_count, 1);
    }

    #[test]
    fn test_analyzer_strategy_analysis() {
        let mut analyzer = Analyzer::new();
        analyzer.add_results(create_test_results());

        let report = analyzer.analyze();

        assert_eq!(report.by_strategy.len(), 3);
    }

    #[test]
    fn test_markdown_generation() {
        let mut analyzer = Analyzer::new();
        analyzer.add_results(create_test_results());

        let report = analyzer.analyze();
        let md = ReportGenerator::to_markdown(&report);

        assert!(md.contains("# U-Nesting Benchmark Analysis Report"));
        assert!(md.contains("## Strategy Analysis"));
        assert!(md.contains("BottomLeftFill"));
    }
}
