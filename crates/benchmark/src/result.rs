//! Benchmark result types and recording.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use u_nesting_core::Strategy;

/// Placement info for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementInfo {
    pub geometry_id: String,
    pub position: [f64; 2],
    pub rotation: f64,
    /// Which strip/boundary this placement belongs to (0-indexed)
    pub strip_index: usize,
}

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// Dataset name
    pub dataset: String,
    /// Instance name (if applicable)
    pub instance: String,
    /// Strategy used
    pub strategy: String,
    /// Strip length achieved (lower is better)
    pub strip_length: f64,
    /// Strip height (boundary height)
    pub strip_height: f64,
    /// Number of pieces placed
    pub pieces_placed: usize,
    /// Total pieces in the problem
    pub total_pieces: usize,
    /// Utilization ratio (0.0 - 1.0)
    pub utilization: f64,
    /// Computation time in milliseconds
    pub time_ms: u64,
    /// Number of iterations (if applicable)
    pub iterations: Option<u32>,
    /// Best known solution for comparison
    pub best_known: Option<f64>,
    /// Gap from best known (percentage)
    pub gap_percent: Option<f64>,
    /// Placement coordinates (optional, for visualization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placements: Option<Vec<PlacementInfo>>,
}

impl RunResult {
    /// Creates a new run result.
    pub fn new(
        dataset: String,
        instance: String,
        strategy: Strategy,
        strip_length: f64,
        strip_height: f64,
        pieces_placed: usize,
        total_pieces: usize,
        time_ms: u64,
    ) -> Self {
        let utilization = if strip_length > 0.0 {
            pieces_placed as f64 / total_pieces as f64
        } else {
            0.0
        };

        Self {
            dataset,
            instance,
            strategy: format!("{:?}", strategy),
            strip_length,
            strip_height,
            pieces_placed,
            total_pieces,
            utilization,
            time_ms,
            iterations: None,
            best_known: None,
            gap_percent: None,
            placements: None,
        }
    }

    /// Sets the placements for visualization.
    pub fn with_placements(mut self, placements: Vec<PlacementInfo>) -> Self {
        self.placements = Some(placements);
        self
    }

    /// Sets the best known solution and calculates gap.
    pub fn with_best_known(mut self, best: f64) -> Self {
        self.best_known = Some(best);
        if best > 0.0 && self.strip_length > 0.0 {
            self.gap_percent = Some(((self.strip_length - best) / best) * 100.0);
        }
        self
    }

    /// Sets the iteration count.
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = Some(iterations);
        self
    }
}

/// Collection of benchmark results.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Individual run results
    pub runs: Vec<RunResult>,
    /// Timestamp when the benchmark was run
    pub timestamp: String,
    /// Additional metadata
    pub metadata: BenchmarkMetadata,
}

/// Metadata about the benchmark run.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// U-Nesting version
    pub version: String,
    /// System information
    pub system: String,
    /// Configuration used
    pub config: String,
}

impl BenchmarkResult {
    /// Creates a new benchmark result.
    pub fn new() -> Self {
        let now = chrono_lite::Datetime::now();
        Self {
            runs: Vec::new(),
            timestamp: now.to_string(),
            metadata: BenchmarkMetadata::default(),
        }
    }

    /// Adds a run result.
    pub fn add_run(&mut self, result: RunResult) {
        self.runs.push(result);
    }

    /// Saves results to a JSON file.
    pub fn save_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    /// Saves results to a CSV file.
    pub fn save_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(
            file,
            "dataset,instance,strategy,strip_length,pieces_placed,total_pieces,utilization,time_ms,iterations,best_known,gap_percent"
        )?;

        // Write data
        for run in &self.runs {
            writeln!(
                file,
                "{},{},{},{:.2},{},{},{:.4},{},{},{},{}",
                run.dataset,
                run.instance,
                run.strategy,
                run.strip_length,
                run.pieces_placed,
                run.total_pieces,
                run.utilization,
                run.time_ms,
                run.iterations.map_or(String::new(), |i| i.to_string()),
                run.best_known
                    .map_or(String::new(), |b| format!("{:.2}", b)),
                run.gap_percent
                    .map_or(String::new(), |g| format!("{:.2}", g)),
            )?;
        }

        Ok(())
    }

    /// Prints a summary table to stdout.
    pub fn print_summary(&self) {
        println!("\n{:=<100}", "");
        println!("BENCHMARK RESULTS");
        println!("{:=<100}", "");
        println!(
            "{:<15} {:<15} {:<20} {:>12} {:>10} {:>10} {:>10}",
            "Dataset", "Instance", "Strategy", "Length", "Util%", "Time(ms)", "Gap%"
        );
        println!("{:-<100}", "");

        for run in &self.runs {
            let gap_str = run
                .gap_percent
                .map_or("-".to_string(), |g| format!("{:.1}", g));
            println!(
                "{:<15} {:<15} {:<20} {:>12.2} {:>10.1} {:>10} {:>10}",
                run.dataset,
                run.instance,
                run.strategy,
                run.strip_length,
                run.utilization * 100.0,
                run.time_ms,
                gap_str
            );
        }

        println!("{:=<100}\n", "");
    }

    /// Computes summary statistics grouped by strategy.
    pub fn summary_by_strategy(&self) -> Vec<StrategySummary> {
        use std::collections::HashMap;

        let mut by_strategy: HashMap<String, Vec<&RunResult>> = HashMap::new();
        for run in &self.runs {
            by_strategy
                .entry(run.strategy.clone())
                .or_default()
                .push(run);
        }

        by_strategy
            .into_iter()
            .map(|(strategy, runs)| {
                let n = runs.len() as f64;
                let avg_utilization = runs.iter().map(|r| r.utilization).sum::<f64>() / n;
                let avg_time = runs.iter().map(|r| r.time_ms).sum::<u64>() as f64 / n;
                let avg_gap = {
                    let gaps: Vec<f64> = runs.iter().filter_map(|r| r.gap_percent).collect();
                    if gaps.is_empty() {
                        None
                    } else {
                        Some(gaps.iter().sum::<f64>() / gaps.len() as f64)
                    }
                };

                StrategySummary {
                    strategy,
                    run_count: runs.len(),
                    avg_utilization,
                    avg_time_ms: avg_time as u64,
                    avg_gap_percent: avg_gap,
                }
            })
            .collect()
    }
}

/// Summary statistics for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySummary {
    pub strategy: String,
    pub run_count: usize,
    pub avg_utilization: f64,
    pub avg_time_ms: u64,
    pub avg_gap_percent: Option<f64>,
}

/// Simple datetime implementation (to avoid chrono dependency).
mod chrono_lite {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub struct Datetime {
        year: u32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
    }

    impl Datetime {
        pub fn now() -> Self {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Simple calculation (not accounting for leap years properly)
            let days = secs / 86400;
            let time_of_day = secs % 86400;

            let hour = (time_of_day / 3600) as u32;
            let minute = ((time_of_day % 3600) / 60) as u32;
            let second = (time_of_day % 60) as u32;

            // Approximate year/month/day (good enough for timestamps)
            let mut year = 1970;
            let mut remaining_days = days as i64;

            loop {
                let days_in_year = if year % 4 == 0 { 366 } else { 365 };
                if remaining_days < days_in_year {
                    break;
                }
                remaining_days -= days_in_year;
                year += 1;
            }

            let days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
            let mut month = 1;
            for &days in &days_in_months {
                if remaining_days < days {
                    break;
                }
                remaining_days -= days;
                month += 1;
            }

            let day = remaining_days as u32 + 1;

            Self {
                year,
                month,
                day,
                hour,
                minute,
                second,
            }
        }
    }

    impl std::fmt::Display for Datetime {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                self.year, self.month, self.day, self.hour, self.minute, self.second
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_result() {
        let result = RunResult::new(
            "SHAPES".to_string(),
            "shapes0".to_string(),
            Strategy::BottomLeftFill,
            100.0,
            40.0, // strip_height
            43,
            43,
            1500,
        )
        .with_best_known(90.0);

        assert_eq!(result.dataset, "SHAPES");
        assert!((result.gap_percent.unwrap() - 11.11).abs() < 0.1);
    }
}
