//! Benchmark Suite for U-Nesting
//!
//! This crate provides:
//! - ESICUP dataset parser for 2D irregular nesting problems
//! - Dataset download and management
//! - Synthetic dataset generation for edge case testing
//! - MPV (Martello-Pisinger-Vigo) instance generator for 3D bin packing
//! - Benchmark runner with multiple strategies
//! - Result recording and comparison
//! - Result analysis and report generation

mod analyzer;
mod dataset;
mod dataset3d;
mod download;
mod parser;
mod result;
mod runner;
mod runner3d;
mod scenario;
mod scenario_runner;
mod synthetic;

// Analysis exports
pub use analyzer::{
    AnalysisReport, Analyzer, DatasetAnalysis, OverallStats, RankEntry, Rankings, ReportGenerator,
    StrategyAnalysis, StrategyComparison,
};

// 2D exports
pub use dataset::{Dataset, DatasetInfo, Item, Shape};
pub use download::{
    DatasetInfo as DownloadDatasetInfo, DatasetManager, DownloadError, ESICUP_DATASETS,
};
pub use parser::DatasetParser;
pub use result::{BenchmarkResult, RunResult, StrategySummary};
pub use runner::{BenchmarkConfig, BenchmarkRunner};
pub use synthetic::{SyntheticDatasets, SyntheticGenerator};

// 3D exports
pub use dataset3d::{Dataset3D, Dataset3DInfo, InstanceClass, InstanceGenerator, Item3D};
pub use runner3d::{BenchmarkConfig3D, BenchmarkRunner3D, BenchmarkSummary3D};

// Scenario exports
pub use scenario::{
    create_phase0_scenarios, CriterionResult, DatasetRef, Defect, DefectCategory, DefectSeverity,
    Scenario, ScenarioCategory, ScenarioResult, ScenarioRunResult, ScenarioSet, StrategyRef,
    SuccessCriteria,
};
pub use scenario_runner::{ScenarioReport, ScenarioRunner, ScenarioRunnerConfig};
