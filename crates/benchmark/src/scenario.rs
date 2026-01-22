//! Test scenario definitions for systematic defect discovery.
//!
//! Scenarios define structured test cases with specific goals, datasets,
//! strategies, and success criteria for quality validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use u_nesting_core::Strategy;

/// A test scenario for benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Unique scenario ID (e.g., "2D-S01")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// What this scenario tests
    pub purpose: String,
    /// Category: "2d", "3d", or "common"
    pub category: ScenarioCategory,
    /// Datasets to use (can be ESICUP names or paths)
    pub datasets: Vec<DatasetRef>,
    /// Strategies to test
    pub strategies: Vec<StrategyRef>,
    /// Time limit per run in seconds
    pub time_limit_secs: u64,
    /// Number of runs per configuration
    pub runs: usize,
    /// Success criteria
    pub criteria: SuccessCriteria,
    /// Tags for filtering
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Category of scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScenarioCategory {
    /// 2D nesting scenarios
    #[serde(rename = "2d")]
    TwoD,
    /// 3D bin packing scenarios
    #[serde(rename = "3d")]
    ThreeD,
    /// Common/cross-cutting scenarios
    Common,
}

/// Reference to a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DatasetRef {
    /// ESICUP dataset by name and instance
    Esicup { dataset: String, instance: String },
    /// Local file path
    File { path: String },
    /// Synthetic dataset type
    Synthetic { synthetic: String },
    /// 3D MPV instance
    Mpv { mpv_class: String },
}

/// Reference to a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrategyRef {
    Blf,
    Nfp,
    Ga,
    Brkga,
    Sa,
    #[serde(rename = "extreme_point")]
    ExtremePoint,
    #[serde(rename = "layer")]
    Layer,
}

impl From<StrategyRef> for Strategy {
    fn from(r: StrategyRef) -> Self {
        match r {
            StrategyRef::Blf => Strategy::BottomLeftFill,
            StrategyRef::Nfp => Strategy::NfpGuided,
            StrategyRef::Ga => Strategy::GeneticAlgorithm,
            StrategyRef::Brkga => Strategy::Brkga,
            StrategyRef::Sa => Strategy::SimulatedAnnealing,
            StrategyRef::ExtremePoint => Strategy::ExtremePoint,
            StrategyRef::Layer => Strategy::BottomLeftFill, // Layer maps to BLF for 3D
        }
    }
}

/// Success criteria for a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Minimum placement ratio (0.0-1.0)
    #[serde(default)]
    pub min_placement_ratio: Option<f64>,
    /// Minimum utilization (0.0-1.0)
    #[serde(default)]
    pub min_utilization: Option<f64>,
    /// Maximum gap from best-known (0.0-1.0, e.g., 0.15 = 15%)
    #[serde(default)]
    pub max_gap_from_best: Option<f64>,
    /// Maximum time in milliseconds
    #[serde(default)]
    pub max_time_ms: Option<u64>,
    /// Must not crash
    #[serde(default = "default_true")]
    pub must_not_crash: bool,
    /// All items must be placed
    #[serde(default)]
    pub all_items_placed: bool,
}

fn default_true() -> bool {
    true
}

impl Default for SuccessCriteria {
    fn default() -> Self {
        Self {
            min_placement_ratio: None,
            min_utilization: None,
            max_gap_from_best: None,
            max_time_ms: None,
            must_not_crash: true,
            all_items_placed: false,
        }
    }
}

/// A collection of scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSet {
    /// Name of the scenario set
    pub name: String,
    /// Description
    pub description: String,
    /// Scenarios in this set
    pub scenarios: Vec<Scenario>,
}

impl ScenarioSet {
    /// Load a scenario set from a TOML file.
    pub fn from_toml_file(path: &Path) -> Result<Self, ScenarioError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ScenarioError::IoError(e.to_string()))?;
        toml::from_str(&content).map_err(|e| ScenarioError::ParseError(e.to_string()))
    }

    /// Load a scenario set from a JSON file.
    pub fn from_json_file(path: &Path) -> Result<Self, ScenarioError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ScenarioError::IoError(e.to_string()))?;
        serde_json::from_str(&content).map_err(|e| ScenarioError::ParseError(e.to_string()))
    }

    /// Filter scenarios by category.
    pub fn filter_by_category(&self, category: ScenarioCategory) -> Vec<&Scenario> {
        self.scenarios
            .iter()
            .filter(|s| s.category == category)
            .collect()
    }

    /// Filter scenarios by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&Scenario> {
        self.scenarios
            .iter()
            .filter(|s| s.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Get a scenario by ID.
    pub fn get_by_id(&self, id: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.id == id)
    }
}

/// Error type for scenario operations.
#[derive(Debug)]
pub enum ScenarioError {
    IoError(String),
    ParseError(String),
}

impl std::fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScenarioError::IoError(msg) => write!(f, "IO error: {}", msg),
            ScenarioError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for ScenarioError {}

/// Result of running a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario ID
    pub scenario_id: String,
    /// Whether all criteria were met
    pub passed: bool,
    /// Individual run results
    pub runs: Vec<ScenarioRunResult>,
    /// Criteria evaluation
    pub criteria_results: HashMap<String, CriterionResult>,
    /// Defects discovered
    pub defects: Vec<Defect>,
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
}

/// Result of a single run within a scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioRunResult {
    pub dataset: String,
    pub strategy: String,
    pub run_index: usize,
    pub placement_ratio: f64,
    pub utilization: f64,
    pub strip_length: f64,
    pub time_ms: u64,
    pub placed_count: usize,
    pub total_count: usize,
    pub error: Option<String>,
    pub gap_from_best: Option<f64>,
}

/// Result of evaluating a single criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
    pub criterion: String,
    pub passed: bool,
    pub expected: String,
    pub actual: String,
}

/// A defect discovered during scenario execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defect {
    /// Unique defect ID
    pub id: String,
    /// Defect category
    pub category: DefectCategory,
    /// Severity level
    pub severity: DefectSeverity,
    /// Short description
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Which scenario discovered this
    pub scenario_id: String,
    /// Which dataset triggered this
    pub dataset: String,
    /// Which strategy triggered this
    pub strategy: String,
    /// Evidence/reproduction steps
    pub evidence: String,
}

/// Category of defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DefectCategory {
    /// Crashes, panics, infinite loops
    Bug,
    /// NFP errors, collision detection failures
    Accuracy,
    /// Slow performance, high memory
    Performance,
    /// Low utilization, suboptimal results
    Quality,
    /// Inconvenient interface, missing features
    Api,
    /// Missing or unclear documentation
    Docs,
}

/// Severity of defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DefectSeverity {
    /// Critical: crashes, data corruption, security
    P0,
    /// High: wrong results, severe performance
    P1,
    /// Medium: quality degradation, usability
    P2,
    /// Low: minor improvements, code cleanup
    P3,
}

/// Create the standard Phase 0 scenario set.
pub fn create_phase0_scenarios() -> ScenarioSet {
    ScenarioSet {
        name: "Phase 0: Quality Validation".to_string(),
        description: "Comprehensive test scenarios for defect discovery before v0.1.0 release"
            .to_string(),
        scenarios: vec![
            // 2D Scenarios
            create_2d_s01(),
            create_2d_s02(),
            create_2d_s03(),
            create_2d_s04(),
            create_2d_s05(),
            create_2d_s06(),
            create_2d_s07(),
            create_2d_s08(),
            create_2d_s09(),
            create_2d_s10(),
            // 3D Scenarios
            create_3d_s01(),
            create_3d_s02(),
            create_3d_s03(),
            create_3d_s04(),
            create_3d_s05(),
            create_3d_s06(),
            create_3d_s07(),
            // Common Scenarios
            create_c_s01(),
            create_c_s02(),
            create_c_s03(),
            create_c_s04(),
            create_c_s05(),
        ],
    }
}

// 2D Scenario factories

fn create_2d_s01() -> Scenario {
    Scenario {
        id: "2D-S01".to_string(),
        name: "Basic Functionality".to_string(),
        purpose: "Verify all strategies produce valid results".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "SHAPES".to_string(),
                instance: "shapes0".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "BLAZ".to_string(),
                instance: "blaz1".to_string(),
            },
        ],
        strategies: vec![
            StrategyRef::Blf,
            StrategyRef::Nfp,
            StrategyRef::Ga,
            StrategyRef::Brkga,
            StrategyRef::Sa,
        ],
        time_limit_secs: 30,
        runs: 1,
        criteria: SuccessCriteria {
            min_placement_ratio: Some(0.5),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["basic".to_string(), "smoke".to_string()],
    }
}

fn create_2d_s02() -> Scenario {
    Scenario {
        id: "2D-S02".to_string(),
        name: "Convex Polygon Optimization".to_string(),
        purpose: "Test NFP accuracy and placement quality for convex shapes".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![DatasetRef::Synthetic {
            synthetic: "synthetic_convex".to_string(),
        }],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp, StrategyRef::Ga],
        time_limit_secs: 60,
        runs: 3,
        criteria: SuccessCriteria {
            min_utilization: Some(0.70),
            all_items_placed: true,
            ..Default::default()
        },
        tags: vec!["convex".to_string(), "nfp".to_string()],
    }
}

fn create_2d_s03() -> Scenario {
    Scenario {
        id: "2D-S03".to_string(),
        name: "Non-Convex Polygon Processing".to_string(),
        purpose: "Test triangulation + NFP union accuracy for concave shapes".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "FU".to_string(),
                instance: "fu".to_string(),
            },
            DatasetRef::Synthetic {
                synthetic: "synthetic_concave".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp],
        time_limit_secs: 120,
        runs: 2,
        criteria: SuccessCriteria {
            min_placement_ratio: Some(0.8),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec![
            "concave".to_string(),
            "nfp".to_string(),
            "triangulation".to_string(),
        ],
    }
}

fn create_2d_s04() -> Scenario {
    Scenario {
        id: "2D-S04".to_string(),
        name: "Polygons with Holes".to_string(),
        purpose: "Test hole handling in geometry and NFP computation".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "MARQUES".to_string(),
                instance: "marques".to_string(),
            },
            DatasetRef::Synthetic {
                synthetic: "synthetic_with_holes".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp],
        time_limit_secs: 120,
        runs: 2,
        criteria: SuccessCriteria {
            min_placement_ratio: Some(0.6),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["holes".to_string(), "complex".to_string()],
    }
}

fn create_2d_s05() -> Scenario {
    Scenario {
        id: "2D-S05".to_string(),
        name: "Large Instance Scale".to_string(),
        purpose: "Test performance and memory with large item counts".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "SHIRTS".to_string(),
                instance: "shirts".to_string(),
            },
            DatasetRef::Synthetic {
                synthetic: "synthetic_large".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp],
        time_limit_secs: 300,
        runs: 1,
        criteria: SuccessCriteria {
            max_time_ms: Some(300_000),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["scale".to_string(), "performance".to_string()],
    }
}

fn create_2d_s06() -> Scenario {
    Scenario {
        id: "2D-S06".to_string(),
        name: "Rotation Optimization".to_string(),
        purpose: "Test multi-angle rotation effects on placement quality".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "JAKOBS".to_string(),
                instance: "jakobs1".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "TROUSERS".to_string(),
                instance: "trousers".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Nfp, StrategyRef::Ga, StrategyRef::Brkga],
        time_limit_secs: 120,
        runs: 2,
        criteria: SuccessCriteria {
            min_utilization: Some(0.60),
            ..Default::default()
        },
        tags: vec!["rotation".to_string(), "optimization".to_string()],
    }
}

fn create_2d_s07() -> Scenario {
    Scenario {
        id: "2D-S07".to_string(),
        name: "Numerical Stability".to_string(),
        purpose: "Test edge cases: near-collinear edges, tiny items, extreme aspect ratios"
            .to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Synthetic {
                synthetic: "synthetic_near_collinear".to_string(),
            },
            DatasetRef::Synthetic {
                synthetic: "synthetic_tiny".to_string(),
            },
            DatasetRef::Synthetic {
                synthetic: "synthetic_extreme_aspect".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp],
        time_limit_secs: 60,
        runs: 1,
        criteria: SuccessCriteria {
            must_not_crash: true,
            min_placement_ratio: Some(0.5),
            ..Default::default()
        },
        tags: vec!["stability".to_string(), "edge-case".to_string()],
    }
}

fn create_2d_s08() -> Scenario {
    Scenario {
        id: "2D-S08".to_string(),
        name: "Strategy Comparison".to_string(),
        purpose: "Compare all strategies across ESICUP datasets".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "SHAPES".to_string(),
                instance: "shapes0".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "SHAPES".to_string(),
                instance: "shapes1".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "ALBANO".to_string(),
                instance: "albano".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "JAKOBS".to_string(),
                instance: "jakobs1".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "JAKOBS".to_string(),
                instance: "jakobs2".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "FU".to_string(),
                instance: "fu".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "SWIM".to_string(),
                instance: "swim".to_string(),
            },
        ],
        strategies: vec![
            StrategyRef::Blf,
            StrategyRef::Nfp,
            StrategyRef::Ga,
            StrategyRef::Brkga,
            StrategyRef::Sa,
        ],
        time_limit_secs: 60,
        runs: 3,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["comparison".to_string(), "comprehensive".to_string()],
    }
}

fn create_2d_s09() -> Scenario {
    Scenario {
        id: "2D-S09".to_string(),
        name: "Known Optimal Verification".to_string(),
        purpose: "Test if 100% optimal solutions can be achieved for jigsaw instances".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![DatasetRef::Synthetic {
            synthetic: "synthetic_jigsaw_4x4".to_string(),
        }],
        strategies: vec![
            StrategyRef::Nfp,
            StrategyRef::Ga,
            StrategyRef::Brkga,
            StrategyRef::Sa,
        ],
        time_limit_secs: 120,
        runs: 3,
        criteria: SuccessCriteria {
            all_items_placed: true,
            min_utilization: Some(0.95),
            ..Default::default()
        },
        tags: vec!["optimal".to_string(), "jigsaw".to_string()],
    }
}

fn create_2d_s10() -> Scenario {
    Scenario {
        id: "2D-S10".to_string(),
        name: "Time-Constrained Quality".to_string(),
        purpose: "Measure maximum quality within strict time limits".to_string(),
        category: ScenarioCategory::TwoD,
        datasets: vec![
            DatasetRef::Esicup {
                dataset: "ALBANO".to_string(),
                instance: "albano".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "DAGLI".to_string(),
                instance: "dagli".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp, StrategyRef::Ga],
        time_limit_secs: 10,
        runs: 5,
        criteria: SuccessCriteria {
            max_time_ms: Some(10_000),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["time-constrained".to_string(), "quick".to_string()],
    }
}

// 3D Scenario factories

fn create_3d_s01() -> Scenario {
    Scenario {
        id: "3D-S01".to_string(),
        name: "Basic 3D Functionality".to_string(),
        purpose: "Verify Layer and EP strategies produce valid results".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![
            DatasetRef::Mpv {
                mpv_class: "MPV1".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV2".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV3".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Layer, StrategyRef::ExtremePoint],
        time_limit_secs: 30,
        runs: 1,
        criteria: SuccessCriteria {
            min_placement_ratio: Some(0.5),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["basic".to_string(), "3d".to_string()],
    }
}

fn create_3d_s02() -> Scenario {
    Scenario {
        id: "3D-S02".to_string(),
        name: "Varied Box Sizes".to_string(),
        purpose: "Test handling of items with diverse dimensions".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![
            DatasetRef::Mpv {
                mpv_class: "MPV4".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV5".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Layer, StrategyRef::ExtremePoint],
        time_limit_secs: 60,
        runs: 2,
        criteria: SuccessCriteria {
            min_placement_ratio: Some(0.7),
            ..Default::default()
        },
        tags: vec!["varied".to_string(), "3d".to_string()],
    }
}

fn create_3d_s03() -> Scenario {
    Scenario {
        id: "3D-S03".to_string(),
        name: "Rotation Optimization 3D".to_string(),
        purpose: "Test 6-direction rotation effects".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![
            DatasetRef::Mpv {
                mpv_class: "BW6".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "BW7".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "BW8".to_string(),
            },
        ],
        strategies: vec![StrategyRef::ExtremePoint, StrategyRef::Ga],
        time_limit_secs: 60,
        runs: 2,
        criteria: SuccessCriteria {
            min_utilization: Some(0.50),
            ..Default::default()
        },
        tags: vec!["rotation".to_string(), "3d".to_string()],
    }
}

fn create_3d_s04() -> Scenario {
    Scenario {
        id: "3D-S04".to_string(),
        name: "Mass Constraints".to_string(),
        purpose: "Test weight limit handling".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![DatasetRef::Mpv {
            mpv_class: "MPV1".to_string(),
        }],
        strategies: vec![StrategyRef::Layer],
        time_limit_secs: 30,
        runs: 1,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec![
            "mass".to_string(),
            "constraint".to_string(),
            "3d".to_string(),
        ],
    }
}

fn create_3d_s05() -> Scenario {
    Scenario {
        id: "3D-S05".to_string(),
        name: "Extreme Point Quality".to_string(),
        purpose: "Evaluate EP heuristic quality across all MPV instances".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![
            DatasetRef::Mpv {
                mpv_class: "MPV1".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV2".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV3".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV4".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV5".to_string(),
            },
        ],
        strategies: vec![StrategyRef::ExtremePoint],
        time_limit_secs: 60,
        runs: 3,
        criteria: SuccessCriteria {
            min_utilization: Some(0.60),
            ..Default::default()
        },
        tags: vec![
            "extreme-point".to_string(),
            "quality".to_string(),
            "3d".to_string(),
        ],
    }
}

fn create_3d_s06() -> Scenario {
    Scenario {
        id: "3D-S06".to_string(),
        name: "3D Strategy Comparison".to_string(),
        purpose: "Compare GA, BRKGA, SA for 3D packing".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![
            DatasetRef::Mpv {
                mpv_class: "MPV1".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV2".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV3".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV4".to_string(),
            },
            DatasetRef::Mpv {
                mpv_class: "MPV5".to_string(),
            },
        ],
        strategies: vec![
            StrategyRef::Layer,
            StrategyRef::ExtremePoint,
            StrategyRef::Ga,
            StrategyRef::Brkga,
            StrategyRef::Sa,
        ],
        time_limit_secs: 60,
        runs: 3,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["comparison".to_string(), "3d".to_string()],
    }
}

fn create_3d_s07() -> Scenario {
    Scenario {
        id: "3D-S07".to_string(),
        name: "Large 3D Instance".to_string(),
        purpose: "Test 100+ item 3D packing performance".to_string(),
        category: ScenarioCategory::ThreeD,
        datasets: vec![DatasetRef::Mpv {
            mpv_class: "Custom".to_string(),
        }],
        strategies: vec![StrategyRef::Layer, StrategyRef::ExtremePoint],
        time_limit_secs: 300,
        runs: 1,
        criteria: SuccessCriteria {
            max_time_ms: Some(300_000),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec![
            "scale".to_string(),
            "performance".to_string(),
            "3d".to_string(),
        ],
    }
}

// Common Scenario factories

fn create_c_s01() -> Scenario {
    Scenario {
        id: "C-S01".to_string(),
        name: "FFI Integration".to_string(),
        purpose: "Test JSON API boundary conditions".to_string(),
        category: ScenarioCategory::Common,
        datasets: vec![DatasetRef::Synthetic {
            synthetic: "synthetic_convex".to_string(),
        }],
        strategies: vec![StrategyRef::Blf],
        time_limit_secs: 10,
        runs: 1,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["ffi".to_string(), "integration".to_string()],
    }
}

fn create_c_s02() -> Scenario {
    Scenario {
        id: "C-S02".to_string(),
        name: "Cancellation".to_string(),
        purpose: "Test cancellation token functionality".to_string(),
        category: ScenarioCategory::Common,
        datasets: vec![DatasetRef::Synthetic {
            synthetic: "synthetic_large".to_string(),
        }],
        strategies: vec![StrategyRef::Ga],
        time_limit_secs: 5,
        runs: 1,
        criteria: SuccessCriteria {
            max_time_ms: Some(6_000),
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["cancellation".to_string(), "timeout".to_string()],
    }
}

fn create_c_s03() -> Scenario {
    Scenario {
        id: "C-S03".to_string(),
        name: "Progress Callback".to_string(),
        purpose: "Test progress reporting accuracy".to_string(),
        category: ScenarioCategory::Common,
        datasets: vec![DatasetRef::Synthetic {
            synthetic: "synthetic_convex".to_string(),
        }],
        strategies: vec![StrategyRef::Ga],
        time_limit_secs: 30,
        runs: 1,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["progress".to_string(), "callback".to_string()],
    }
}

fn create_c_s04() -> Scenario {
    Scenario {
        id: "C-S04".to_string(),
        name: "Memory Usage".to_string(),
        purpose: "Profile memory consumption on large instances".to_string(),
        category: ScenarioCategory::Common,
        datasets: vec![
            DatasetRef::Synthetic {
                synthetic: "synthetic_large".to_string(),
            },
            DatasetRef::Esicup {
                dataset: "SHIRTS".to_string(),
                instance: "shirts".to_string(),
            },
        ],
        strategies: vec![StrategyRef::Blf, StrategyRef::Nfp],
        time_limit_secs: 120,
        runs: 1,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["memory".to_string(), "profiling".to_string()],
    }
}

fn create_c_s05() -> Scenario {
    Scenario {
        id: "C-S05".to_string(),
        name: "Parallel Scaling".to_string(),
        purpose: "Test multi-thread performance scaling".to_string(),
        category: ScenarioCategory::Common,
        datasets: vec![DatasetRef::Esicup {
            dataset: "SHAPES".to_string(),
            instance: "shapes0".to_string(),
        }],
        strategies: vec![StrategyRef::Ga, StrategyRef::Brkga],
        time_limit_secs: 60,
        runs: 3,
        criteria: SuccessCriteria {
            must_not_crash: true,
            ..Default::default()
        },
        tags: vec!["parallel".to_string(), "scaling".to_string()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_phase0_scenarios() {
        let set = create_phase0_scenarios();
        assert_eq!(set.scenarios.len(), 22); // 10 + 7 + 5

        // Check categories
        let two_d = set.filter_by_category(ScenarioCategory::TwoD);
        let three_d = set.filter_by_category(ScenarioCategory::ThreeD);
        let common = set.filter_by_category(ScenarioCategory::Common);

        assert_eq!(two_d.len(), 10);
        assert_eq!(three_d.len(), 7);
        assert_eq!(common.len(), 5);
    }

    #[test]
    fn test_scenario_serialization() {
        let scenario = create_2d_s01();
        let json = serde_json::to_string_pretty(&scenario).unwrap();
        let parsed: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, scenario.id);
    }

    #[test]
    fn test_scenario_set_toml() {
        let set = create_phase0_scenarios();
        let toml = toml::to_string_pretty(&set).unwrap();
        assert!(toml.contains("2D-S01"));
    }
}
