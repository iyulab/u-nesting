//! Solve result representation.

use crate::geometry::GeometryId;
use crate::placement::{Placement, PlacementStats};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Statistics for a single strip/boundary in multi-strip packing.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StripStats {
    /// Index of the strip (0-based).
    pub strip_index: usize,
    /// Used length of the strip (max X extent of pieces).
    pub used_length: f64,
    /// Total area of pieces placed on this strip.
    pub piece_area: f64,
    /// Number of pieces placed on this strip.
    pub piece_count: usize,
    /// Strip width (height dimension for horizontal strips).
    pub strip_width: f64,
    /// Strip height (or max possible length).
    pub strip_height: f64,
}

/// Result of a nesting or packing solve operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveResult<S> {
    /// List of placements for all successfully placed geometry instances.
    pub placements: Vec<Placement<S>>,

    /// Number of boundaries (bins) used.
    pub boundaries_used: usize,

    /// Utilization ratio (0.0 - 1.0).
    /// Calculated as: total_geometry_measure / total_boundary_measure
    pub utilization: f64,

    /// IDs of geometries that could not be placed.
    pub unplaced: Vec<GeometryId>,

    /// Computation time in milliseconds.
    pub computation_time_ms: u64,

    /// Number of generations (for GA-based solvers).
    pub generations: Option<u32>,

    /// Number of iterations (for SA-based solvers).
    pub iterations: Option<u64>,

    /// Best fitness value achieved (for GA/SA-based solvers).
    pub best_fitness: Option<f64>,

    /// Fitness history over generations (for analysis).
    pub fitness_history: Option<Vec<f64>>,

    /// Strategy used for solving.
    pub strategy: Option<String>,

    /// Whether the solve was cancelled early.
    pub cancelled: bool,

    /// Whether the target utilization was reached.
    pub target_reached: bool,

    /// Per-strip statistics for multi-strip packing.
    /// Contains used_length, piece_area, piece_count for each strip.
    pub strip_stats: Vec<StripStats>,

    /// Total area of all placed pieces.
    pub total_piece_area: f64,

    /// Total material area consumed (sum of strip_width × used_length for each strip).
    pub total_material_used: f64,

    /// Total number of geometry **instances** requested (Σ of every geometry's
    /// quantity). This is the instance-level denominator: the count of unplaced
    /// instances is `total_requested - placements.len()`, since `placements` is
    /// instance-level while `unplaced` is deduplicated to unique geometry IDs.
    /// Set once at the top-level `solve`/`solve_with_progress` entry point.
    pub total_requested: usize,

    /// Axis-aligned bounding box `[width, height]` of the placed pieces' actual
    /// footprint (2D). Unlike `utilization` (piece area over the *full* boundary,
    /// which shrinks arbitrarily as boundary height grows), the used bounding box
    /// is boundary-padding independent — for an open-ended roll the larger axis is
    /// the material length actually consumed. `[0.0, 0.0]` when nothing is placed.
    /// Populated at the 2D validation/filter step; unused by 3D packing.
    pub used_bounding_box: [f64; 2],
}

impl<S> SolveResult<S> {
    /// Creates a new empty result.
    pub fn new() -> Self {
        Self {
            placements: Vec::new(),
            boundaries_used: 0,
            utilization: 0.0,
            unplaced: Vec::new(),
            computation_time_ms: 0,
            generations: None,
            iterations: None,
            best_fitness: None,
            fitness_history: None,
            strategy: None,
            cancelled: false,
            target_reached: false,
            strip_stats: Vec::new(),
            total_piece_area: 0.0,
            total_material_used: 0.0,
            total_requested: 0,
            used_bounding_box: [0.0, 0.0],
        }
    }

    /// Returns true if all geometries were placed.
    pub fn all_placed(&self) -> bool {
        self.unplaced.is_empty()
    }

    /// Returns the number of placed geometry instances.
    pub fn placed_count(&self) -> usize {
        self.placements.len()
    }

    /// Returns the number of unplaced geometry types.
    pub fn unplaced_count(&self) -> usize {
        self.unplaced.len()
    }

    /// Returns true if the solve was successful (at least one placement).
    pub fn is_successful(&self) -> bool {
        !self.placements.is_empty()
    }

    /// Returns true if the solve completed within the time limit.
    pub fn completed_normally(&self) -> bool {
        !self.cancelled
    }

    /// Sets the strategy name.
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = Some(strategy.into());
        self
    }

    /// Sets the generations count.
    pub fn with_generations(mut self, generations: u32) -> Self {
        self.generations = Some(generations);
        self
    }

    /// Sets the best fitness.
    pub fn with_best_fitness(mut self, fitness: f64) -> Self {
        self.best_fitness = Some(fitness);
        self
    }

    /// Sets the fitness history.
    pub fn with_fitness_history(mut self, history: Vec<f64>) -> Self {
        self.fitness_history = Some(history);
        self
    }

    /// Removes duplicate entries from the unplaced list.
    /// This is useful when multiple instances of the same geometry failed to place.
    pub fn deduplicate_unplaced(&mut self) {
        let mut seen = std::collections::HashSet::new();
        self.unplaced.retain(|id| seen.insert(id.clone()));
    }

    /// Computes placement statistics.
    pub fn placement_stats(&self) -> PlacementStats {
        PlacementStats::from_placements(&self.placements)
    }

    /// Returns utilization as a percentage string.
    pub fn utilization_percent(&self) -> String {
        format!("{:.1}%", self.utilization * 100.0)
    }

    /// Merges placements from another result (for multi-bin scenarios).
    pub fn merge(&mut self, other: SolveResult<S>, boundary_offset: usize) {
        // Offset boundary indices
        for mut placement in other.placements {
            placement.boundary_index += boundary_offset;
            self.placements.push(placement);
        }

        self.boundaries_used = self
            .boundaries_used
            .max(other.boundaries_used + boundary_offset);
        self.unplaced.extend(other.unplaced);
        self.computation_time_ms += other.computation_time_ms;

        // Merge strip stats with offset
        for mut strip_stat in other.strip_stats {
            strip_stat.strip_index += boundary_offset;
            self.strip_stats.push(strip_stat);
        }
        self.total_piece_area += other.total_piece_area;
        self.total_material_used += other.total_material_used;
        self.total_requested += other.total_requested;
        // Used footprints live in per-sheet local frames, so a union box is
        // ill-defined across sheets; report the element-wise max extent seen.
        self.used_bounding_box = [
            self.used_bounding_box[0].max(other.used_bounding_box[0]),
            self.used_bounding_box[1].max(other.used_bounding_box[1]),
        ];

        // Recalculate utilization
        if self.total_material_used > 0.0 {
            self.utilization = self.total_piece_area / self.total_material_used;
        }
    }

    /// Sets strip statistics.
    pub fn with_strip_stats(mut self, stats: Vec<StripStats>) -> Self {
        self.strip_stats = stats;
        self
    }

    /// Calculates and sets utilization from strip stats.
    /// This is the accurate utilization based on actual material consumed.
    pub fn calculate_utilization(&mut self) {
        if self.strip_stats.is_empty() {
            return;
        }

        self.total_piece_area = self.strip_stats.iter().map(|s| s.piece_area).sum();
        self.total_material_used = self
            .strip_stats
            .iter()
            .map(|s| s.strip_width * s.used_length)
            .sum();

        if self.total_material_used > 0.0 {
            self.utilization = self.total_piece_area / self.total_material_used;
        }
    }
}

impl<S> Default for SolveResult<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl SolveResult<f64> {
    /// Rewrites placement x-coordinates from the global multi-strip frame
    /// (sheets laid side-by-side: `x ≈ boundary_index * extent`) into the
    /// per-boundary **local** frame, so each placement's x is relative to the
    /// origin of its own sheet (`boundary_index` selects the sheet).
    ///
    /// `solve_multi_strip` emits global strip coordinates (the strip-packing
    /// representation its benchmark callers rely on). Binding/wire consumers that
    /// render per-sheet panels want sheet-local coordinates instead; this is the
    /// single shared transform they apply at the response boundary.
    ///
    /// `extent` is the boundary's x-axis extent (`b_max[0] - b_min[0]`), identical
    /// to the `strip_width` used during solving. After this call every placement's
    /// x lies in `[0, extent)` (modulo spacing/margin) within its sheet.
    pub fn to_boundary_local(&mut self, extent: f64) {
        for placement in &mut self.placements {
            if let Some(x) = placement.position.first_mut() {
                *x -= placement.boundary_index as f64 * extent;
            }
        }
    }
}

/// Summary statistics for a solve result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SolveSummary {
    /// Total geometries requested.
    pub total_requested: usize,
    /// Total geometries placed.
    pub total_placed: usize,
    /// Utilization percentage.
    pub utilization_percent: f64,
    /// Number of bins/boundaries used.
    pub bins_used: usize,
    /// Computation time in milliseconds.
    pub time_ms: u64,
    /// Strategy used.
    pub strategy: String,
}

impl<S> From<&SolveResult<S>> for SolveSummary {
    fn from(result: &SolveResult<S>) -> Self {
        Self {
            // `result.unplaced` is deduplicated to unique geometry IDs, so
            // `placements.len() + unplaced.len()` undercounts whenever a
            // multi-quantity geometry is partially/fully unplaced. Use the
            // authoritative instance-level total recorded at solve time.
            total_requested: result.total_requested,
            total_placed: result.placements.len(),
            utilization_percent: result.utilization * 100.0,
            bins_used: result.boundaries_used,
            time_ms: result.computation_time_ms,
            strategy: result
                .strategy
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_new() {
        let result: SolveResult<f64> = SolveResult::new();
        assert!(result.placements.is_empty());
        assert_eq!(result.utilization, 0.0);
        assert!(result.all_placed());
    }

    #[test]
    fn test_result_with_placements() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result
            .placements
            .push(Placement::new_2d("test".to_string(), 0, 0.0, 0.0, 0.0));
        result.utilization = 0.85;

        assert_eq!(result.placed_count(), 1);
        assert!(result.is_successful());
        assert_eq!(result.utilization_percent(), "85.0%");
    }

    #[test]
    fn test_result_with_unplaced() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());

        assert!(!result.all_placed());
        assert_eq!(result.unplaced_count(), 2);
    }

    #[test]
    fn test_solve_summary() {
        let mut result: SolveResult<f64> = SolveResult::new();
        result
            .placements
            .push(Placement::new_2d("test".to_string(), 0, 0.0, 0.0, 0.0));
        result.utilization = 0.75;
        result.boundaries_used = 1;
        result.computation_time_ms = 100;
        result.strategy = Some("GA".to_string());

        let summary = SolveSummary::from(&result);
        assert_eq!(summary.total_placed, 1);
        assert_eq!(summary.utilization_percent, 75.0);
        assert_eq!(summary.strategy, "GA");
    }

    #[test]
    fn test_solve_summary_total_requested_is_instance_level() {
        // Reproduces the reported defect: one geometry with quantity 5, of which
        // only 1 instance is placed. `unplaced` is deduplicated to the single
        // failing geometry ID. The old summary formula
        // (placements.len() + unplaced.len() = 1 + 1 = 2) undercounts the true
        // request total. With the authoritative `total_requested`, it is 5.
        let mut result: SolveResult<f64> = SolveResult::new();
        result
            .placements
            .push(Placement::new_2d("A".to_string(), 0, 0.0, 0.0, 0.0));
        result.unplaced.push("A".to_string()); // deduplicated single ID
        result.total_requested = 5;

        let summary = SolveSummary::from(&result);
        assert_eq!(summary.total_requested, 5);
        assert_eq!(summary.total_placed, 1);
        // Instance-level unplaced count, recovered by consumers:
        assert_eq!(summary.total_requested - summary.total_placed, 4);
    }

    #[test]
    fn test_merge_sums_total_requested() {
        let mut a: SolveResult<f64> = SolveResult::new();
        a.total_requested = 3;
        let mut b: SolveResult<f64> = SolveResult::new();
        b.total_requested = 4;
        a.merge(b, 0);
        assert_eq!(a.total_requested, 7);
    }

    #[test]
    fn test_deduplicate_unplaced() {
        let mut result: SolveResult<f64> = SolveResult::new();
        // Simulate multiple instances of same geometry failing to place
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());
        result.unplaced.push("G1".to_string());
        result.unplaced.push("G2".to_string());

        assert_eq!(result.unplaced.len(), 5);

        result.deduplicate_unplaced();

        assert_eq!(result.unplaced.len(), 2);
        assert!(result.unplaced.contains(&"G1".to_string()));
        assert!(result.unplaced.contains(&"G2".to_string()));
    }

    #[test]
    fn test_to_boundary_local_subtracts_per_sheet_offset() {
        // Three placements at the same sheet-local x=10, on sheets 0/1/2, in a global
        // strip frame (x = local + sheet * 100). Localizing must recover x=10 on each.
        let extent = 100.0;
        let mut result: SolveResult<f64> = SolveResult::new();
        for sheet in 0..3 {
            result.placements.push(
                Placement::new_2d(
                    "p".to_string(),
                    sheet,
                    10.0 + sheet as f64 * extent,
                    5.0,
                    0.0,
                )
                .with_boundary(sheet),
            );
        }

        result.to_boundary_local(extent);

        for p in &result.placements {
            assert_eq!(p.position[0], 10.0, "x must be sheet-local after transform");
            assert_eq!(p.position[1], 5.0, "y is untouched");
        }
    }
}
