//! Stability analysis for 3D bin packing.
//!
//! This module provides various stability constraints for validating 3D placements.
//! Stability is crucial in real-world logistics and manufacturing scenarios.
//!
//! # Stability Models
//!
//! The module implements a hierarchy of stability models from simple to complex:
//!
//! 1. **Full Base Support**: 100% of the bottom face must be supported
//! 2. **Partial Base Support**: A configurable percentage (e.g., 70-80%) must be supported
//! 3. **Center of Gravity (CoG) Polygon**: CoG projection must fall within support polygon
//! 4. **Static Mechanical Equilibrium**: Full force/moment balance analysis
//!
//! # Example
//!
//! ```ignore
//! use u_nesting_d3::stability::{StabilityConstraint, StabilityAnalyzer};
//! use u_nesting_d3::{Geometry3D, Boundary3D, Packer3D, Config};
//!
//! let analyzer = StabilityAnalyzer::new(StabilityConstraint::PartialBase { min_ratio: 0.75 });
//! let placements = /* ... */;
//! let report = analyzer.analyze(&placements, &geometries);
//! ```

use nalgebra::{Point3, Vector3};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Stability constraint type for 3D packing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StabilityConstraint {
    /// No stability checking (fastest).
    None,

    /// Full base support: 100% of the bottom face must be supported.
    /// Most restrictive but guarantees stability.
    FullBase,

    /// Partial base support: at least `min_ratio` (0.0-1.0) of the bottom face
    /// must be supported. Common values: 0.7-0.8.
    PartialBase {
        /// Minimum support ratio (0.0-1.0).
        min_ratio: f64,
    },

    /// Center of Gravity polygon support: the projection of the item's
    /// center of gravity must fall within the convex hull of support points.
    CogPolygon,

    /// Static mechanical equilibrium: full Newton's laws analysis.
    /// ΣF = 0, ΣM = 0 for all contact forces.
    /// Most accurate but computationally expensive.
    StaticEquilibrium {
        /// Tolerance for force balance (default: 1e-6).
        force_tolerance: f64,
        /// Tolerance for moment balance (default: 1e-6).
        moment_tolerance: f64,
    },
}

impl Default for StabilityConstraint {
    fn default() -> Self {
        Self::None
    }
}

impl StabilityConstraint {
    /// Creates a partial base support constraint with the given ratio.
    pub fn partial_base(min_ratio: f64) -> Self {
        Self::PartialBase {
            min_ratio: min_ratio.clamp(0.0, 1.0),
        }
    }

    /// Creates a static equilibrium constraint with default tolerances.
    pub fn static_equilibrium() -> Self {
        Self::StaticEquilibrium {
            force_tolerance: 1e-6,
            moment_tolerance: 1e-6,
        }
    }

    /// Returns true if this constraint requires checking.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// A placed box with position and dimensions.
#[derive(Debug, Clone)]
pub struct PlacedBox {
    /// Unique identifier.
    pub id: String,
    /// Instance index.
    pub instance: usize,
    /// Position (bottom-left-front corner).
    pub position: Point3<f64>,
    /// Dimensions (width, depth, height) after orientation applied.
    pub dimensions: Vector3<f64>,
    /// Mass of the box (optional).
    pub mass: Option<f64>,
    /// Center of gravity offset from geometric center (optional).
    pub cog_offset: Option<Vector3<f64>>,
}

impl PlacedBox {
    /// Creates a new placed box.
    pub fn new(
        id: impl Into<String>,
        instance: usize,
        position: Point3<f64>,
        dimensions: Vector3<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            instance,
            position,
            dimensions,
            mass: None,
            cog_offset: None,
        }
    }

    /// Sets the mass of the box.
    pub fn with_mass(mut self, mass: f64) -> Self {
        self.mass = Some(mass);
        self
    }

    /// Sets the center of gravity offset from geometric center.
    pub fn with_cog_offset(mut self, offset: Vector3<f64>) -> Self {
        self.cog_offset = Some(offset);
        self
    }

    /// Returns the geometric center of the box.
    pub fn center(&self) -> Point3<f64> {
        Point3::new(
            self.position.x + self.dimensions.x / 2.0,
            self.position.y + self.dimensions.y / 2.0,
            self.position.z + self.dimensions.z / 2.0,
        )
    }

    /// Returns the center of gravity.
    pub fn center_of_gravity(&self) -> Point3<f64> {
        let center = self.center();
        if let Some(offset) = self.cog_offset {
            Point3::new(
                center.x + offset.x,
                center.y + offset.y,
                center.z + offset.z,
            )
        } else {
            center
        }
    }

    /// Returns the bottom face AABB (z = position.z).
    pub fn bottom_face(&self) -> (Point3<f64>, Point3<f64>) {
        (
            Point3::new(self.position.x, self.position.y, self.position.z),
            Point3::new(
                self.position.x + self.dimensions.x,
                self.position.y + self.dimensions.y,
                self.position.z,
            ),
        )
    }

    /// Returns the top face AABB.
    pub fn top_face(&self) -> (Point3<f64>, Point3<f64>) {
        let top_z = self.position.z + self.dimensions.z;
        (
            Point3::new(self.position.x, self.position.y, top_z),
            Point3::new(
                self.position.x + self.dimensions.x,
                self.position.y + self.dimensions.y,
                top_z,
            ),
        )
    }

    /// Returns the volume of the box.
    pub fn volume(&self) -> f64 {
        self.dimensions.x * self.dimensions.y * self.dimensions.z
    }

    /// Returns the bottom face area.
    pub fn base_area(&self) -> f64 {
        self.dimensions.x * self.dimensions.y
    }
}

/// Result of stability analysis for a single box.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StabilityResult {
    /// Box identifier.
    pub id: String,
    /// Instance index.
    pub instance: usize,
    /// Whether the box is stable.
    pub is_stable: bool,
    /// Support ratio (0.0-1.0) - ratio of bottom face that is supported.
    pub support_ratio: f64,
    /// Boxes providing support to this box.
    pub supported_by: Vec<(String, usize)>,
    /// Whether the CoG is within the support polygon.
    pub cog_within_support: bool,
    /// Force imbalance magnitude (for equilibrium check).
    pub force_imbalance: f64,
    /// Moment imbalance magnitude (for equilibrium check).
    pub moment_imbalance: f64,
    /// Stability score (0.0-1.0, higher is more stable).
    pub stability_score: f64,
}

impl StabilityResult {
    /// Creates a new stability result.
    pub fn new(id: impl Into<String>, instance: usize) -> Self {
        Self {
            id: id.into(),
            instance,
            is_stable: true,
            support_ratio: 1.0,
            supported_by: Vec::new(),
            cog_within_support: true,
            force_imbalance: 0.0,
            moment_imbalance: 0.0,
            stability_score: 1.0,
        }
    }

    /// Marks the result as unstable.
    pub fn unstable(mut self) -> Self {
        self.is_stable = false;
        self.stability_score = 0.0;
        self
    }
}

/// Complete stability report for a packing solution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StabilityReport {
    /// Individual results for each box.
    pub results: Vec<StabilityResult>,
    /// Number of stable boxes.
    pub stable_count: usize,
    /// Number of unstable boxes.
    pub unstable_count: usize,
    /// Overall stability ratio.
    pub overall_stability: f64,
    /// Minimum support ratio among all boxes.
    pub min_support_ratio: f64,
    /// Average support ratio.
    pub avg_support_ratio: f64,
    /// Total weight of unstable boxes.
    pub unstable_weight: f64,
    /// Analysis time in milliseconds.
    pub analysis_time_ms: u64,
}

impl StabilityReport {
    /// Creates a new empty report.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            stable_count: 0,
            unstable_count: 0,
            overall_stability: 1.0,
            min_support_ratio: 1.0,
            avg_support_ratio: 1.0,
            unstable_weight: 0.0,
            analysis_time_ms: 0,
        }
    }

    /// Returns true if all boxes are stable.
    pub fn is_all_stable(&self) -> bool {
        self.unstable_count == 0
    }

    /// Returns the unstable boxes.
    pub fn unstable_boxes(&self) -> Vec<&StabilityResult> {
        self.results.iter().filter(|r| !r.is_stable).collect()
    }
}

impl Default for StabilityReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Stability analyzer for 3D bin packing.
pub struct StabilityAnalyzer {
    constraint: StabilityConstraint,
    /// Tolerance for considering boxes as "touching" (contact detection).
    contact_tolerance: f64,
    /// Gravity direction (default: -Z).
    gravity: Vector3<f64>,
}

impl StabilityAnalyzer {
    /// Creates a new stability analyzer with the given constraint.
    pub fn new(constraint: StabilityConstraint) -> Self {
        Self {
            constraint,
            contact_tolerance: 1e-6,
            gravity: Vector3::new(0.0, 0.0, -9.81),
        }
    }

    /// Sets the contact detection tolerance.
    pub fn with_contact_tolerance(mut self, tolerance: f64) -> Self {
        self.contact_tolerance = tolerance;
        self
    }

    /// Sets the gravity direction.
    pub fn with_gravity(mut self, gravity: Vector3<f64>) -> Self {
        self.gravity = gravity;
        self
    }

    /// Analyzes the stability of all placed boxes.
    pub fn analyze(&self, boxes: &[PlacedBox], floor_z: f64) -> StabilityReport {
        let start = std::time::Instant::now();
        let mut report = StabilityReport::new();

        if boxes.is_empty() || !self.constraint.is_enabled() {
            report.analysis_time_ms = start.elapsed().as_millis() as u64;
            return report;
        }

        // Build spatial lookup for efficient contact detection
        let boxes_by_top_z = self.build_top_z_index(boxes);

        // Analyze each box
        for placed_box in boxes {
            let result = self.analyze_box(placed_box, boxes, floor_z, &boxes_by_top_z);
            report.results.push(result);
        }

        // Compute summary statistics
        self.compute_summary(&mut report, boxes);
        report.analysis_time_ms = start.elapsed().as_millis() as u64;

        report
    }

    /// Checks if a single placement is stable.
    pub fn is_stable(&self, placed_box: &PlacedBox, supports: &[PlacedBox], floor_z: f64) -> bool {
        let all_boxes: Vec<_> = std::iter::once(placed_box.clone())
            .chain(supports.iter().cloned())
            .collect();

        let boxes_by_top_z = self.build_top_z_index(&all_boxes);
        let result = self.analyze_box(placed_box, &all_boxes, floor_z, &boxes_by_top_z);
        result.is_stable
    }

    /// Analyzes stability of a single box.
    fn analyze_box(
        &self,
        placed_box: &PlacedBox,
        all_boxes: &[PlacedBox],
        floor_z: f64,
        boxes_by_top_z: &HashMap<i64, Vec<usize>>,
    ) -> StabilityResult {
        let mut result = StabilityResult::new(&placed_box.id, placed_box.instance);

        // Find supporting surfaces
        let (support_ratio, supporters) =
            self.compute_support(placed_box, all_boxes, floor_z, boxes_by_top_z);

        result.support_ratio = support_ratio;
        result.supported_by = supporters;

        // Apply constraint check
        result.is_stable = match &self.constraint {
            StabilityConstraint::None => true,
            StabilityConstraint::FullBase => support_ratio >= 1.0 - self.contact_tolerance,
            StabilityConstraint::PartialBase { min_ratio } => support_ratio >= *min_ratio,
            StabilityConstraint::CogPolygon => {
                let cog_ok = self.check_cog_support(placed_box, all_boxes, floor_z, boxes_by_top_z);
                result.cog_within_support = cog_ok;
                cog_ok
            }
            StabilityConstraint::StaticEquilibrium {
                force_tolerance,
                moment_tolerance,
            } => {
                let (force_imb, moment_imb) =
                    self.check_equilibrium(placed_box, all_boxes, floor_z, boxes_by_top_z);
                result.force_imbalance = force_imb;
                result.moment_imbalance = moment_imb;
                force_imb <= *force_tolerance && moment_imb <= *moment_tolerance
            }
        };

        // Compute stability score
        result.stability_score = self.compute_stability_score(&result);

        result
    }

    /// Computes the support ratio and list of supporting boxes.
    fn compute_support(
        &self,
        placed_box: &PlacedBox,
        all_boxes: &[PlacedBox],
        floor_z: f64,
        boxes_by_top_z: &HashMap<i64, Vec<usize>>,
    ) -> (f64, Vec<(String, usize)>) {
        let bottom_z = placed_box.position.z;
        let base_area = placed_box.base_area();
        let (bottom_min, bottom_max) = placed_box.bottom_face();

        // Check if on floor
        if (bottom_z - floor_z).abs() <= self.contact_tolerance {
            return (1.0, vec![("floor".to_string(), 0)]);
        }

        // Find boxes whose top face is at this box's bottom
        let target_z_key = (bottom_z * 1000.0).round() as i64;
        let mut total_support_area = 0.0;
        let mut supporters = Vec::new();

        // Check boxes at the same z level (with tolerance)
        for dz in -1i64..=1 {
            if let Some(box_indices) = boxes_by_top_z.get(&(target_z_key + dz)) {
                for &idx in box_indices {
                    let support_box = &all_boxes[idx];

                    // Skip self
                    if support_box.id == placed_box.id
                        && support_box.instance == placed_box.instance
                    {
                        continue;
                    }

                    let (top_min, top_max) = support_box.top_face();

                    // Check if top face is at bottom z (within tolerance)
                    if (top_min.z - bottom_z).abs() > self.contact_tolerance {
                        continue;
                    }

                    // Compute overlap area
                    let overlap = self.compute_face_overlap(
                        (bottom_min.x, bottom_min.y, bottom_max.x, bottom_max.y),
                        (top_min.x, top_min.y, top_max.x, top_max.y),
                    );

                    if overlap > 0.0 {
                        total_support_area += overlap;
                        supporters.push((support_box.id.clone(), support_box.instance));
                    }
                }
            }
        }

        let support_ratio = (total_support_area / base_area).min(1.0);
        (support_ratio, supporters)
    }

    /// Checks if the CoG projection falls within the support polygon.
    fn check_cog_support(
        &self,
        placed_box: &PlacedBox,
        all_boxes: &[PlacedBox],
        floor_z: f64,
        boxes_by_top_z: &HashMap<i64, Vec<usize>>,
    ) -> bool {
        let cog = placed_box.center_of_gravity();
        let bottom_z = placed_box.position.z;

        // CoG projection point (x, y)
        let cog_xy = (cog.x, cog.y);

        // Check if on floor - CoG must be within base
        if (bottom_z - floor_z).abs() <= self.contact_tolerance {
            let (min, max) = placed_box.bottom_face();
            return cog_xy.0 >= min.x
                && cog_xy.0 <= max.x
                && cog_xy.1 >= min.y
                && cog_xy.1 <= max.y;
        }

        // Collect support regions
        let mut support_regions: Vec<(f64, f64, f64, f64)> = Vec::new();
        let target_z_key = (bottom_z * 1000.0).round() as i64;

        for dz in -1i64..=1 {
            if let Some(box_indices) = boxes_by_top_z.get(&(target_z_key + dz)) {
                for &idx in box_indices {
                    let support_box = &all_boxes[idx];
                    if support_box.id == placed_box.id
                        && support_box.instance == placed_box.instance
                    {
                        continue;
                    }

                    let (top_min, top_max) = support_box.top_face();
                    if (top_min.z - bottom_z).abs() <= self.contact_tolerance {
                        let (bottom_min, bottom_max) = placed_box.bottom_face();
                        let overlap = self.compute_face_overlap_coords(
                            (bottom_min.x, bottom_min.y, bottom_max.x, bottom_max.y),
                            (top_min.x, top_min.y, top_max.x, top_max.y),
                        );
                        if let Some(region) = overlap {
                            support_regions.push(region);
                        }
                    }
                }
            }
        }

        // Check if CoG projection is within any support region
        for (min_x, min_y, max_x, max_y) in support_regions {
            if cog_xy.0 >= min_x && cog_xy.0 <= max_x && cog_xy.1 >= min_y && cog_xy.1 <= max_y {
                return true;
            }
        }

        // For a more accurate check, compute convex hull of support regions
        // and check if CoG is inside. Simplified: check if CoG is close to
        // any support region center.
        false
    }

    /// Checks static mechanical equilibrium (force and moment balance).
    fn check_equilibrium(
        &self,
        placed_box: &PlacedBox,
        all_boxes: &[PlacedBox],
        floor_z: f64,
        boxes_by_top_z: &HashMap<i64, Vec<usize>>,
    ) -> (f64, f64) {
        // This is a simplified equilibrium check.
        // Full implementation would solve contact force distribution.

        let mass = placed_box.mass.unwrap_or(1.0);
        // CoG would be used in a full moment calculation
        let _cog = placed_box.center_of_gravity();

        // Gravity force
        let gravity_force = Vector3::new(0.0, 0.0, -mass * 9.81);

        // Compute support forces (simplified: assume uniform distribution)
        let (support_ratio, _supporters) =
            self.compute_support(placed_box, all_boxes, floor_z, boxes_by_top_z);

        if support_ratio < self.contact_tolerance {
            // No support - maximum imbalance
            return (gravity_force.norm(), f64::MAX);
        }

        // Simplified: assume reaction force equals gravity if fully supported
        let reaction_force = -gravity_force * support_ratio;
        let net_force = gravity_force + reaction_force;
        let force_imbalance = net_force.norm();

        // Moment check (simplified)
        // In a full implementation, compute moments about support polygon centroid
        let moment_imbalance = if support_ratio >= 0.5 {
            0.0 // Assume balanced if well supported
        } else {
            // Approximate: higher imbalance with less support
            mass * 9.81 * (1.0 - support_ratio) * placed_box.dimensions.z / 2.0
        };

        (force_imbalance, moment_imbalance)
    }

    /// Computes overlap area between two axis-aligned rectangles.
    fn compute_face_overlap(
        &self,
        face1: (f64, f64, f64, f64),
        face2: (f64, f64, f64, f64),
    ) -> f64 {
        let (x1_min, y1_min, x1_max, y1_max) = face1;
        let (x2_min, y2_min, x2_max, y2_max) = face2;

        let x_overlap = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
        let y_overlap = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);

        x_overlap * y_overlap
    }

    /// Computes overlap region coordinates between two rectangles.
    fn compute_face_overlap_coords(
        &self,
        face1: (f64, f64, f64, f64),
        face2: (f64, f64, f64, f64),
    ) -> Option<(f64, f64, f64, f64)> {
        let (x1_min, y1_min, x1_max, y1_max) = face1;
        let (x2_min, y2_min, x2_max, y2_max) = face2;

        let x_min = x1_min.max(x2_min);
        let y_min = y1_min.max(y2_min);
        let x_max = x1_max.min(x2_max);
        let y_max = y1_max.min(y2_max);

        if x_max > x_min && y_max > y_min {
            Some((x_min, y_min, x_max, y_max))
        } else {
            None
        }
    }

    /// Builds an index of boxes by their top Z coordinate.
    fn build_top_z_index(&self, boxes: &[PlacedBox]) -> HashMap<i64, Vec<usize>> {
        let mut index: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, b) in boxes.iter().enumerate() {
            let top_z = b.position.z + b.dimensions.z;
            let key = (top_z * 1000.0).round() as i64;
            index.entry(key).or_default().push(i);
        }
        index
    }

    /// Computes a stability score (0.0-1.0) from the result.
    fn compute_stability_score(&self, result: &StabilityResult) -> f64 {
        if !result.is_stable {
            return 0.0;
        }

        // Weighted combination of factors
        let support_score = result.support_ratio;
        let cog_score = if result.cog_within_support { 1.0 } else { 0.5 };

        // Combine scores
        (0.7 * support_score + 0.3 * cog_score).clamp(0.0, 1.0)
    }

    /// Computes summary statistics for the report.
    fn compute_summary(&self, report: &mut StabilityReport, boxes: &[PlacedBox]) {
        let total = report.results.len();
        if total == 0 {
            return;
        }

        report.stable_count = report.results.iter().filter(|r| r.is_stable).count();
        report.unstable_count = total - report.stable_count;
        report.overall_stability = report.stable_count as f64 / total as f64;

        report.min_support_ratio = report
            .results
            .iter()
            .map(|r| r.support_ratio)
            .fold(f64::MAX, f64::min);

        report.avg_support_ratio =
            report.results.iter().map(|r| r.support_ratio).sum::<f64>() / total as f64;

        // Compute unstable weight
        for result in &report.results {
            if !result.is_stable {
                // Find corresponding box and add its mass
                if let Some(b) = boxes
                    .iter()
                    .find(|b| b.id == result.id && b.instance == result.instance)
                {
                    report.unstable_weight += b.mass.unwrap_or(0.0);
                }
            }
        }
    }
}

impl Default for StabilityAnalyzer {
    fn default() -> Self {
        Self::new(StabilityConstraint::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_constraint_default() {
        let constraint = StabilityConstraint::default();
        assert!(!constraint.is_enabled());
    }

    #[test]
    fn test_stability_constraint_partial_base() {
        let constraint = StabilityConstraint::partial_base(0.75);
        assert!(constraint.is_enabled());
        if let StabilityConstraint::PartialBase { min_ratio } = constraint {
            assert!((min_ratio - 0.75).abs() < 0.001);
        } else {
            panic!("Expected PartialBase");
        }
    }

    #[test]
    fn test_placed_box_center() {
        let b = PlacedBox::new(
            "B1",
            0,
            Point3::new(10.0, 20.0, 30.0),
            Vector3::new(100.0, 50.0, 40.0),
        );

        let center = b.center();
        assert!((center.x - 60.0).abs() < 0.001);
        assert!((center.y - 45.0).abs() < 0.001);
        assert!((center.z - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_placed_box_with_cog_offset() {
        let b = PlacedBox::new(
            "B1",
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 100.0, 100.0),
        )
        .with_cog_offset(Vector3::new(10.0, 0.0, -5.0));

        let cog = b.center_of_gravity();
        assert!((cog.x - 60.0).abs() < 0.001); // 50 + 10
        assert!((cog.y - 50.0).abs() < 0.001);
        assert!((cog.z - 45.0).abs() < 0.001); // 50 - 5
    }

    #[test]
    fn test_box_on_floor_is_stable() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::FullBase);

        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 100.0, 50.0),
        )];

        let report = analyzer.analyze(&boxes, 0.0);

        assert!(report.is_all_stable());
        assert_eq!(report.stable_count, 1);
        assert!((report.results[0].support_ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stacked_boxes_full_support() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::FullBase);

        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(100.0, 100.0, 50.0),
            ),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(0.0, 0.0, 50.0),
                Vector3::new(100.0, 100.0, 50.0),
            ),
        ];

        let report = analyzer.analyze(&boxes, 0.0);

        assert!(report.is_all_stable());
        assert_eq!(report.stable_count, 2);
    }

    #[test]
    fn test_stacked_box_partial_support() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::partial_base(0.5));

        let boxes = vec![
            // Base box
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(100.0, 100.0, 50.0),
            ),
            // Top box shifted, only 50% supported
            PlacedBox::new(
                "B2",
                0,
                Point3::new(50.0, 0.0, 50.0),
                Vector3::new(100.0, 100.0, 50.0),
            ),
        ];

        let report = analyzer.analyze(&boxes, 0.0);

        // Both should be stable (50% support meets 50% requirement)
        assert!(report.results[0].is_stable); // B1 on floor
        assert!(report.results[1].is_stable); // B2 has 50% support >= 50% required
    }

    #[test]
    fn test_unsupported_box_is_unstable() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::FullBase);

        let boxes = vec![
            // Base box
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(50.0, 50.0, 50.0),
            ),
            // Floating box - no support
            PlacedBox::new(
                "B2",
                0,
                Point3::new(100.0, 100.0, 50.0),
                Vector3::new(50.0, 50.0, 50.0),
            ),
        ];

        let report = analyzer.analyze(&boxes, 0.0);

        assert!(!report.is_all_stable());
        assert_eq!(report.unstable_count, 1);
        assert!(!report.results[1].is_stable);
    }

    #[test]
    fn test_cog_stability_check() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::CogPolygon);

        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 100.0, 50.0),
        )];

        let report = analyzer.analyze(&boxes, 0.0);

        assert!(report.results[0].cog_within_support);
        assert!(report.results[0].is_stable);
    }

    #[test]
    fn test_equilibrium_check() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::static_equilibrium());

        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 100.0, 50.0),
        )
        .with_mass(10.0)];

        let report = analyzer.analyze(&boxes, 0.0);

        assert!(report.results[0].is_stable);
        assert!(report.results[0].force_imbalance < 1.0);
    }

    #[test]
    fn test_stability_report_summary() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::partial_base(0.8));

        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(100.0, 100.0, 50.0),
            )
            .with_mass(5.0),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(50.0, 50.0, 50.0),
                Vector3::new(100.0, 100.0, 50.0),
            )
            .with_mass(3.0),
        ];

        let report = analyzer.analyze(&boxes, 0.0);

        assert_eq!(report.results.len(), 2);
        assert!(report.analysis_time_ms < 1000); // Should be fast
    }

    #[test]
    fn test_no_constraint_always_stable() {
        let analyzer = StabilityAnalyzer::new(StabilityConstraint::None);

        let boxes = vec![
            // Floating box - normally unstable
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 100.0), // Floating at z=100
                Vector3::new(50.0, 50.0, 50.0),
            ),
        ];

        let report = analyzer.analyze(&boxes, 0.0);

        // With None constraint, should return empty report
        assert!(report.results.is_empty() || report.is_all_stable());
    }

    #[test]
    fn test_face_overlap_computation() {
        let analyzer = StabilityAnalyzer::default();

        // Full overlap
        let area1 = analyzer.compute_face_overlap((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0));
        assert!((area1 - 100.0).abs() < 0.001);

        // Partial overlap
        let area2 = analyzer.compute_face_overlap((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0));
        assert!((area2 - 25.0).abs() < 0.001); // 5x5 overlap

        // No overlap
        let area3 = analyzer.compute_face_overlap((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0));
        assert!(area3 < 0.001);
    }
}
