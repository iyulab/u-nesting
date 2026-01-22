//! Synthetic dataset generator for testing edge cases.
//!
//! Generates various types of 2D polygons for stress testing and edge case validation.

use crate::dataset::{Dataset, Item, Shape};
use rand::prelude::*;
use std::f64::consts::PI;

/// Generator for synthetic 2D benchmark datasets.
#[derive(Debug, Clone)]
pub struct SyntheticGenerator {
    rng: StdRng,
}

impl SyntheticGenerator {
    /// Creates a new generator with a random seed.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Creates a new generator with a specific seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generates a dataset of convex polygons only.
    ///
    /// Useful for testing NFP accuracy with known optimal solutions.
    pub fn convex_only(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let sides = self.rng.gen_range(3..=8);
            let radius = self.rng.gen_range(5.0..20.0);
            let demand = self.rng.gen_range(1..=5);
            let shape = self.regular_polygon(sides, radius);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0, 90.0, 180.0, 270.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_convex".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset of complex non-convex polygons.
    pub fn concave_complex(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let shape_type = self.rng.gen_range(0..4);
            let shape = match shape_type {
                0 => {
                    let points = self.rng.gen_range(5..=8);
                    self.star_polygon(points)
                }
                1 => self.l_shape(),
                2 => self.t_shape(),
                _ => self.cross_shape(),
            };
            let demand = self.rng.gen_range(1..=3);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0, 90.0, 180.0, 270.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_concave".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset of polygons with holes.
    pub fn with_holes(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let outer_radius = self.rng.gen_range(15.0..30.0);
            let hole_radius = self.rng.gen_range(3.0..outer_radius * 0.4);
            let outer_sides = self.rng.gen_range(4..=8);
            let hole_sides = self.rng.gen_range(3..=6);
            let demand = self.rng.gen_range(1..=3);

            let outer = self.regular_polygon(outer_sides, outer_radius);
            let hole = self.regular_polygon(hole_sides, hole_radius);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0, 90.0, 180.0, 270.0],
                shape: Shape::PolygonWithHoles {
                    outer,
                    holes: vec![hole],
                },
            });
        }

        Dataset {
            name: "synthetic_with_holes".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset with extreme aspect ratios.
    ///
    /// Tests handling of very long/thin or very short/wide polygons.
    pub fn extreme_aspect(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let is_tall = self.rng.gen_bool(0.5);
            let shape = if is_tall {
                // Very tall and thin
                let width = self.rng.gen_range(2.0..5.0);
                let height = self.rng.gen_range(40.0..80.0);
                self.rectangle(width, height)
            } else {
                // Very wide and short
                let width = self.rng.gen_range(40.0..80.0);
                let height = self.rng.gen_range(2.0..5.0);
                self.rectangle(width, height)
            };
            let demand = self.rng.gen_range(1..=3);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0, 90.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_extreme_aspect".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset with very small items (precision testing).
    pub fn tiny_items(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let size = self.rng.gen_range(0.1..2.0);
            let sides = self.rng.gen_range(3..=6);
            let demand = self.rng.gen_range(5..=20);
            let shape = self.regular_polygon(sides, size);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_tiny".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a large-scale dataset (stress testing).
    pub fn large_count(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let shape_type = self.rng.gen_range(0..3);
            let shape = match shape_type {
                0 => {
                    let sides = self.rng.gen_range(3..=6);
                    let radius = self.rng.gen_range(5.0..15.0);
                    self.regular_polygon(sides, radius)
                }
                1 => {
                    let w = self.rng.gen_range(5.0..20.0);
                    let h = self.rng.gen_range(5.0..20.0);
                    self.rectangle(w, h)
                }
                _ => self.l_shape(),
            };

            items.push(Item {
                id,
                demand: 1,
                allowed_orientations: vec![0.0, 90.0, 180.0, 270.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_large".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset with near-collinear edges (numerical stability).
    pub fn near_collinear(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            // Create a rectangle with small perturbations to edges
            let w = self.rng.gen_range(10.0..20.0);
            let h = self.rng.gen_range(10.0..20.0);
            let epsilon = 1e-6;
            let demand = self.rng.gen_range(1..=3);

            let shape = vec![
                [0.0, 0.0],
                [w * 0.3, epsilon],  // Nearly collinear
                [w * 0.7, -epsilon], // Nearly collinear
                [w, 0.0],
                [w, h * 0.5 + epsilon], // Nearly collinear
                [w, h],
                [w * 0.5, h - epsilon], // Nearly collinear
                [0.0, h],
                [0.0, 0.0],
            ];

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_near_collinear".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a dataset with self-touching polygons.
    ///
    /// Note: Some systems may not handle these correctly.
    pub fn self_touching(&mut self, count: usize, strip_height: f64) -> Dataset {
        let mut items = Vec::with_capacity(count);
        for id in 0..count {
            let size = self.rng.gen_range(10.0..20.0);
            let demand = self.rng.gen_range(1..=2);
            // Figure-8 like shape that touches itself at a point
            let shape = self.figure_eight(size);

            items.push(Item {
                id,
                demand,
                allowed_orientations: vec![0.0, 90.0],
                shape: Shape::SimplePolygon(shape),
            });
        }

        Dataset {
            name: "synthetic_self_touching".to_string(),
            items,
            strip_width: None,
            strip_height,
            best_known: None,
        }
    }

    /// Generates a "jigsaw puzzle" dataset where pieces should fit perfectly.
    ///
    /// Returns a dataset with known 100% utilization solution.
    pub fn jigsaw_puzzle(&mut self, grid_size: usize, cell_size: f64) -> Dataset {
        let mut items = Vec::new();
        let mut id = 0;

        // Create interlocking pieces based on a grid
        for row in 0..grid_size {
            for col in 0..grid_size {
                let shape = self.jigsaw_piece(
                    cell_size,
                    row == 0,
                    col == grid_size - 1,
                    row == grid_size - 1,
                    col == 0,
                );
                items.push(Item {
                    id,
                    demand: 1,
                    allowed_orientations: vec![0.0],
                    shape: Shape::SimplePolygon(shape),
                });
                id += 1;
            }
        }

        let total_size = grid_size as f64 * cell_size;

        Dataset {
            name: format!("synthetic_jigsaw_{}x{}", grid_size, grid_size),
            items,
            strip_width: None,
            strip_height: total_size,
            best_known: Some(total_size), // Perfect packing should fit in a square
        }
    }

    // === Helper methods for generating specific shapes ===

    fn regular_polygon(&self, sides: usize, radius: f64) -> Vec<[f64; 2]> {
        let mut points = Vec::with_capacity(sides + 1);
        for i in 0..sides {
            let angle = 2.0 * PI * (i as f64) / (sides as f64) - PI / 2.0;
            points.push([radius * angle.cos(), radius * angle.sin()]);
        }
        points.push(points[0]); // Close the polygon
        points
    }

    fn rectangle(&self, width: f64, height: f64) -> Vec<[f64; 2]> {
        vec![
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
            [0.0, 0.0],
        ]
    }

    fn star_polygon(&mut self, points: usize) -> Vec<[f64; 2]> {
        let outer_radius = self.rng.gen_range(15.0..25.0);
        let inner_radius = outer_radius * self.rng.gen_range(0.3..0.5);

        let mut vertices = Vec::with_capacity(points * 2 + 1);
        for i in 0..(points * 2) {
            let angle = PI * (i as f64) / (points as f64) - PI / 2.0;
            let radius = if i % 2 == 0 {
                outer_radius
            } else {
                inner_radius
            };
            vertices.push([radius * angle.cos(), radius * angle.sin()]);
        }
        vertices.push(vertices[0]);
        vertices
    }

    fn l_shape(&mut self) -> Vec<[f64; 2]> {
        let w = self.rng.gen_range(15.0..25.0);
        let h = self.rng.gen_range(15.0..25.0);
        let notch_w = w * self.rng.gen_range(0.4..0.6);
        let notch_h = h * self.rng.gen_range(0.4..0.6);

        vec![
            [0.0, 0.0],
            [w, 0.0],
            [w, h - notch_h],
            [w - notch_w, h - notch_h],
            [w - notch_w, h],
            [0.0, h],
            [0.0, 0.0],
        ]
    }

    fn t_shape(&mut self) -> Vec<[f64; 2]> {
        let w = self.rng.gen_range(20.0..30.0);
        let h = self.rng.gen_range(20.0..30.0);
        let stem_w = w * 0.3;
        let stem_h = h * 0.6;
        let left_offset = (w - stem_w) / 2.0;

        vec![
            [0.0, h - stem_h],
            [left_offset, h - stem_h],
            [left_offset, 0.0],
            [left_offset + stem_w, 0.0],
            [left_offset + stem_w, h - stem_h],
            [w, h - stem_h],
            [w, h],
            [0.0, h],
            [0.0, h - stem_h],
        ]
    }

    fn cross_shape(&mut self) -> Vec<[f64; 2]> {
        let size = self.rng.gen_range(20.0..30.0);
        let arm = size * 0.3;
        let center = (size - arm) / 2.0;

        vec![
            [center, 0.0],
            [center + arm, 0.0],
            [center + arm, center],
            [size, center],
            [size, center + arm],
            [center + arm, center + arm],
            [center + arm, size],
            [center, size],
            [center, center + arm],
            [0.0, center + arm],
            [0.0, center],
            [center, center],
            [center, 0.0],
        ]
    }

    fn figure_eight(&self, size: f64) -> Vec<[f64; 2]> {
        // Create a shape that comes close to touching itself
        let half = size / 2.0;
        let small = size * 0.2;

        vec![
            [half, 0.0],
            [size, small],
            [size, half - small],
            [half + small, half], // Near touch point
            [size, half + small],
            [size, size - small],
            [half, size],
            [small, size - small],
            [0.0, half + small],
            [half - small, half], // Near touch point (other side)
            [0.0, half - small],
            [small, small],
            [half, 0.0],
        ]
    }

    fn jigsaw_piece(
        &self,
        size: f64,
        is_top: bool,
        is_right: bool,
        is_bottom: bool,
        is_left: bool,
    ) -> Vec<[f64; 2]> {
        // Simplified jigsaw piece - just a rectangle with optional tabs
        let tab = size * 0.1;
        let half = size / 2.0;

        let mut points = vec![[0.0, 0.0]];

        // Bottom edge
        if is_bottom {
            points.push([size, 0.0]);
        } else {
            points.push([half - tab, 0.0]);
            points.push([half - tab, -tab]);
            points.push([half + tab, -tab]);
            points.push([half + tab, 0.0]);
            points.push([size, 0.0]);
        }

        // Right edge
        if is_right {
            points.push([size, size]);
        } else {
            points.push([size, half - tab]);
            points.push([size + tab, half - tab]);
            points.push([size + tab, half + tab]);
            points.push([size, half + tab]);
            points.push([size, size]);
        }

        // Top edge
        if is_top {
            points.push([0.0, size]);
        } else {
            points.push([half + tab, size]);
            points.push([half + tab, size + tab]);
            points.push([half - tab, size + tab]);
            points.push([half - tab, size]);
            points.push([0.0, size]);
        }

        // Left edge
        if is_left {
            points.push([0.0, 0.0]);
        } else {
            points.push([0.0, half + tab]);
            points.push([-tab, half + tab]);
            points.push([-tab, half - tab]);
            points.push([0.0, half - tab]);
            points.push([0.0, 0.0]);
        }

        points
    }
}

impl Default for SyntheticGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined synthetic datasets for benchmarking.
pub struct SyntheticDatasets;

impl SyntheticDatasets {
    /// Generates all standard synthetic datasets with default parameters.
    pub fn all(seed: u64) -> Vec<Dataset> {
        let mut gen = SyntheticGenerator::with_seed(seed);
        vec![
            gen.convex_only(20, 100.0),
            gen.concave_complex(15, 100.0),
            gen.with_holes(10, 150.0),
            gen.extreme_aspect(15, 100.0),
            gen.tiny_items(30, 50.0),
            gen.large_count(100, 200.0),
            gen.near_collinear(10, 100.0),
            gen.jigsaw_puzzle(4, 25.0),
        ]
    }

    /// Returns the names of all synthetic datasets.
    pub fn names() -> &'static [&'static str] {
        &[
            "synthetic_convex",
            "synthetic_concave",
            "synthetic_with_holes",
            "synthetic_extreme_aspect",
            "synthetic_tiny",
            "synthetic_large",
            "synthetic_near_collinear",
            "synthetic_jigsaw_4x4",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_generation() {
        let mut gen = SyntheticGenerator::with_seed(42);
        let dataset = gen.convex_only(10, 100.0);
        assert_eq!(dataset.items.len(), 10);
        assert_eq!(dataset.strip_height, 100.0);
    }

    #[test]
    fn test_reproducibility() {
        let mut gen1 = SyntheticGenerator::with_seed(123);
        let mut gen2 = SyntheticGenerator::with_seed(123);

        let ds1 = gen1.convex_only(5, 50.0);
        let ds2 = gen2.convex_only(5, 50.0);

        // Same seed should produce same data
        for (i, (item1, item2)) in ds1.items.iter().zip(ds2.items.iter()).enumerate() {
            assert_eq!(item1.demand, item2.demand, "Mismatch at item {}", i);
        }
    }

    #[test]
    fn test_jigsaw_puzzle() {
        let mut gen = SyntheticGenerator::with_seed(42);
        let dataset = gen.jigsaw_puzzle(3, 10.0);
        assert_eq!(dataset.items.len(), 9); // 3x3 grid
        assert!(dataset.best_known.is_some());
    }
}
