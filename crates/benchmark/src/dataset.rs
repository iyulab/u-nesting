//! Dataset types for ESICUP benchmark instances.

use serde::{Deserialize, Serialize};

/// Information about a benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name (e.g., "SHAPES", "SHIRTS", "SWIM")
    pub name: String,
    /// Short description
    pub description: Option<String>,
    /// Source/reference
    pub source: Option<String>,
    /// Number of item types
    pub item_types: usize,
    /// Total pieces when demand is expanded
    pub total_pieces: usize,
    /// Strip height (width is infinite for strip packing)
    pub strip_height: f64,
    /// Best known solution (strip length if known)
    pub best_known: Option<f64>,
}

/// A parsed benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Dataset name
    pub name: String,
    /// Items to be placed
    pub items: Vec<Item>,
    /// Strip width (optional, if None will be estimated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strip_width: Option<f64>,
    /// Strip height (container height for strip packing)
    pub strip_height: f64,
    /// Best known solution (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_known: Option<f64>,
}

impl Dataset {
    /// Returns dataset information.
    pub fn info(&self) -> DatasetInfo {
        let total_pieces: usize = self.items.iter().map(|i| i.demand).sum();
        DatasetInfo {
            name: self.name.clone(),
            description: None,
            source: None,
            item_types: self.items.len(),
            total_pieces,
            strip_height: self.strip_height,
            best_known: self.best_known,
        }
    }

    /// Total area of every piece, demand included.
    pub fn total_area(&self) -> f64 {
        self.items
            .iter()
            .map(|i| i.shape.area().abs() * i.demand as f64)
            .sum()
    }

    /// The strip length at which these pieces fill `density` of the strip
    /// (`total_area / (strip_height * density)`), or `None` when the density is
    /// outside (0, 1] or the strip has no height.
    pub fn length_at_density(&self, density: f64) -> Option<f64> {
        (density > 0.0 && density <= 1.0 && self.strip_height > 0.0)
            .then(|| self.total_area() / (self.strip_height * density))
    }

    /// Expands items by demand, returning individual pieces.
    /// Each piece gets a unique ID.
    pub fn expand_items(&self) -> Vec<ExpandedItem> {
        let mut expanded = Vec::new();
        let mut piece_id = 0;
        for item in &self.items {
            for _ in 0..item.demand {
                expanded.push(ExpandedItem {
                    piece_id,
                    item_id: item.id,
                    shape: item.shape.clone(),
                    allowed_orientations: item.allowed_orientations.clone(),
                });
                piece_id += 1;
            }
        }
        expanded
    }
}

/// An item type in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    /// Item type ID
    pub id: usize,
    /// Number of copies needed
    pub demand: usize,
    /// Allowed rotation angles in degrees
    pub allowed_orientations: Vec<f64>,
    /// Shape definition
    pub shape: Shape,
}

/// An expanded item (single piece after demand expansion).
#[derive(Debug, Clone)]
pub struct ExpandedItem {
    /// Unique piece ID
    pub piece_id: usize,
    /// Original item type ID
    pub item_id: usize,
    /// Shape definition
    pub shape: Shape,
    /// Allowed rotation angles in degrees
    pub allowed_orientations: Vec<f64>,
}

/// Shape definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Shape {
    /// Simple polygon without holes
    #[serde(rename = "simple_polygon")]
    SimplePolygon(Vec<[f64; 2]>),

    /// Polygon with holes
    #[serde(rename = "polygon_with_holes")]
    PolygonWithHoles {
        /// Outer boundary (counter-clockwise)
        outer: Vec<[f64; 2]>,
        /// Inner holes (clockwise)
        holes: Vec<Vec<[f64; 2]>>,
    },

    /// Multi-polygon (multiple separate polygons)
    #[serde(rename = "multi_polygon")]
    MultiPolygon(Vec<Vec<[f64; 2]>>),
}

impl Shape {
    /// Computes the bounding box of the shape.
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
        let points = self.all_points();
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for [x, y] in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        (min_x, min_y, max_x, max_y)
    }

    /// Returns all points in the shape.
    fn all_points(&self) -> Vec<[f64; 2]> {
        match self {
            Shape::SimplePolygon(points) => points.clone(),
            Shape::PolygonWithHoles { outer, holes } => {
                let mut all = outer.clone();
                for hole in holes {
                    all.extend(hole.iter().cloned());
                }
                all
            }
            Shape::MultiPolygon(polygons) => polygons.iter().flatten().cloned().collect(),
        }
    }

    /// Computes the approximate area of the shape.
    pub fn area(&self) -> f64 {
        match self {
            Shape::SimplePolygon(points) => polygon_area(points),
            Shape::PolygonWithHoles { outer, holes } => {
                let outer_area = polygon_area(outer);
                let holes_area: f64 = holes.iter().map(|h| polygon_area(h).abs()).sum();
                outer_area.abs() - holes_area
            }
            Shape::MultiPolygon(polygons) => polygons.iter().map(|p| polygon_area(p).abs()).sum(),
        }
    }
}

/// Computes the signed area of a polygon using the shoelace formula.
fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }
    area / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The best-known length is derived from the instance's own pieces and strip:
    /// four 1x1 squares on a strip 2 high fill a 2-long strip at density 1, a
    /// 2.5-long one at 0.8.
    #[test]
    fn length_at_density_is_area_over_height_and_density() {
        let square = Shape::SimplePolygon(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let ds = Dataset {
            name: "t".into(),
            items: vec![Item {
                id: 0,
                demand: 4,
                allowed_orientations: vec![0.0],
                shape: square,
            }],
            strip_width: None,
            strip_height: 2.0,
            best_known: None,
        };
        assert_eq!(ds.total_area(), 4.0);
        assert_eq!(ds.length_at_density(1.0), Some(2.0));
        assert_eq!(ds.length_at_density(0.8), Some(2.5));
        assert_eq!(ds.length_at_density(1.05), None);
        assert_eq!(ds.length_at_density(0.0), None);
    }

    #[test]
    fn test_simple_polygon_area() {
        // Unit square
        let square = Shape::SimplePolygon(vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]);
        assert!((square.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box() {
        let shape = Shape::SimplePolygon(vec![
            [1.0, 2.0],
            [5.0, 2.0],
            [5.0, 8.0],
            [1.0, 8.0],
            [1.0, 2.0],
        ]);
        let (min_x, min_y, max_x, max_y) = shape.bounding_box();
        assert!((min_x - 1.0).abs() < 1e-10);
        assert!((min_y - 2.0).abs() < 1e-10);
        assert!((max_x - 5.0).abs() < 1e-10);
        assert!((max_y - 8.0).abs() < 1e-10);
    }
}
