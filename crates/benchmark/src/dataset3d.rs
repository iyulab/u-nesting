//! 3D benchmark dataset types and generators.
//!
//! Based on Martello, Pisinger, Vigo (2000) instance generation methodology.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Information about a 3D benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset3DInfo {
    /// Dataset name
    pub name: String,
    /// Instance class (MPV1-8, etc.)
    pub instance_class: InstanceClass,
    /// Number of items
    pub num_items: usize,
    /// Bin dimensions (W, H, D)
    pub bin_dimensions: [f64; 3],
    /// Lower bound on bins needed (if known)
    pub lower_bound: Option<usize>,
    /// Best known solution (bins used, if known)
    pub best_known: Option<usize>,
}

/// A 3D bin packing benchmark dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset3D {
    /// Dataset name
    pub name: String,
    /// Instance class
    pub instance_class: InstanceClass,
    /// Items to be packed
    pub items: Vec<Item3D>,
    /// Bin dimensions (W, H, D)
    pub bin_dimensions: [f64; 3],
    /// Lower bound on bins needed (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_bound: Option<usize>,
    /// Best known solution (bins used, if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_known: Option<usize>,
}

impl Dataset3D {
    /// Returns dataset information.
    pub fn info(&self) -> Dataset3DInfo {
        Dataset3DInfo {
            name: self.name.clone(),
            instance_class: self.instance_class,
            num_items: self.items.len(),
            bin_dimensions: self.bin_dimensions,
            lower_bound: self.lower_bound,
            best_known: self.best_known,
        }
    }

    /// Total volume of all items.
    pub fn total_item_volume(&self) -> f64 {
        self.items.iter().map(|i| i.volume()).sum()
    }

    /// Bin volume.
    pub fn bin_volume(&self) -> f64 {
        self.bin_dimensions[0] * self.bin_dimensions[1] * self.bin_dimensions[2]
    }

    /// Theoretical lower bound based on volume.
    pub fn volume_lower_bound(&self) -> usize {
        (self.total_item_volume() / self.bin_volume()).ceil() as usize
    }
}

/// A 3D item (box) to be packed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item3D {
    /// Item ID
    pub id: usize,
    /// Dimensions (width, height, depth)
    pub dimensions: [f64; 3],
    /// Quantity (demand)
    #[serde(default = "default_quantity")]
    pub quantity: usize,
    /// Allowed orientations
    #[serde(default = "default_orientations")]
    pub allowed_orientations: OrientationConstraint,
}

fn default_quantity() -> usize {
    1
}

fn default_orientations() -> OrientationConstraint {
    OrientationConstraint::Any
}

impl Item3D {
    /// Creates a new item.
    pub fn new(id: usize, width: f64, height: f64, depth: f64) -> Self {
        Self {
            id,
            dimensions: [width, height, depth],
            quantity: 1,
            allowed_orientations: OrientationConstraint::Any,
        }
    }

    /// Sets the quantity.
    pub fn with_quantity(mut self, quantity: usize) -> Self {
        self.quantity = quantity;
        self
    }

    /// Sets the orientation constraint.
    pub fn with_orientations(mut self, orientations: OrientationConstraint) -> Self {
        self.allowed_orientations = orientations;
        self
    }

    /// Item volume.
    pub fn volume(&self) -> f64 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2] * self.quantity as f64
    }

    /// Width.
    pub fn width(&self) -> f64 {
        self.dimensions[0]
    }

    /// Height.
    pub fn height(&self) -> f64 {
        self.dimensions[1]
    }

    /// Depth.
    pub fn depth(&self) -> f64 {
        self.dimensions[2]
    }
}

/// Orientation constraints for 3D items.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrientationConstraint {
    /// Any of 6 axis-aligned orientations allowed
    Any,
    /// Only upright orientations (height stays vertical)
    Upright,
    /// Fixed orientation only
    Fixed,
}

/// Instance class based on MPV (Martello-Pisinger-Vigo) methodology.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InstanceClass {
    /// Type 1: Small width, large height and depth
    /// w: [1, W/2], h: [2H/3, H], d: [2D/3, D]
    MPV1,
    /// Type 2: Large width, small height, large depth
    /// w: [2W/3, W], h: [1, H/2], d: [2D/3, D]
    MPV2,
    /// Type 3: Large width and height, small depth
    /// w: [2W/3, W], h: [2H/3, H], d: [1, D/2]
    MPV3,
    /// Type 4: Medium-sized items (half of each dimension)
    /// w: [W/2, W], h: [H/2, H], d: [D/2, D]
    MPV4,
    /// Type 5: Small items (quarter of each dimension)
    /// w: [1, W/2], h: [1, H/2], d: [1, D/2]
    MPV5,
    /// Type 6: Berkey-Wang uniform [1, 10]
    BW6,
    /// Type 7: Berkey-Wang uniform [1, 35]
    BW7,
    /// Type 8: Berkey-Wang uniform [1, 100]
    BW8,
    /// Custom instance class
    Custom,
}

impl InstanceClass {
    /// Returns a string identifier for the class.
    pub fn id(&self) -> &'static str {
        match self {
            InstanceClass::MPV1 => "MPV1",
            InstanceClass::MPV2 => "MPV2",
            InstanceClass::MPV3 => "MPV3",
            InstanceClass::MPV4 => "MPV4",
            InstanceClass::MPV5 => "MPV5",
            InstanceClass::BW6 => "BW6",
            InstanceClass::BW7 => "BW7",
            InstanceClass::BW8 => "BW8",
            InstanceClass::Custom => "Custom",
        }
    }

    /// All standard instance classes.
    pub fn all_standard() -> Vec<InstanceClass> {
        vec![
            InstanceClass::MPV1,
            InstanceClass::MPV2,
            InstanceClass::MPV3,
            InstanceClass::MPV4,
            InstanceClass::MPV5,
            InstanceClass::BW6,
            InstanceClass::BW7,
            InstanceClass::BW8,
        ]
    }
}

/// 3D instance generator following MPV methodology.
pub struct InstanceGenerator {
    bin_dim: f64,
    seed: Option<u64>,
}

impl InstanceGenerator {
    /// Creates a new generator with specified bin dimension.
    /// MPV uses cubic bins where W = H = D = bin_dim.
    pub fn new(bin_dim: f64) -> Self {
        Self {
            bin_dim,
            seed: None,
        }
    }

    /// Creates a generator with default bin dimension (100).
    pub fn default_bin() -> Self {
        Self::new(100.0)
    }

    /// Sets the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generates an instance of the specified class.
    pub fn generate(&self, class: InstanceClass, num_items: usize) -> Dataset3D {
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(thread_rng()),
        };

        let items: Vec<Item3D> = (0..num_items)
            .map(|id| self.generate_item(class, id, &mut rng))
            .collect();

        let name = format!("{}_{}_n{}", class.id(), self.bin_dim as u32, num_items);

        Dataset3D {
            name,
            instance_class: class,
            items,
            bin_dimensions: [self.bin_dim, self.bin_dim, self.bin_dim],
            lower_bound: None,
            best_known: None,
        }
    }

    /// Generates a single item based on class constraints.
    fn generate_item(&self, class: InstanceClass, id: usize, rng: &mut dyn RngCore) -> Item3D {
        let (w, h, d) = match class {
            InstanceClass::MPV1 => {
                // Small width, large height and depth
                let w = rng.gen_range(1.0..=self.bin_dim / 2.0);
                let h = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                let d = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                (w, h, d)
            }
            InstanceClass::MPV2 => {
                // Large width, small height, large depth
                let w = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                let h = rng.gen_range(1.0..=self.bin_dim / 2.0);
                let d = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                (w, h, d)
            }
            InstanceClass::MPV3 => {
                // Large width and height, small depth
                let w = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                let h = rng.gen_range(2.0 * self.bin_dim / 3.0..=self.bin_dim);
                let d = rng.gen_range(1.0..=self.bin_dim / 2.0);
                (w, h, d)
            }
            InstanceClass::MPV4 => {
                // Medium-sized items
                let w = rng.gen_range(self.bin_dim / 2.0..=self.bin_dim);
                let h = rng.gen_range(self.bin_dim / 2.0..=self.bin_dim);
                let d = rng.gen_range(self.bin_dim / 2.0..=self.bin_dim);
                (w, h, d)
            }
            InstanceClass::MPV5 => {
                // Small items
                let w = rng.gen_range(1.0..=self.bin_dim / 2.0);
                let h = rng.gen_range(1.0..=self.bin_dim / 2.0);
                let d = rng.gen_range(1.0..=self.bin_dim / 2.0);
                (w, h, d)
            }
            InstanceClass::BW6 => {
                // Berkey-Wang [1, 10]
                let w = rng.gen_range(1.0..=10.0);
                let h = rng.gen_range(1.0..=10.0);
                let d = rng.gen_range(1.0..=10.0);
                (w, h, d)
            }
            InstanceClass::BW7 => {
                // Berkey-Wang [1, 35]
                let w = rng.gen_range(1.0..=35.0);
                let h = rng.gen_range(1.0..=35.0);
                let d = rng.gen_range(1.0..=35.0);
                (w, h, d)
            }
            InstanceClass::BW8 => {
                // Berkey-Wang [1, 100]
                let w = rng.gen_range(1.0..=100.0);
                let h = rng.gen_range(1.0..=100.0);
                let d = rng.gen_range(1.0..=100.0);
                (w, h, d)
            }
            InstanceClass::Custom => {
                // Default to uniform [1, bin_dim]
                let w = rng.gen_range(1.0..=self.bin_dim);
                let h = rng.gen_range(1.0..=self.bin_dim);
                let d = rng.gen_range(1.0..=self.bin_dim);
                (w, h, d)
            }
        };

        Item3D::new(id, w.round(), h.round(), d.round())
    }

    /// Generates a batch of instances for all standard classes.
    pub fn generate_batch(&self, num_items: usize, instances_per_class: usize) -> Vec<Dataset3D> {
        let mut datasets = Vec::new();

        for class in InstanceClass::all_standard() {
            for i in 0..instances_per_class {
                let gen = InstanceGenerator::new(self.bin_dim).with_seed(i as u64 * 12345 + 1);
                let mut dataset = gen.generate(class, num_items);
                dataset.name = format!(
                    "{}_{}_n{}_i{}",
                    class.id(),
                    self.bin_dim as u32,
                    num_items,
                    i + 1
                );
                datasets.push(dataset);
            }
        }

        datasets
    }
}

impl Default for InstanceGenerator {
    fn default() -> Self {
        Self::default_bin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_mpv1() {
        let gen = InstanceGenerator::new(100.0).with_seed(42);
        let dataset = gen.generate(InstanceClass::MPV1, 10);

        assert_eq!(dataset.items.len(), 10);
        assert_eq!(dataset.bin_dimensions, [100.0, 100.0, 100.0]);

        // Verify MPV1 constraints: small width, large height/depth
        for item in &dataset.items {
            assert!(item.width() >= 1.0 && item.width() <= 50.0);
            assert!(item.height() >= 66.0 && item.height() <= 100.0);
            assert!(item.depth() >= 66.0 && item.depth() <= 100.0);
        }
    }

    #[test]
    fn test_generate_mpv5() {
        let gen = InstanceGenerator::new(100.0).with_seed(42);
        let dataset = gen.generate(InstanceClass::MPV5, 20);

        assert_eq!(dataset.items.len(), 20);

        // Verify MPV5 constraints: all small
        for item in &dataset.items {
            assert!(item.width() >= 1.0 && item.width() <= 50.0);
            assert!(item.height() >= 1.0 && item.height() <= 50.0);
            assert!(item.depth() >= 1.0 && item.depth() <= 50.0);
        }
    }

    #[test]
    fn test_volume_lower_bound() {
        let gen = InstanceGenerator::new(100.0).with_seed(42);
        let dataset = gen.generate(InstanceClass::MPV5, 50);

        let lb = dataset.volume_lower_bound();
        assert!(lb >= 1);
    }

    #[test]
    fn test_generate_batch() {
        let gen = InstanceGenerator::new(100.0);
        let datasets = gen.generate_batch(10, 2);

        // 8 classes * 2 instances each = 16 datasets
        assert_eq!(datasets.len(), 16);
    }

    #[test]
    fn test_deterministic_generation() {
        let gen1 = InstanceGenerator::new(100.0).with_seed(12345);
        let gen2 = InstanceGenerator::new(100.0).with_seed(12345);

        let ds1 = gen1.generate(InstanceClass::MPV1, 5);
        let ds2 = gen2.generate(InstanceClass::MPV1, 5);

        for (item1, item2) in ds1.items.iter().zip(ds2.items.iter()) {
            assert_eq!(item1.dimensions, item2.dimensions);
        }
    }
}
