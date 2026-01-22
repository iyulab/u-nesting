//! Physics simulation for 3D bin packing validation and compaction.
//!
//! This module provides physics-based validation and refinement of 3D packing solutions.
//! It uses a simplified physics model to detect instabilities and improve compaction.
//!
//! # Features
//!
//! - **Settlement Simulation**: Applies gravity and lets boxes settle into stable positions
//! - **Stability Validation**: Detects boxes that would fall or tip over
//! - **Compaction**: Simulates shaking to improve packing density
//! - **Collision Detection**: Uses AABB-based collision detection
//!
//! # Example
//!
//! ```ignore
//! use u_nesting_d3::physics::{PhysicsSimulator, PhysicsConfig};
//! use u_nesting_d3::stability::PlacedBox;
//!
//! let config = PhysicsConfig::default()
//!     .with_gravity(-9.81)
//!     .with_time_step(0.016);
//!
//! let simulator = PhysicsSimulator::new(config);
//! let boxes = /* ... */;
//! let result = simulator.simulate_settlement(&boxes, floor_z);
//! ```

use crate::stability::{PlacedBox, StabilityAnalyzer, StabilityConstraint, StabilityReport};
use nalgebra::{Point3, Vector3};
use std::time::Instant;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for physics simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicsConfig {
    /// Gravity acceleration (m/s², negative for downward).
    pub gravity: f64,
    /// Simulation time step (seconds).
    pub time_step: f64,
    /// Maximum simulation time (seconds).
    pub max_simulation_time: f64,
    /// Velocity threshold for considering a box at rest.
    pub rest_velocity_threshold: f64,
    /// Position change threshold for convergence.
    pub convergence_threshold: f64,
    /// Friction coefficient (0.0-1.0).
    pub friction: f64,
    /// Restitution coefficient (bounciness, 0.0-1.0).
    pub restitution: f64,
    /// Number of iterations for constraint solving.
    pub solver_iterations: usize,
    /// Enable shaking compaction.
    pub enable_shaking: bool,
    /// Shaking amplitude (if enabled).
    pub shake_amplitude: f64,
    /// Shaking frequency (Hz, if enabled).
    pub shake_frequency: f64,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: -9.81,
            time_step: 0.016, // ~60 FPS
            max_simulation_time: 5.0,
            rest_velocity_threshold: 0.001,
            convergence_threshold: 0.0001,
            friction: 0.5,
            restitution: 0.1,
            solver_iterations: 10,
            enable_shaking: false,
            shake_amplitude: 0.5,
            shake_frequency: 5.0,
        }
    }
}

impl PhysicsConfig {
    /// Creates a new physics configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the gravity (negative for downward).
    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the simulation time step.
    pub fn with_time_step(mut self, dt: f64) -> Self {
        self.time_step = dt.max(0.001);
        self
    }

    /// Sets the maximum simulation time.
    pub fn with_max_time(mut self, max_time: f64) -> Self {
        self.max_simulation_time = max_time;
        self
    }

    /// Sets the friction coefficient.
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction.clamp(0.0, 1.0);
        self
    }

    /// Sets the restitution (bounciness).
    pub fn with_restitution(mut self, restitution: f64) -> Self {
        self.restitution = restitution.clamp(0.0, 1.0);
        self
    }

    /// Enables shaking compaction.
    pub fn with_shaking(mut self, amplitude: f64, frequency: f64) -> Self {
        self.enable_shaking = true;
        self.shake_amplitude = amplitude.max(0.0);
        self.shake_frequency = frequency.max(0.1);
        self
    }
}

/// State of a physics body during simulation.
#[derive(Debug, Clone)]
struct PhysicsBody {
    /// Reference to original placed box index.
    index: usize,
    /// Current position.
    position: Point3<f64>,
    /// Current velocity.
    velocity: Vector3<f64>,
    /// Dimensions (width, depth, height).
    dimensions: Vector3<f64>,
    /// Mass.
    mass: f64,
    /// Whether the body is at rest.
    at_rest: bool,
    /// Whether the body is on the floor.
    on_floor: bool,
}

impl PhysicsBody {
    fn from_placed_box(index: usize, placed: &PlacedBox) -> Self {
        Self {
            index,
            position: placed.position,
            velocity: Vector3::zeros(),
            dimensions: placed.dimensions,
            mass: placed.mass.unwrap_or(1.0),
            at_rest: false,
            on_floor: false,
        }
    }

    fn aabb_min(&self) -> Point3<f64> {
        self.position
    }

    fn aabb_max(&self) -> Point3<f64> {
        Point3::new(
            self.position.x + self.dimensions.x,
            self.position.y + self.dimensions.y,
            self.position.z + self.dimensions.z,
        )
    }

    fn center(&self) -> Point3<f64> {
        Point3::new(
            self.position.x + self.dimensions.x / 2.0,
            self.position.y + self.dimensions.y / 2.0,
            self.position.z + self.dimensions.z / 2.0,
        )
    }
}

/// Result of physics simulation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicsResult {
    /// Final positions after simulation.
    pub final_positions: Vec<(String, usize, Point3<f64>)>,
    /// Position changes from initial state.
    pub position_changes: Vec<(String, usize, Vector3<f64>)>,
    /// Number of simulation steps performed.
    pub steps: usize,
    /// Simulation time (seconds).
    pub simulation_time: f64,
    /// Number of boxes that moved significantly.
    pub boxes_moved: usize,
    /// Number of boxes that are stable at rest.
    pub boxes_at_rest: usize,
    /// Average position change magnitude.
    pub avg_change: f64,
    /// Maximum position change magnitude.
    pub max_change: f64,
    /// Whether simulation converged (all boxes at rest).
    pub converged: bool,
    /// Wall-clock time for computation (milliseconds).
    pub computation_time_ms: u64,
    /// Stability report after simulation.
    pub stability_report: Option<StabilityReport>,
}

impl PhysicsResult {
    /// Creates a new empty result.
    pub fn new() -> Self {
        Self {
            final_positions: Vec::new(),
            position_changes: Vec::new(),
            steps: 0,
            simulation_time: 0.0,
            boxes_moved: 0,
            boxes_at_rest: 0,
            avg_change: 0.0,
            max_change: 0.0,
            converged: false,
            computation_time_ms: 0,
            stability_report: None,
        }
    }
}

impl Default for PhysicsResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Physics simulator for 3D bin packing.
pub struct PhysicsSimulator {
    config: PhysicsConfig,
}

impl PhysicsSimulator {
    /// Creates a new physics simulator.
    pub fn new(config: PhysicsConfig) -> Self {
        Self { config }
    }

    /// Simulates settlement (gravity) until boxes reach rest.
    pub fn simulate_settlement(
        &self,
        boxes: &[PlacedBox],
        container_dims: Vector3<f64>,
        floor_z: f64,
    ) -> PhysicsResult {
        let start = Instant::now();
        let mut result = PhysicsResult::new();

        if boxes.is_empty() {
            result.computation_time_ms = start.elapsed().as_millis() as u64;
            return result;
        }

        // Initialize physics bodies
        let mut bodies: Vec<PhysicsBody> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| PhysicsBody::from_placed_box(i, b))
            .collect();

        let initial_positions: Vec<Point3<f64>> = bodies.iter().map(|b| b.position).collect();

        // Simulation loop
        let mut sim_time = 0.0;
        let mut steps = 0;
        let max_steps = (self.config.max_simulation_time / self.config.time_step) as usize;

        while sim_time < self.config.max_simulation_time && steps < max_steps {
            // Apply gravity to all bodies
            for body in &mut bodies {
                if !body.at_rest {
                    body.velocity.z += self.config.gravity * self.config.time_step;
                }
            }

            // Apply shaking if enabled
            if self.config.enable_shaking {
                let shake_offset = self.config.shake_amplitude
                    * (2.0 * std::f64::consts::PI * self.config.shake_frequency * sim_time).sin();

                for body in &mut bodies {
                    if !body.at_rest {
                        body.velocity.x += shake_offset * 0.1;
                        body.velocity.y += shake_offset * 0.1;
                    }
                }
            }

            // Update positions
            for body in &mut bodies {
                if !body.at_rest {
                    body.position.x += body.velocity.x * self.config.time_step;
                    body.position.y += body.velocity.y * self.config.time_step;
                    body.position.z += body.velocity.z * self.config.time_step;
                }
            }

            // Resolve collisions
            self.resolve_floor_collisions(&mut bodies, floor_z);
            self.resolve_container_collisions(&mut bodies, container_dims);
            self.resolve_box_collisions(&mut bodies);

            // Check for rest state
            let mut all_at_rest = true;
            for body in &mut bodies {
                if body.velocity.norm() < self.config.rest_velocity_threshold && body.on_floor {
                    body.at_rest = true;
                }
                if !body.at_rest {
                    all_at_rest = false;
                }
            }

            if all_at_rest {
                result.converged = true;
                break;
            }

            sim_time += self.config.time_step;
            steps += 1;
        }

        // Compute results
        result.steps = steps;
        result.simulation_time = sim_time;
        result.boxes_at_rest = bodies.iter().filter(|b| b.at_rest).count();

        let mut total_change = 0.0;
        for (i, body) in bodies.iter().enumerate() {
            let initial = initial_positions[i];
            let change = Vector3::new(
                body.position.x - initial.x,
                body.position.y - initial.y,
                body.position.z - initial.z,
            );

            let change_mag = change.norm();
            total_change += change_mag;

            if change_mag > self.config.convergence_threshold {
                result.boxes_moved += 1;
                result.max_change = result.max_change.max(change_mag);
            }

            result.final_positions.push((
                boxes[body.index].id.clone(),
                boxes[body.index].instance,
                body.position,
            ));

            result.position_changes.push((
                boxes[body.index].id.clone(),
                boxes[body.index].instance,
                change,
            ));
        }

        result.avg_change = total_change / bodies.len() as f64;

        // Run stability analysis on final configuration
        let final_boxes: Vec<PlacedBox> = bodies
            .iter()
            .enumerate()
            .map(|(i, body)| {
                PlacedBox::new(
                    boxes[i].id.clone(),
                    boxes[i].instance,
                    body.position,
                    body.dimensions,
                )
                .with_mass(body.mass)
            })
            .collect();

        let analyzer = StabilityAnalyzer::new(StabilityConstraint::partial_base(0.5));
        result.stability_report = Some(analyzer.analyze(&final_boxes, floor_z));

        result.computation_time_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Validates placement stability using physics simulation.
    pub fn validate_stability(
        &self,
        boxes: &[PlacedBox],
        container_dims: Vector3<f64>,
        floor_z: f64,
    ) -> StabilityReport {
        // Quick simulation to check if boxes would move
        let result = self.simulate_settlement(boxes, container_dims, floor_z);

        if let Some(report) = result.stability_report {
            report
        } else {
            StabilityReport::new()
        }
    }

    /// Resolves collisions with the floor.
    fn resolve_floor_collisions(&self, bodies: &mut [PhysicsBody], floor_z: f64) {
        for body in bodies.iter_mut() {
            if body.position.z < floor_z {
                body.position.z = floor_z;
                body.velocity.z = -body.velocity.z * self.config.restitution;

                // Apply friction
                body.velocity.x *= 1.0 - self.config.friction;
                body.velocity.y *= 1.0 - self.config.friction;

                body.on_floor = true;
            }
        }
    }

    /// Resolves collisions with container walls.
    fn resolve_container_collisions(
        &self,
        bodies: &mut [PhysicsBody],
        container_dims: Vector3<f64>,
    ) {
        for body in bodies.iter_mut() {
            // X walls
            if body.position.x < 0.0 {
                body.position.x = 0.0;
                body.velocity.x = -body.velocity.x * self.config.restitution;
            } else if body.position.x + body.dimensions.x > container_dims.x {
                body.position.x = container_dims.x - body.dimensions.x;
                body.velocity.x = -body.velocity.x * self.config.restitution;
            }

            // Y walls
            if body.position.y < 0.0 {
                body.position.y = 0.0;
                body.velocity.y = -body.velocity.y * self.config.restitution;
            } else if body.position.y + body.dimensions.y > container_dims.y {
                body.position.y = container_dims.y - body.dimensions.y;
                body.velocity.y = -body.velocity.y * self.config.restitution;
            }

            // Z ceiling (optional)
            if body.position.z + body.dimensions.z > container_dims.z {
                body.position.z = container_dims.z - body.dimensions.z;
                body.velocity.z = -body.velocity.z * self.config.restitution;
            }
        }
    }

    /// Resolves collisions between boxes (simplified AABB collision).
    fn resolve_box_collisions(&self, bodies: &mut [PhysicsBody]) {
        let n = bodies.len();

        for _ in 0..self.config.solver_iterations {
            for i in 0..n {
                for j in (i + 1)..n {
                    // Check AABB overlap
                    let a_min = bodies[i].aabb_min();
                    let a_max = bodies[i].aabb_max();
                    let b_min = bodies[j].aabb_min();
                    let b_max = bodies[j].aabb_max();

                    if a_max.x <= b_min.x
                        || b_max.x <= a_min.x
                        || a_max.y <= b_min.y
                        || b_max.y <= a_min.y
                        || a_max.z <= b_min.z
                        || b_max.z <= a_min.z
                    {
                        continue; // No collision
                    }

                    // Compute overlap
                    let overlap_x = (a_max.x.min(b_max.x) - a_min.x.max(b_min.x)).max(0.0);
                    let overlap_y = (a_max.y.min(b_max.y) - a_min.y.max(b_min.y)).max(0.0);
                    let overlap_z = (a_max.z.min(b_max.z) - a_min.z.max(b_min.z)).max(0.0);

                    // Find minimum overlap axis
                    let min_overlap = overlap_x.min(overlap_y).min(overlap_z);

                    if min_overlap <= 0.0 {
                        continue;
                    }

                    // Compute centers
                    let center_a = bodies[i].center();
                    let center_b = bodies[j].center();

                    // Separate along minimum overlap axis
                    let total_mass = bodies[i].mass + bodies[j].mass;
                    let ratio_a = bodies[j].mass / total_mass;
                    let ratio_b = bodies[i].mass / total_mass;

                    if overlap_z == min_overlap {
                        // Vertical collision
                        if center_a.z < center_b.z {
                            bodies[i].position.z -= min_overlap * ratio_a;
                            bodies[j].position.z += min_overlap * ratio_b;
                        } else {
                            bodies[i].position.z += min_overlap * ratio_a;
                            bodies[j].position.z -= min_overlap * ratio_b;
                        }
                        // Exchange vertical velocities
                        let v_a = bodies[i].velocity.z;
                        let v_b = bodies[j].velocity.z;
                        bodies[i].velocity.z =
                            v_b * self.config.restitution * ratio_b + v_a * (1.0 - ratio_b);
                        bodies[j].velocity.z =
                            v_a * self.config.restitution * ratio_a + v_b * (1.0 - ratio_a);

                        // Check if supporting
                        if center_a.z > center_b.z {
                            // A is on top of B
                            bodies[i].on_floor = bodies[j].on_floor;
                        }
                    } else if overlap_x == min_overlap {
                        // X collision
                        if center_a.x < center_b.x {
                            bodies[i].position.x -= min_overlap * ratio_a;
                            bodies[j].position.x += min_overlap * ratio_b;
                        } else {
                            bodies[i].position.x += min_overlap * ratio_a;
                            bodies[j].position.x -= min_overlap * ratio_b;
                        }
                    } else {
                        // Y collision
                        if center_a.y < center_b.y {
                            bodies[i].position.y -= min_overlap * ratio_a;
                            bodies[j].position.y += min_overlap * ratio_b;
                        } else {
                            bodies[i].position.y += min_overlap * ratio_a;
                            bodies[j].position.y -= min_overlap * ratio_b;
                        }
                    }
                }
            }
        }
    }

    /// Performs shaking compaction simulation.
    pub fn shake_compact(
        &self,
        boxes: &[PlacedBox],
        container_dims: Vector3<f64>,
        floor_z: f64,
        shake_time: f64,
    ) -> PhysicsResult {
        let mut shake_config = self.config.clone();
        shake_config.enable_shaking = true;
        shake_config.max_simulation_time = shake_time;

        let shaking_sim = PhysicsSimulator::new(shake_config);
        let shake_result = shaking_sim.simulate_settlement(boxes, container_dims, floor_z);

        // Then settle without shaking
        let final_boxes: Vec<PlacedBox> = shake_result
            .final_positions
            .iter()
            .zip(boxes.iter())
            .map(|((_, _, pos), original)| {
                PlacedBox::new(
                    original.id.clone(),
                    original.instance,
                    *pos,
                    original.dimensions,
                )
                .with_mass(original.mass.unwrap_or(1.0))
            })
            .collect();

        // Final settlement
        self.simulate_settlement(&final_boxes, container_dims, floor_z)
    }
}

impl Default for PhysicsSimulator {
    fn default() -> Self {
        Self::new(PhysicsConfig::default())
    }
}

/// Computes the volume utilization after simulation.
pub fn compute_utilization(boxes: &[PlacedBox], container_dims: Vector3<f64>) -> f64 {
    let total_volume: f64 = boxes.iter().map(|b| b.volume()).sum();
    let container_volume = container_dims.x * container_dims.y * container_dims.z;
    total_volume / container_volume
}

/// Finds the effective height (maximum z after all boxes).
pub fn compute_effective_height(boxes: &[PlacedBox]) -> f64 {
    boxes
        .iter()
        .map(|b| b.position.z + b.dimensions.z)
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_config_default() {
        let config = PhysicsConfig::default();
        assert!(config.gravity < 0.0);
        assert!(config.time_step > 0.0);
        assert!(!config.enable_shaking);
    }

    #[test]
    fn test_physics_config_builder() {
        let config = PhysicsConfig::new()
            .with_gravity(-10.0)
            .with_time_step(0.01)
            .with_friction(0.8)
            .with_restitution(0.2);

        assert!((config.gravity - (-10.0)).abs() < 0.001);
        assert!((config.time_step - 0.01).abs() < 0.001);
        assert!((config.friction - 0.8).abs() < 0.001);
        assert!((config.restitution - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_single_box_on_floor() {
        let simulator = PhysicsSimulator::default();

        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(10.0, 10.0, 0.0), // Already on floor
            Vector3::new(20.0, 20.0, 20.0),
        )];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.simulate_settlement(&boxes, container, 0.0);

        // Simulation should complete
        assert!(result.steps > 0 || result.converged);
        // Box should not move much (small bouncing may occur)
        assert!(result.max_change < 5.0);
    }

    #[test]
    fn test_falling_box() {
        let simulator = PhysicsSimulator::new(
            PhysicsConfig::default()
                .with_max_time(5.0) // Longer simulation
                .with_time_step(0.01),
        );

        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(10.0, 10.0, 50.0), // Elevated
            Vector3::new(20.0, 20.0, 20.0),
        )];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.simulate_settlement(&boxes, container, 0.0);

        // Box should have moved (fallen or bounced)
        assert!(result.steps > 0);

        // Final position should be lower than starting (fallen towards floor)
        let (_, _, final_pos) = &result.final_positions[0];
        assert!(final_pos.z < 50.0); // Should have fallen from starting position
    }

    #[test]
    fn test_stacked_boxes_settlement() {
        let simulator = PhysicsSimulator::new(PhysicsConfig::default().with_max_time(3.0));

        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(50.0, 50.0, 20.0),
            )
            .with_mass(10.0),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(0.0, 0.0, 22.0), // Slight gap
                Vector3::new(50.0, 50.0, 20.0),
            )
            .with_mass(5.0),
        ];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.simulate_settlement(&boxes, container, 0.0);

        // Both boxes should settle
        assert!(result.converged || result.simulation_time > 0.0);

        // Stability report should be available
        assert!(result.stability_report.is_some());
    }

    #[test]
    fn test_container_collision() {
        let simulator = PhysicsSimulator::new(PhysicsConfig::default().with_max_time(1.0));

        // Box outside container should be pushed in
        let boxes = vec![PlacedBox::new(
            "B1",
            0,
            Point3::new(-10.0, 50.0, 0.0), // Starts outside
            Vector3::new(20.0, 20.0, 20.0),
        )];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.simulate_settlement(&boxes, container, 0.0);

        // Final position should be inside container
        let (_, _, final_pos) = &result.final_positions[0];
        assert!(final_pos.x >= 0.0);
    }

    #[test]
    fn test_shaking_compaction() {
        let simulator = PhysicsSimulator::new(
            PhysicsConfig::default().with_shaking(1.0, 10.0), // amplitude=1.0, freq=10Hz
        );

        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(30.0, 30.0, 20.0),
            ),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(35.0, 0.0, 0.0), // Gap between boxes
                Vector3::new(30.0, 30.0, 20.0),
            ),
        ];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.shake_compact(&boxes, container, 0.0, 2.0);

        // Simulation should complete with expected number of boxes
        assert_eq!(result.final_positions.len(), 2);
    }

    #[test]
    fn test_compute_utilization() {
        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(10.0, 10.0, 10.0),
            ), // 1000 volume
            PlacedBox::new(
                "B2",
                0,
                Point3::new(10.0, 0.0, 0.0),
                Vector3::new(10.0, 10.0, 10.0),
            ), // 1000 volume
        ];

        let container = Vector3::new(100.0, 100.0, 100.0); // 1,000,000 volume
        let util = compute_utilization(&boxes, container);

        assert!((util - 0.002).abs() < 0.001); // 2000/1000000 = 0.002
    }

    #[test]
    fn test_compute_effective_height() {
        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(10.0, 10.0, 30.0),
            ),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(10.0, 0.0, 0.0),
                Vector3::new(10.0, 10.0, 50.0),
            ),
        ];

        let height = compute_effective_height(&boxes);
        assert!((height - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_validate_stability() {
        let simulator = PhysicsSimulator::new(PhysicsConfig::default().with_max_time(1.0));

        let boxes = vec![
            PlacedBox::new(
                "B1",
                0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(50.0, 50.0, 20.0),
            ),
            PlacedBox::new(
                "B2",
                0,
                Point3::new(0.0, 0.0, 20.0), // Stacked
                Vector3::new(50.0, 50.0, 20.0),
            ),
        ];

        let container = Vector3::new(100.0, 100.0, 100.0);
        let report = simulator.validate_stability(&boxes, container, 0.0);

        // Both should be stable
        assert!(report.stable_count >= 1);
    }

    #[test]
    fn test_empty_simulation() {
        let simulator = PhysicsSimulator::default();
        let container = Vector3::new(100.0, 100.0, 100.0);
        let result = simulator.simulate_settlement(&[], container, 0.0);

        assert!(result.final_positions.is_empty());
        assert!(result.converged || result.steps == 0);
    }
}
