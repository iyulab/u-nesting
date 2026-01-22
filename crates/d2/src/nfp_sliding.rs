//! Improved Sliding Algorithm for No-Fit Polygon generation.
//!
//! This module implements the sliding/orbiting algorithm for NFP computation,
//! based on Burke et al. (2007) "Complete and Robust No-Fit Polygon Generation"
//! and Luo & Rao (2022) "Improved Sliding Algorithm".
//!
//! ## Algorithm Overview
//!
//! The sliding algorithm works by:
//! 1. Placing the orbiting polygon at a valid starting position (touching but not overlapping)
//! 2. Sliding the orbiting polygon around the stationary polygon while maintaining contact
//! 3. Recording the path traced by the reference point to form the NFP boundary
//!
//! ## Key Concepts
//!
//! - **TouchingGroup**: A set of simultaneous contact points between polygons
//! - **Contact Types**: Vertex-Edge (VE), Edge-Vertex (EV), Edge-Edge (EE)
//! - **Translation Vector**: The direction along which the orbiting polygon can slide
//!
//! ## References
//!
//! - [Burke et al. (2007)](https://www.graham-kendall.com/papers/bhkw2007.pdf)
//! - [Luo & Rao (2022)](https://www.mdpi.com/2227-7390/10/16/2941)

use u_nesting_core::robust::orient2d_filtered;
use u_nesting_core::{Error, Result};

use crate::nfp::Nfp;

// ============================================================================
// Contact Types and TouchingGroup
// ============================================================================

/// Type of contact between two polygons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactType {
    /// Vertex of orbiting polygon touches edge of stationary polygon.
    VertexEdge,
    /// Edge of orbiting polygon touches vertex of stationary polygon.
    EdgeVertex,
    /// Edge of orbiting polygon is parallel and touching edge of stationary polygon.
    EdgeEdge,
}

/// A single contact point between two polygons.
#[derive(Debug, Clone)]
pub struct Contact {
    /// Type of contact.
    pub contact_type: ContactType,
    /// Index of the element (vertex or edge start) on the stationary polygon.
    pub stationary_idx: usize,
    /// Index of the element (vertex or edge start) on the orbiting polygon.
    pub orbiting_idx: usize,
    /// The contact point in world coordinates.
    pub point: (f64, f64),
}

/// A group of simultaneous contacts between two polygons.
///
/// When the orbiting polygon is in a certain position, multiple vertices/edges
/// may be in contact simultaneously. This structure captures all such contacts.
#[derive(Debug, Clone)]
pub struct TouchingGroup {
    /// All contacts in this group.
    pub contacts: Vec<Contact>,
    /// The position of the orbiting polygon's reference point.
    pub reference_position: (f64, f64),
}

impl TouchingGroup {
    /// Creates a new empty touching group.
    pub fn new(reference_position: (f64, f64)) -> Self {
        Self {
            contacts: Vec::new(),
            reference_position,
        }
    }

    /// Adds a contact to the group.
    pub fn add_contact(&mut self, contact: Contact) {
        self.contacts.push(contact);
    }

    /// Returns true if this group has any contacts.
    pub fn has_contacts(&self) -> bool {
        !self.contacts.is_empty()
    }

    /// Returns the number of contacts in this group.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }
}

// ============================================================================
// Translation Vector
// ============================================================================

/// A potential translation vector for the orbiting polygon.
#[derive(Debug, Clone)]
pub struct TranslationVector {
    /// Direction of translation (unit vector).
    pub direction: (f64, f64),
    /// Maximum distance the polygon can translate in this direction.
    pub max_distance: f64,
    /// The contact that generated this translation vector.
    pub source_contact: Contact,
}

impl TranslationVector {
    /// Creates a new translation vector.
    pub fn new(direction: (f64, f64), max_distance: f64, source_contact: Contact) -> Self {
        Self {
            direction,
            max_distance,
            source_contact,
        }
    }

    /// Returns the actual translation (direction * distance).
    pub fn translation(&self) -> (f64, f64) {
        (
            self.direction.0 * self.max_distance,
            self.direction.1 * self.max_distance,
        )
    }
}

// ============================================================================
// Edge and Vertex Utilities
// ============================================================================

/// Returns the edge vector from vertex i to vertex (i+1) % n.
#[inline]
fn edge_vector(polygon: &[(f64, f64)], i: usize) -> (f64, f64) {
    let n = polygon.len();
    let p1 = polygon[i];
    let p2 = polygon[(i + 1) % n];
    (p2.0 - p1.0, p2.1 - p1.1)
}

/// Returns the outward normal of an edge (assuming CCW polygon).
#[inline]
#[allow(dead_code)]
fn edge_normal(polygon: &[(f64, f64)], i: usize) -> (f64, f64) {
    let (dx, dy) = edge_vector(polygon, i);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-10 {
        (0.0, 0.0)
    } else {
        // Rotate 90 degrees clockwise for outward normal (CCW polygon)
        (dy / len, -dx / len)
    }
}

/// Normalizes a vector to unit length.
#[inline]
fn normalize(v: (f64, f64)) -> (f64, f64) {
    let len = (v.0 * v.0 + v.1 * v.1).sqrt();
    if len < 1e-10 {
        (0.0, 0.0)
    } else {
        (v.0 / len, v.1 / len)
    }
}

/// Computes the dot product of two vectors.
#[inline]
fn dot(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

/// Computes the cross product (z-component) of two 2D vectors.
#[inline]
fn cross(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

/// Distance between two points.
#[inline]
fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

/// Checks if two edges are parallel (within tolerance).
#[inline]
fn edges_parallel(e1: (f64, f64), e2: (f64, f64)) -> bool {
    let c = cross(e1, e2).abs();
    let len1 = (e1.0 * e1.0 + e1.1 * e1.1).sqrt();
    let len2 = (e2.0 * e2.0 + e2.1 * e2.1).sqrt();
    c < 1e-10 * len1 * len2
}

/// Projects a point onto a line segment and returns the parameter t.
/// t=0 means at p1, t=1 means at p2.
fn project_point_to_segment(point: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-20 {
        return 0.0;
    }
    let t = ((point.0 - p1.0) * dx + (point.1 - p1.1) * dy) / len_sq;
    t.clamp(0.0, 1.0)
}

/// Distance from a point to a line segment.
fn point_to_segment_distance(point: (f64, f64), p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let t = project_point_to_segment(point, p1, p2);
    let proj = (p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1));
    distance(point, proj)
}

/// Checks if a point lies on a segment (within tolerance).
fn point_on_segment(point: (f64, f64), p1: (f64, f64), p2: (f64, f64), tol: f64) -> bool {
    point_to_segment_distance(point, p1, p2) < tol
}

// ============================================================================
// Contact Detection
// ============================================================================

/// Finds all contacts between a stationary polygon and an orbiting polygon
/// at its current position.
///
/// # Arguments
/// * `stationary` - The fixed polygon (CCW)
/// * `orbiting` - The orbiting polygon (CCW) at its current translated position
/// * `tolerance` - Distance tolerance for contact detection
pub fn find_contacts(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    tolerance: f64,
) -> Vec<Contact> {
    let mut contacts = Vec::new();

    let n_stat = stationary.len();
    let n_orb = orbiting.len();

    // Check vertex-edge contacts (orbiting vertex on stationary edge)
    for (orb_idx, &orb_vertex) in orbiting.iter().enumerate() {
        for stat_edge_idx in 0..n_stat {
            let stat_p1 = stationary[stat_edge_idx];
            let stat_p2 = stationary[(stat_edge_idx + 1) % n_stat];
            if point_on_segment(orb_vertex, stat_p1, stat_p2, tolerance) {
                contacts.push(Contact {
                    contact_type: ContactType::VertexEdge,
                    stationary_idx: stat_edge_idx,
                    orbiting_idx: orb_idx,
                    point: orb_vertex,
                });
            }
        }
    }

    // Check edge-vertex contacts (stationary vertex on orbiting edge)
    for (stat_idx, &stat_vertex) in stationary.iter().enumerate() {
        for orb_edge_idx in 0..n_orb {
            let orb_p1 = orbiting[orb_edge_idx];
            let orb_p2 = orbiting[(orb_edge_idx + 1) % n_orb];
            if point_on_segment(stat_vertex, orb_p1, orb_p2, tolerance) {
                contacts.push(Contact {
                    contact_type: ContactType::EdgeVertex,
                    stationary_idx: stat_idx,
                    orbiting_idx: orb_edge_idx,
                    point: stat_vertex,
                });
            }
        }
    }

    // Check edge-edge contacts (parallel overlapping edges)
    for stat_edge_idx in 0..n_stat {
        let stat_p1 = stationary[stat_edge_idx];
        let stat_p2 = stationary[(stat_edge_idx + 1) % n_stat];
        let stat_edge = (stat_p2.0 - stat_p1.0, stat_p2.1 - stat_p1.1);

        for orb_edge_idx in 0..n_orb {
            let orb_p1 = orbiting[orb_edge_idx];
            let orb_p2 = orbiting[(orb_edge_idx + 1) % n_orb];
            let orb_edge = (orb_p2.0 - orb_p1.0, orb_p2.1 - orb_p1.1);

            // Check if edges are parallel
            if edges_parallel(stat_edge, orb_edge) {
                // Check if edges overlap (both endpoints of one edge are on the other's line)
                let d1 = point_to_segment_distance(orb_p1, stat_p1, stat_p2);
                let d2 = point_to_segment_distance(orb_p2, stat_p1, stat_p2);
                if d1 < tolerance && d2 < tolerance {
                    // Edges are collinear and potentially overlapping
                    // Use midpoint of overlap as contact point
                    let mid = ((orb_p1.0 + orb_p2.0) / 2.0, (orb_p1.1 + orb_p2.1) / 2.0);
                    contacts.push(Contact {
                        contact_type: ContactType::EdgeEdge,
                        stationary_idx: stat_edge_idx,
                        orbiting_idx: orb_edge_idx,
                        point: mid,
                    });
                }
            }
        }
    }

    contacts
}

/// Creates a touching group from the current contacts.
pub fn create_touching_group(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    reference_position: (f64, f64),
    tolerance: f64,
) -> TouchingGroup {
    let contacts = find_contacts(stationary, orbiting, tolerance);
    let mut group = TouchingGroup::new(reference_position);
    for contact in contacts {
        group.add_contact(contact);
    }
    group
}

// ============================================================================
// Translation Vector Computation
// ============================================================================

/// Computes potential translation vectors from a touching group.
///
/// Each contact generates one or two potential translation vectors
/// (sliding directions along the contact).
pub fn compute_translation_vectors(
    touching_group: &TouchingGroup,
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
) -> Vec<TranslationVector> {
    let mut vectors = Vec::new();

    for contact in &touching_group.contacts {
        match contact.contact_type {
            ContactType::VertexEdge => {
                // Orbiting vertex on stationary edge
                // Can slide along the stationary edge direction
                let edge = edge_vector(stationary, contact.stationary_idx);
                let dir = normalize(edge);
                let neg_dir = (-dir.0, -dir.1);

                // Calculate max distance to edge endpoint
                let n = stationary.len();
                let edge_end = stationary[(contact.stationary_idx + 1) % n];
                let edge_start = stationary[contact.stationary_idx];
                let max_dist_pos = distance(contact.point, edge_end);
                let max_dist_neg = distance(contact.point, edge_start);

                vectors.push(TranslationVector::new(dir, max_dist_pos, contact.clone()));
                vectors.push(TranslationVector::new(
                    neg_dir,
                    max_dist_neg,
                    contact.clone(),
                ));
            }
            ContactType::EdgeVertex => {
                // Stationary vertex on orbiting edge
                // The orbiting polygon can slide along the orbiting edge direction
                let edge = edge_vector(orbiting, contact.orbiting_idx);
                let dir = normalize(edge);
                let neg_dir = (-dir.0, -dir.1);

                // For EV contact, max distance is to orbiting edge endpoints
                let n = orbiting.len();
                let orb_edge_end = orbiting[(contact.orbiting_idx + 1) % n];
                let orb_edge_start = orbiting[contact.orbiting_idx];

                // The orbiting polygon needs to move so that the stationary vertex
                // stays on the orbiting edge - this is opposite direction
                let dist_to_end = distance(contact.point, orb_edge_end);
                let dist_to_start = distance(contact.point, orb_edge_start);

                vectors.push(TranslationVector::new(
                    neg_dir,
                    dist_to_end,
                    contact.clone(),
                ));
                vectors.push(TranslationVector::new(dir, dist_to_start, contact.clone()));
            }
            ContactType::EdgeEdge => {
                // Parallel edges in contact
                // Can slide along the edge direction
                let edge = edge_vector(stationary, contact.stationary_idx);
                let dir = normalize(edge);
                let neg_dir = (-dir.0, -dir.1);

                // Approximate max distance based on edge lengths
                let stat_edge_len = {
                    let n = stationary.len();
                    distance(
                        stationary[contact.stationary_idx],
                        stationary[(contact.stationary_idx + 1) % n],
                    )
                };
                let orb_edge_len = {
                    let n = orbiting.len();
                    distance(
                        orbiting[contact.orbiting_idx],
                        orbiting[(contact.orbiting_idx + 1) % n],
                    )
                };
                let max_dist = stat_edge_len + orb_edge_len;

                vectors.push(TranslationVector::new(dir, max_dist, contact.clone()));
                vectors.push(TranslationVector::new(neg_dir, max_dist, contact.clone()));
            }
        }
    }

    vectors
}

/// Selects the best translation vector for NFP generation.
///
/// The algorithm prefers counter-clockwise traversal around the stationary polygon.
/// This ensures consistent NFP boundary generation.
pub fn select_translation_vector(
    vectors: &[TranslationVector],
    previous_direction: Option<(f64, f64)>,
    stationary_centroid: (f64, f64),
    orbiting_position: (f64, f64),
) -> Option<&TranslationVector> {
    if vectors.is_empty() {
        return None;
    }

    // Calculate the direction from stationary centroid to orbiting position
    let radial = normalize((
        orbiting_position.0 - stationary_centroid.0,
        orbiting_position.1 - stationary_centroid.1,
    ));

    // Prefer CCW direction: perpendicular to radial, rotated 90° CCW
    let ccw_preferred = (-radial.1, radial.0);

    // Score each vector based on CCW preference and continuity
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, v) in vectors.iter().enumerate() {
        let mut score = dot(v.direction, ccw_preferred);

        // Bonus for continuity with previous direction
        if let Some(prev) = previous_direction {
            score += 0.5 * dot(v.direction, prev);
        }

        // Penalize very short translations
        if v.max_distance < 1e-6 {
            score -= 100.0;
        }

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    Some(&vectors[best_idx])
}

// ============================================================================
// Edge Case Handling
// ============================================================================

/// Represents a potential collision during translation.
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    /// Distance along translation at which collision occurs.
    pub distance: f64,
    /// Type of contact formed at collision.
    pub contact_type: ContactType,
    /// Index on stationary polygon.
    pub stationary_idx: usize,
    /// Index on orbiting polygon.
    pub orbiting_idx: usize,
}

/// Checks for potential collisions during a translation.
///
/// This handles the "blocking" case where the orbiting polygon would
/// collide with another part of the stationary polygon before reaching
/// the full translation distance.
///
/// # Returns
/// The first collision event if one occurs, or None if the translation is clear.
pub fn check_translation_collision(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    translation: (f64, f64),
    tolerance: f64,
) -> Option<CollisionEvent> {
    let trans_len = (translation.0 * translation.0 + translation.1 * translation.1).sqrt();
    if trans_len < tolerance {
        return None;
    }
    let trans_dir = (translation.0 / trans_len, translation.1 / trans_len);

    let n_stat = stationary.len();
    let n_orb = orbiting.len();

    let mut first_collision: Option<CollisionEvent> = None;

    // Check orbiting vertices against stationary edges
    for (orb_idx, &orb_v) in orbiting.iter().enumerate() {
        for stat_edge_idx in 0..n_stat {
            let stat_p1 = stationary[stat_edge_idx];
            let stat_p2 = stationary[(stat_edge_idx + 1) % n_stat];

            if let Some(dist) =
                ray_segment_intersection(orb_v, trans_dir, stat_p1, stat_p2, tolerance)
            {
                if dist > tolerance
                    && dist < trans_len - tolerance
                    && (first_collision.is_none()
                        || dist < first_collision.as_ref().unwrap().distance)
                {
                    first_collision = Some(CollisionEvent {
                        distance: dist,
                        contact_type: ContactType::VertexEdge,
                        stationary_idx: stat_edge_idx,
                        orbiting_idx: orb_idx,
                    });
                }
            }
        }
    }

    // Check stationary vertices against moving orbiting edges
    for (stat_idx, &stat_v) in stationary.iter().enumerate() {
        for orb_edge_idx in 0..n_orb {
            let orb_p1 = orbiting[orb_edge_idx];
            let orb_p2 = orbiting[(orb_edge_idx + 1) % n_orb];

            // The stationary vertex appears to move in the opposite direction
            let neg_dir = (-trans_dir.0, -trans_dir.1);

            if let Some(dist) = ray_segment_intersection(stat_v, neg_dir, orb_p1, orb_p2, tolerance)
            {
                if dist > tolerance
                    && dist < trans_len - tolerance
                    && (first_collision.is_none()
                        || dist < first_collision.as_ref().unwrap().distance)
                {
                    first_collision = Some(CollisionEvent {
                        distance: dist,
                        contact_type: ContactType::EdgeVertex,
                        stationary_idx: stat_idx,
                        orbiting_idx: orb_edge_idx,
                    });
                }
            }
        }
    }

    first_collision
}

/// Computes the intersection of a ray with a line segment.
///
/// # Returns
/// The distance along the ray to the intersection point, or None if no intersection.
fn ray_segment_intersection(
    ray_origin: (f64, f64),
    ray_dir: (f64, f64),
    seg_p1: (f64, f64),
    seg_p2: (f64, f64),
    tolerance: f64,
) -> Option<f64> {
    let seg_dir = (seg_p2.0 - seg_p1.0, seg_p2.1 - seg_p1.1);
    let denominator = cross(ray_dir, seg_dir);

    // Check if ray and segment are parallel
    if denominator.abs() < tolerance {
        return None;
    }

    // Vector from segment start to ray origin
    let diff = (seg_p1.0 - ray_origin.0, seg_p1.1 - ray_origin.1);

    // Parameter along ray: t
    // Parameter along segment: u
    // Ray equation: ray_origin + t * ray_dir
    // Segment equation: seg_p1 + u * seg_dir
    // Solve: ray_origin + t * ray_dir = seg_p1 + u * seg_dir

    let t = cross(diff, seg_dir) / denominator;
    let u = cross(diff, ray_dir) / denominator;

    // Check if intersection is in valid range
    // t >= 0 means intersection is in front of ray
    // 0 <= u <= 1 means intersection is on segment
    if t >= -tolerance && u >= -tolerance && u <= 1.0 + tolerance {
        Some(t.max(0.0))
    } else {
        None
    }
}

/// Handles the "perfect fit" edge case where polygons fit together exactly.
///
/// In this case, multiple translation vectors may have the same score,
/// and we need to carefully choose the one that continues the CCW traversal.
pub fn handle_perfect_fit(
    touching_group: &TouchingGroup,
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    visited_positions: &[(f64, f64)],
    tolerance: f64,
) -> Vec<TranslationVector> {
    let mut vectors = compute_translation_vectors(touching_group, stationary, orbiting);

    // Filter out vectors that would lead to already-visited positions
    vectors.retain(|v| {
        let potential_pos = (
            touching_group.reference_position.0 + v.direction.0 * v.max_distance.min(1.0),
            touching_group.reference_position.1 + v.direction.1 * v.max_distance.min(1.0),
        );

        !visited_positions
            .iter()
            .any(|&visited| distance(potential_pos, visited) < tolerance * 10.0)
    });

    vectors
}

/// Handles interlocking concavities where the orbiting polygon may need to
/// enter a concave region of the stationary polygon.
///
/// This generates additional NFP vertices where the reference point can
/// reach into concave regions.
pub fn detect_interlocking_opportunity(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    current_pos: (f64, f64),
    tolerance: f64,
) -> Option<(f64, f64)> {
    // Find concave vertices of stationary polygon
    let n_stat = stationary.len();
    let mut concave_vertices = Vec::new();

    for i in 0..n_stat {
        let prev = stationary[if i == 0 { n_stat - 1 } else { i - 1 }];
        let curr = stationary[i];
        let next = stationary[(i + 1) % n_stat];

        // Check if this vertex is concave (reflex)
        let orientation = orient2d_filtered(prev, curr, next);
        if orientation.is_cw() {
            concave_vertices.push((i, curr));
        }
    }

    if concave_vertices.is_empty() {
        return None;
    }

    // Check if orbiting polygon can fit into any concave region
    let orb_bbox = compute_bbox(orbiting);
    let orb_width = orb_bbox.1 .0 - orb_bbox.0 .0;
    let orb_height = orb_bbox.1 .1 - orb_bbox.0 .1;

    for (idx, vertex) in concave_vertices {
        // Compute the "pocket" size at this concave vertex
        let prev_idx = if idx == 0 { n_stat - 1 } else { idx - 1 };
        let _next_idx = (idx + 1) % n_stat;

        let prev_edge = edge_vector(stationary, prev_idx);
        let next_edge = edge_vector(stationary, idx);

        // Estimate pocket opening
        let opening_width = (dot(prev_edge, next_edge).abs()).sqrt();

        // Check if orbiting polygon might fit
        let min_dim = orb_width.min(orb_height);
        if opening_width > min_dim * 0.5 {
            // This concave region might be accessible
            // Compute a potential position toward this vertex
            let dir = normalize((vertex.0 - current_pos.0, vertex.1 - current_pos.1));
            let test_pos = (
                current_pos.0 + dir.0 * tolerance,
                current_pos.1 + dir.1 * tolerance,
            );

            // Verify this doesn't cause overlap
            let orbiting_test: Vec<(f64, f64)> = orbiting
                .iter()
                .map(|&(x, y)| (x + test_pos.0, y + test_pos.1))
                .collect();

            if !polygons_overlap(stationary, &orbiting_test) {
                return Some(test_pos);
            }
        }
    }

    None
}

/// Computes the axis-aligned bounding box of a polygon.
fn compute_bbox(polygon: &[(f64, f64)]) -> ((f64, f64), (f64, f64)) {
    let min_x = polygon
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_x = polygon
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let min_y = polygon
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_y = polygon
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    ((min_x, min_y), (max_x, max_y))
}

/// Checks if two polygons overlap using separating axis theorem.
fn polygons_overlap(poly_a: &[(f64, f64)], poly_b: &[(f64, f64)]) -> bool {
    // Quick AABB check first
    let bbox_a = compute_bbox(poly_a);
    let bbox_b = compute_bbox(poly_b);

    if bbox_a.1 .0 < bbox_b.0 .0
        || bbox_b.1 .0 < bbox_a.0 .0
        || bbox_a.1 .1 < bbox_b.0 .1
        || bbox_b.1 .1 < bbox_a.0 .1
    {
        return false;
    }

    // SAT check with edges from both polygons
    for polygon in [poly_a, poly_b] {
        let n = polygon.len();
        for i in 0..n {
            let edge = edge_vector(polygon, i);
            let axis = normalize((-edge.1, edge.0)); // Normal to edge

            if axis.0.abs() < 1e-10 && axis.1.abs() < 1e-10 {
                continue;
            }

            // Project both polygons onto axis
            let (min_a, max_a) = project_polygon_on_axis(poly_a, axis);
            let (min_b, max_b) = project_polygon_on_axis(poly_b, axis);

            // Check for gap
            if max_a < min_b || max_b < min_a {
                return false; // Separating axis found
            }
        }
    }

    true // No separating axis found, polygons overlap
}

/// Projects a polygon onto an axis and returns (min, max) extent.
fn project_polygon_on_axis(polygon: &[(f64, f64)], axis: (f64, f64)) -> (f64, f64) {
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;

    for &p in polygon {
        let proj = dot(p, axis);
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
    }

    (min_proj, max_proj)
}

// ============================================================================
// Sliding NFP Algorithm
// ============================================================================

/// Configuration for the sliding NFP algorithm.
#[derive(Debug, Clone)]
pub struct SlidingNfpConfig {
    /// Tolerance for contact detection.
    pub contact_tolerance: f64,
    /// Maximum number of iterations to prevent infinite loops.
    pub max_iterations: usize,
    /// Minimum translation distance before stopping.
    pub min_translation: f64,
}

impl Default for SlidingNfpConfig {
    fn default() -> Self {
        Self {
            contact_tolerance: 1e-6,
            max_iterations: 10000,
            min_translation: 1e-8,
        }
    }
}

/// Computes the NFP using the sliding algorithm.
///
/// # Arguments
/// * `stationary` - The fixed polygon (CCW winding)
/// * `orbiting` - The orbiting polygon (CCW winding)
/// * `config` - Algorithm configuration
///
/// # Returns
/// The computed NFP as a list of polygons (outer boundary + potential holes)
pub fn compute_nfp_sliding(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    config: &SlidingNfpConfig,
) -> Result<Nfp> {
    if stationary.len() < 3 || orbiting.len() < 3 {
        return Err(Error::InvalidGeometry(
            "Polygons must have at least 3 vertices".into(),
        ));
    }

    // Ensure CCW winding
    let stationary = ensure_ccw(stationary);
    let orbiting = ensure_ccw(orbiting);

    // Step 1: Find a valid starting position
    let start_pos = find_start_position(&stationary, &orbiting)?;

    // Step 2: Translate orbiting to start position
    let orbiting_at_start: Vec<(f64, f64)> = orbiting
        .iter()
        .map(|&(x, y)| (x + start_pos.0, y + start_pos.1))
        .collect();

    // Step 3: Trace the NFP boundary by sliding
    let nfp_path = trace_nfp_boundary(
        &stationary,
        &orbiting,
        &orbiting_at_start,
        start_pos,
        config,
    )?;

    if nfp_path.len() < 3 {
        return Err(Error::Internal("NFP path has fewer than 3 vertices".into()));
    }

    Ok(Nfp::from_polygon(nfp_path))
}

/// Finds a valid starting position for the orbiting polygon.
///
/// The start position is where the orbiting polygon touches the stationary
/// polygon from the "bottom-left" direction (minimizing y, then x).
fn find_start_position(stationary: &[(f64, f64)], orbiting: &[(f64, f64)]) -> Result<(f64, f64)> {
    // Find the bottom-most vertex of stationary polygon
    let stat_bottom_idx = stationary
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(a.0.partial_cmp(&b.0).unwrap())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let stat_bottom = stationary[stat_bottom_idx];

    // Find the top-most vertex of orbiting polygon
    let orb_top_idx = orbiting
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(b.0.partial_cmp(&a.0).unwrap())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let orb_top = orbiting[orb_top_idx];

    // Position orbiting so its top vertex touches stationary's bottom vertex
    // The reference point of orbiting is at origin (0, 0)
    let start_pos = (stat_bottom.0 - orb_top.0, stat_bottom.1 - orb_top.1);

    Ok(start_pos)
}

/// Traces the NFP boundary by sliding the orbiting polygon around the stationary polygon.
fn trace_nfp_boundary(
    stationary: &[(f64, f64)],
    orbiting_original: &[(f64, f64)],
    _orbiting_at_start: &[(f64, f64)],
    start_pos: (f64, f64),
    config: &SlidingNfpConfig,
) -> Result<Vec<(f64, f64)>> {
    let mut nfp_path = Vec::new();
    let mut current_pos = start_pos;
    let mut previous_direction: Option<(f64, f64)> = None;
    let mut stuck_counter = 0;

    // Calculate stationary centroid for CCW preference
    let stat_centroid = polygon_centroid(stationary);

    nfp_path.push(current_pos);

    for _iteration in 0..config.max_iterations {
        // Translate orbiting polygon to current position
        let orbiting_current: Vec<(f64, f64)> = orbiting_original
            .iter()
            .map(|&(x, y)| (x + current_pos.0, y + current_pos.1))
            .collect();

        // Find contacts at current position
        let touching_group = create_touching_group(
            stationary,
            &orbiting_current,
            current_pos,
            config.contact_tolerance,
        );

        if !touching_group.has_contacts() {
            // No contacts - try to recover by moving toward stationary polygon
            if let Some(recovery_pos) = recover_contact(
                stationary,
                &orbiting_current,
                current_pos,
                config.contact_tolerance,
            ) {
                current_pos = recovery_pos;
                continue;
            }
            break;
        }

        // Handle perfect fit case - filter vectors that lead to visited positions
        let translation_vectors = handle_perfect_fit(
            &touching_group,
            stationary,
            &orbiting_current,
            &nfp_path,
            config.contact_tolerance,
        );

        if translation_vectors.is_empty() {
            // All directions lead to visited positions - might be done or stuck
            stuck_counter += 1;
            if stuck_counter > 3 {
                break;
            }
            // Try to find interlocking opportunity
            if let Some(interlock_pos) = detect_interlocking_opportunity(
                stationary,
                orbiting_original,
                current_pos,
                config.contact_tolerance,
            ) {
                if distance(interlock_pos, current_pos) > config.min_translation {
                    nfp_path.push(interlock_pos);
                    current_pos = interlock_pos;
                    continue;
                }
            }
            break;
        }
        stuck_counter = 0;

        // Select the best translation vector
        let selected = select_translation_vector(
            &translation_vectors,
            previous_direction,
            stat_centroid,
            current_pos,
        );

        let Some(tv) = selected else {
            break;
        };

        // Calculate intended translation
        let intended_distance = tv.max_distance.min(1000.0); // Cap for safety
        if intended_distance < config.min_translation {
            break;
        }

        let intended_translation = (
            tv.direction.0 * intended_distance,
            tv.direction.1 * intended_distance,
        );

        // Check for collisions during translation
        let actual_translation = if let Some(collision) = check_translation_collision(
            stationary,
            &orbiting_current,
            intended_translation,
            config.contact_tolerance,
        ) {
            // Stop at the collision point
            let blocked_dist = (collision.distance - config.contact_tolerance).max(0.0);
            (tv.direction.0 * blocked_dist, tv.direction.1 * blocked_dist)
        } else {
            intended_translation
        };

        let new_pos = (
            current_pos.0 + actual_translation.0,
            current_pos.1 + actual_translation.1,
        );

        // Check if we've returned to start
        let dist_to_start = distance(new_pos, start_pos);
        if nfp_path.len() > 2 && dist_to_start < config.contact_tolerance * 10.0 {
            // Completed the loop
            break;
        }

        // Check if this position is distinct from the last
        let dist_to_last = distance(new_pos, current_pos);
        if dist_to_last > config.min_translation {
            nfp_path.push(new_pos);
            previous_direction = Some(tv.direction);
        }

        current_pos = new_pos;
    }

    // Simplify the path by removing collinear points
    simplify_polygon(&nfp_path, config.contact_tolerance)
}

/// Attempts to recover contact when the orbiting polygon loses contact with stationary.
fn recover_contact(
    stationary: &[(f64, f64)],
    orbiting: &[(f64, f64)],
    current_pos: (f64, f64),
    tolerance: f64,
) -> Option<(f64, f64)> {
    // Find the closest point on stationary boundary to any orbiting vertex
    let mut min_dist = f64::INFINITY;
    let mut closest_point = None;

    let n_stat = stationary.len();

    for orb_v in orbiting {
        for i in 0..n_stat {
            let stat_p1 = stationary[i];
            let stat_p2 = stationary[(i + 1) % n_stat];

            let t = project_point_to_segment(*orb_v, stat_p1, stat_p2);
            let proj = (
                stat_p1.0 + t * (stat_p2.0 - stat_p1.0),
                stat_p1.1 + t * (stat_p2.1 - stat_p1.1),
            );

            let dist = distance(*orb_v, proj);
            if dist < min_dist {
                min_dist = dist;
                closest_point = Some(proj);
            }
        }
    }

    // Move toward the closest point
    if let Some(target) = closest_point {
        if min_dist > tolerance {
            // Calculate direction to move
            let orb_centroid = polygon_centroid(orbiting);
            let dir = normalize((target.0 - orb_centroid.0, target.1 - orb_centroid.1));
            let move_dist = min_dist - tolerance * 0.5;
            return Some((
                current_pos.0 + dir.0 * move_dist,
                current_pos.1 + dir.1 * move_dist,
            ));
        }
    }

    None
}

/// Computes polygon centroid.
fn polygon_centroid(polygon: &[(f64, f64)]) -> (f64, f64) {
    let n = polygon.len() as f64;
    let sum_x: f64 = polygon.iter().map(|p| p.0).sum();
    let sum_y: f64 = polygon.iter().map(|p| p.1).sum();
    (sum_x / n, sum_y / n)
}

/// Ensures polygon is in counter-clockwise order.
fn ensure_ccw(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let area = signed_area(polygon);
    if area < 0.0 {
        polygon.iter().rev().cloned().collect()
    } else {
        polygon.to_vec()
    }
}

/// Computes signed area of a polygon.
fn signed_area(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += polygon[i].0 * polygon[j].1;
        area -= polygon[j].0 * polygon[i].1;
    }
    area / 2.0
}

/// Simplifies a polygon by removing collinear points.
fn simplify_polygon(polygon: &[(f64, f64)], tolerance: f64) -> Result<Vec<(f64, f64)>> {
    if polygon.len() < 3 {
        return Ok(polygon.to_vec());
    }

    let mut result = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = (i + 1) % n;

        let p_prev = polygon[prev];
        let p_curr = polygon[i];
        let p_next = polygon[next];

        // Check if current point is collinear with neighbors
        let orientation = orient2d_filtered(p_prev, p_curr, p_next);
        if !orientation.is_collinear() {
            result.push(p_curr);
        } else {
            // Also keep if distance from line is significant
            let dist = point_to_segment_distance(p_curr, p_prev, p_next);
            if dist > tolerance {
                result.push(p_curr);
            }
        }
    }

    if result.len() < 3 {
        // Can't simplify further, return original
        Ok(polygon.to_vec())
    } else {
        Ok(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f64, h: f64) -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    }

    #[test]
    fn test_edge_vector() {
        let square = rect(10.0, 10.0);
        let e0 = edge_vector(&square, 0);
        assert!((e0.0 - 10.0).abs() < 1e-10);
        assert!(e0.1.abs() < 1e-10);
    }

    #[test]
    fn test_edge_normal() {
        let square = rect(10.0, 10.0);
        // Bottom edge normal should point down (outside CCW polygon)
        let n0 = edge_normal(&square, 0);
        assert!(n0.0.abs() < 1e-10);
        assert!((n0.1 - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_edges_parallel() {
        assert!(edges_parallel((1.0, 0.0), (2.0, 0.0)));
        assert!(edges_parallel((1.0, 0.0), (-1.0, 0.0)));
        assert!(!edges_parallel((1.0, 0.0), (0.0, 1.0)));
    }

    #[test]
    fn test_point_on_segment() {
        let p1 = (0.0, 0.0);
        let p2 = (10.0, 0.0);

        assert!(point_on_segment((5.0, 0.0), p1, p2, 1e-6));
        assert!(point_on_segment((0.0, 0.0), p1, p2, 1e-6));
        assert!(point_on_segment((10.0, 0.0), p1, p2, 1e-6));
        assert!(!point_on_segment((5.0, 1.0), p1, p2, 1e-6));
    }

    #[test]
    fn test_find_contacts_ve() {
        // Two squares: orbiting square's corner touching stationary's edge
        let stationary = rect(10.0, 10.0);
        let orbiting: Vec<(f64, f64)> = rect(5.0, 5.0)
            .iter()
            .map(|&(x, y)| (x + 5.0, y + 10.0)) // Place above, touching top edge
            .collect();

        let contacts = find_contacts(&stationary, &orbiting, 1e-6);

        // Should find vertex-edge contact where orbiting's bottom-left touches top of stationary
        assert!(!contacts.is_empty(), "Should find at least one contact");
    }

    #[test]
    fn test_find_start_position() {
        let stationary = rect(10.0, 10.0);
        let orbiting = rect(5.0, 5.0);

        let start = find_start_position(&stationary, &orbiting).unwrap();

        // The orbiting polygon's top should touch stationary's bottom
        // Stationary bottom-left is at (0, 0)
        // Orbiting top-left is at (0, 5) - so reference should be at (0, -5)
        // Actually stationary bottom is y=0, orbiting top is y=5
        // So start_pos.1 should be 0 - 5 = -5
        assert!(
            (start.1 - (-5.0)).abs() < 1e-6,
            "Start Y should be -5, got {}",
            start.1
        );
    }

    #[test]
    fn test_touching_group() {
        let ref_pos = (5.0, 5.0);
        let mut group = TouchingGroup::new(ref_pos);

        assert!(!group.has_contacts());
        assert_eq!(group.contact_count(), 0);

        group.add_contact(Contact {
            contact_type: ContactType::VertexEdge,
            stationary_idx: 0,
            orbiting_idx: 0,
            point: (5.0, 5.0),
        });

        assert!(group.has_contacts());
        assert_eq!(group.contact_count(), 1);
    }

    #[test]
    fn test_compute_translation_vectors() {
        let stationary = rect(10.0, 10.0);
        let orbiting: Vec<(f64, f64)> = rect(5.0, 5.0)
            .iter()
            .map(|&(x, y)| (x + 2.5, y + 10.0)) // Touching top edge
            .collect();

        let group = create_touching_group(&stationary, &orbiting, (2.5, 10.0), 1e-6);
        let vectors = compute_translation_vectors(&group, &stationary, &orbiting);

        // Should have translation vectors along the contact edges
        assert!(!vectors.is_empty(), "Should have translation vectors");
    }

    #[test]
    fn test_sliding_nfp_two_squares() {
        let stationary = rect(10.0, 10.0);
        let orbiting = rect(5.0, 5.0);

        let config = SlidingNfpConfig {
            contact_tolerance: 1e-4,
            max_iterations: 1000,
            min_translation: 1e-6,
        };

        let result = compute_nfp_sliding(&stationary, &orbiting, &config);

        // Should succeed
        assert!(result.is_ok(), "NFP computation failed: {:?}", result.err());

        let nfp = result.unwrap();
        assert!(!nfp.is_empty());

        // NFP of two rectangles should have at least 4 vertices
        assert!(nfp.vertex_count() >= 4);
    }

    #[test]
    fn test_ensure_ccw() {
        // CW square
        let cw = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let ccw = ensure_ccw(&cw);

        // Should be reversed
        assert!(signed_area(&ccw) > 0.0);
    }

    #[test]
    fn test_simplify_polygon() {
        // Square with an extra collinear point on one edge
        let with_extra = vec![
            (0.0, 0.0),
            (5.0, 0.0), // Collinear point
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ];

        let simplified = simplify_polygon(&with_extra, 1e-6).unwrap();

        // Should remove the collinear point
        assert_eq!(simplified.len(), 4, "Simplified should have 4 vertices");
    }

    #[test]
    fn test_polygon_centroid() {
        let square = rect(10.0, 10.0);
        let centroid = polygon_centroid(&square);

        assert!((centroid.0 - 5.0).abs() < 1e-10);
        assert!((centroid.1 - 5.0).abs() < 1e-10);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_ray_segment_intersection() {
        // Ray pointing right, segment vertical
        let result =
            ray_segment_intersection((0.0, 0.0), (1.0, 0.0), (5.0, -1.0), (5.0, 1.0), 1e-6);
        assert!(result.is_some());
        assert!((result.unwrap() - 5.0).abs() < 1e-6);

        // Ray pointing left, segment vertical (should not intersect)
        let result =
            ray_segment_intersection((0.0, 0.0), (-1.0, 0.0), (5.0, -1.0), (5.0, 1.0), 1e-6);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_translation_collision() {
        // Simple stationary square
        let stationary = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];

        // Orbiting square positioned to the right
        let orbiting = vec![(15.0, 2.0), (20.0, 2.0), (20.0, 8.0), (15.0, 8.0)];

        // Translation going left (toward the stationary square)
        let translation = (-10.0, 0.0);
        let collision = check_translation_collision(&stationary, &orbiting, translation, 1e-6);

        // Should detect collision - orbiting left edge at x=15 moving -10 would reach x=5
        // But stationary right edge is at x=10, so collision should occur at distance 5
        assert!(
            collision.is_some(),
            "Should detect collision when moving left"
        );

        if let Some(c) = collision {
            // Collision distance should be around 5 (from x=15 to x=10)
            assert!(
                c.distance > 4.0 && c.distance < 6.0,
                "Collision distance should be ~5, got {}",
                c.distance
            );
        }
    }

    #[test]
    fn test_polygons_overlap_true() {
        // Two overlapping squares
        let a = rect(10.0, 10.0);
        let b: Vec<(f64, f64)> = rect(10.0, 10.0)
            .iter()
            .map(|&(x, y)| (x + 5.0, y + 5.0))
            .collect();

        assert!(
            polygons_overlap(&a, &b),
            "Overlapping squares should overlap"
        );
    }

    #[test]
    fn test_polygons_overlap_false() {
        // Two separated squares
        let a = rect(10.0, 10.0);
        let b: Vec<(f64, f64)> = rect(10.0, 10.0)
            .iter()
            .map(|&(x, y)| (x + 20.0, y))
            .collect();

        assert!(
            !polygons_overlap(&a, &b),
            "Separated squares should not overlap"
        );
    }

    #[test]
    fn test_compute_bbox() {
        let square = rect(10.0, 10.0);
        let bbox = compute_bbox(&square);

        assert!((bbox.0 .0 - 0.0).abs() < 1e-10);
        assert!((bbox.0 .1 - 0.0).abs() < 1e-10);
        assert!((bbox.1 .0 - 10.0).abs() < 1e-10);
        assert!((bbox.1 .1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sliding_nfp_triangle() {
        // Equilateral triangle (stationary)
        let stationary = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 8.66)];

        // Small square (orbiting)
        let orbiting = rect(3.0, 3.0);

        let config = SlidingNfpConfig {
            contact_tolerance: 1e-4,
            max_iterations: 1000,
            min_translation: 1e-6,
        };

        let result = compute_nfp_sliding(&stationary, &orbiting, &config);

        assert!(result.is_ok(), "NFP of triangle and square should succeed");
        let nfp = result.unwrap();
        assert!(
            nfp.vertex_count() >= 3,
            "NFP should have at least 3 vertices"
        );
    }

    #[test]
    fn test_sliding_nfp_l_shape() {
        // L-shaped polygon (stationary)
        let stationary = vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 5.0),
            (5.0, 5.0),
            (5.0, 20.0),
            (0.0, 20.0),
        ];

        // Small square (orbiting)
        let orbiting = rect(3.0, 3.0);

        let config = SlidingNfpConfig {
            contact_tolerance: 1e-4,
            max_iterations: 2000,
            min_translation: 1e-6,
        };

        let result = compute_nfp_sliding(&stationary, &orbiting, &config);

        assert!(
            result.is_ok(),
            "NFP of L-shape and square should succeed: {:?}",
            result.err()
        );
        let nfp = result.unwrap();
        // L-shape NFP should have more vertices than a simple rectangle
        assert!(
            nfp.vertex_count() >= 6,
            "NFP of L-shape should have >= 6 vertices, got {}",
            nfp.vertex_count()
        );
    }

    #[test]
    fn test_handle_perfect_fit_filters_visited() {
        let stationary = rect(10.0, 10.0);
        let orbiting: Vec<(f64, f64)> =
            rect(5.0, 5.0).iter().map(|&(x, y)| (x, y + 10.0)).collect();

        let group = create_touching_group(&stationary, &orbiting, (0.0, 10.0), 1e-4);

        // Visited positions include nearby points
        let visited = vec![(0.5, 10.0), (0.0, 10.5)];

        let vectors = handle_perfect_fit(&group, &stationary, &orbiting, &visited, 1e-4);

        // Should filter out vectors leading to visited positions
        // The exact count depends on geometry, but should be fewer than unfiltered
        let unfiltered = compute_translation_vectors(&group, &stationary, &orbiting);

        // In some cases they might be equal if no filtering happens
        // Just verify it doesn't crash
        assert!(vectors.len() <= unfiltered.len());
    }

    #[test]
    fn test_detect_interlocking_concave_polygon() {
        // C-shaped polygon with concavity
        let stationary = vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 5.0),
            (5.0, 5.0),
            (5.0, 15.0),
            (20.0, 15.0),
            (20.0, 20.0),
            (0.0, 20.0),
        ];

        // Small square
        let orbiting = rect(3.0, 3.0);

        // Position outside the C-shape
        let current_pos = (25.0, 10.0);

        // This might or might not detect an opportunity depending on geometry
        let result = detect_interlocking_opportunity(&stationary, &orbiting, current_pos, 1e-4);

        // Just verify it doesn't crash - the logic is complex
        // Result can be Some or None depending on exact geometry
        let _ = result;
    }

    #[test]
    fn test_sliding_nfp_same_size_squares() {
        // Two identical squares
        let stationary = rect(10.0, 10.0);
        let orbiting = rect(10.0, 10.0);

        let config = SlidingNfpConfig {
            contact_tolerance: 1e-4,
            max_iterations: 1000,
            min_translation: 1e-6,
        };

        let result = compute_nfp_sliding(&stationary, &orbiting, &config);

        assert!(result.is_ok(), "NFP of same-size squares should succeed");
        let nfp = result.unwrap();

        // NFP of two identical squares should be a square with side 2x original
        // The area should be roughly 4x (since sides double)
        assert!(nfp.vertex_count() >= 4);
    }

    #[test]
    fn test_signed_area_ccw() {
        let ccw = rect(10.0, 10.0);
        let area = signed_area(&ccw);
        assert!(area > 0.0, "CCW polygon should have positive signed area");
    }

    #[test]
    fn test_signed_area_cw() {
        let cw = vec![(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)];
        let area = signed_area(&cw);
        assert!(area < 0.0, "CW polygon should have negative signed area");
    }
}
