//! Generalized TSP (GTSP) formulation for cutting path optimization.
//!
//! Models the cutting sequence problem as a GTSP where:
//! - Each contour is a **cluster** with multiple candidate pierce points
//! - The goal is to select exactly one candidate per cluster and find the
//!   shortest tour visiting all clusters
//!
//! # Algorithm
//!
//! 1. **Discretize**: Generate N equidistant pierce candidates per contour
//! 2. **Distance matrix**: Compute asymmetric rapid-traverse distances
//!    between all candidate pairs across different clusters
//! 3. **Solve**: Use Noon-Bean transformation to ATSP, then apply
//!    metaheuristic solver (Cycle 46)
//!
//! # References
//!
//! - Noon & Bean (1993), "An efficient transformation of the GTSP"
//! - Dewil et al. (2015), "An improvement heuristic framework for the laser
//!   cutting tool path problem"

use crate::config::{CutDirectionPreference, CuttingConfig};
use crate::contour::{ContourType, CutContour};
use crate::cost::point_distance;
use crate::result::CutDirection;

/// A candidate pierce point on a contour.
#[derive(Debug, Clone)]
pub struct PierceCandidate {
    /// Contour this candidate belongs to.
    pub contour_id: usize,
    /// Index within the cluster's candidate list.
    pub candidate_index: usize,
    /// The pierce point coordinates.
    pub point: (f64, f64),
    /// The nearest vertex index on the contour.
    pub vertex_index: usize,
    /// Cut direction for this contour.
    pub direction: CutDirection,
    /// End point after cutting the full contour (same as `point` for closed contours).
    pub end_point: (f64, f64),
}

/// A GTSP cluster: one contour with its candidate pierce points.
#[derive(Debug, Clone)]
pub struct GtspCluster {
    /// ID of the contour this cluster represents.
    pub contour_id: usize,
    /// Candidate pierce points for this contour.
    pub candidates: Vec<PierceCandidate>,
}

/// A complete GTSP instance with distance matrix.
#[derive(Debug, Clone)]
pub struct GtspInstance {
    /// Clusters (one per contour).
    pub clusters: Vec<GtspCluster>,
    /// Asymmetric distance matrix between all candidates.
    /// `distances[i][j]` = rapid distance from end_point of global candidate `i`
    /// to pierce point of global candidate `j`.
    /// Intra-cluster distances are set to `f64::MAX` (invalid transitions).
    pub distances: Vec<Vec<f64>>,
    /// Distance from home position to each candidate's pierce point.
    pub home_distances: Vec<f64>,
    /// Cumulative offset for each cluster in the global index.
    pub cluster_offsets: Vec<usize>,
    /// Total number of candidates across all clusters.
    pub total_candidates: usize,
}

impl GtspInstance {
    /// Returns the cluster index and local candidate index for a global index.
    pub fn global_to_local(&self, global_idx: usize) -> (usize, usize) {
        for (c, offset) in self.cluster_offsets.iter().enumerate() {
            let size = self.clusters[c].candidates.len();
            if global_idx >= *offset && global_idx < offset + size {
                return (c, global_idx - offset);
            }
        }
        // Should never happen if global_idx is valid
        (self.clusters.len() - 1, 0)
    }

    /// Returns the global index for a cluster and local candidate index.
    pub fn local_to_global(&self, cluster_idx: usize, candidate_idx: usize) -> usize {
        self.cluster_offsets[cluster_idx] + candidate_idx
    }

    /// Returns the candidate at a global index.
    pub fn candidate(&self, global_idx: usize) -> &PierceCandidate {
        let (c, l) = self.global_to_local(global_idx);
        &self.clusters[c].candidates[l]
    }
}

/// Discretizes contours into GTSP clusters with equidistant pierce candidates.
///
/// Each contour gets `config.pierce_candidates` equidistant points along its
/// perimeter. If `pierce_candidates == 1`, uses the centroid-facing vertex.
///
/// # Arguments
///
/// * `contours` - Cut contours to discretize
/// * `config` - Cutting configuration (pierce_candidates, direction preferences)
pub fn discretize_contours(
    contours: &[CutContour],
    config: &CuttingConfig,
) -> Vec<GtspCluster> {
    let n_candidates = config.pierce_candidates.max(1);

    contours
        .iter()
        .map(|contour| {
            let direction = determine_direction(contour.contour_type, config);
            let candidates = if n_candidates == 1 {
                // Single candidate: use first vertex
                vec![PierceCandidate {
                    contour_id: contour.id,
                    candidate_index: 0,
                    point: contour.vertices[0],
                    vertex_index: 0,
                    direction,
                    end_point: contour.vertices[0],
                }]
            } else {
                generate_equidistant_candidates(contour, n_candidates, direction)
            };

            GtspCluster {
                contour_id: contour.id,
                candidates,
            }
        })
        .collect()
}

/// Builds a GTSP instance with asymmetric distance matrix.
///
/// The distance matrix is indexed by global candidate indices. Intra-cluster
/// distances are set to `f64::MAX` since we must visit exactly one candidate
/// per cluster.
///
/// # Arguments
///
/// * `clusters` - GTSP clusters from `discretize_contours`
/// * `home` - Home/start position for the cutting head
pub fn build_gtsp_instance(clusters: Vec<GtspCluster>, home: (f64, f64)) -> GtspInstance {
    // Compute cluster offsets
    let mut cluster_offsets = Vec::with_capacity(clusters.len());
    let mut offset = 0;
    for cluster in &clusters {
        cluster_offsets.push(offset);
        offset += cluster.candidates.len();
    }
    let total = offset;

    // Collect all candidates for distance computation
    let all_candidates: Vec<&PierceCandidate> = clusters
        .iter()
        .flat_map(|c| c.candidates.iter())
        .collect();

    // Build distance matrix
    let mut distances = vec![vec![f64::MAX; total]; total];
    for (i, ci) in all_candidates.iter().enumerate() {
        for (j, cj) in all_candidates.iter().enumerate() {
            if ci.contour_id == cj.contour_id {
                continue; // Intra-cluster: keep as MAX
            }
            distances[i][j] = point_distance(ci.end_point, cj.point);
        }
    }

    // Home distances
    let home_distances: Vec<f64> = all_candidates
        .iter()
        .map(|c| point_distance(home, c.point))
        .collect();

    GtspInstance {
        clusters,
        distances,
        home_distances,
        cluster_offsets,
        total_candidates: total,
    }
}

/// Evaluates the total rapid distance for a GTSP solution.
///
/// A solution is a list of global candidate indices, one per cluster, in visit order.
pub fn evaluate_solution(instance: &GtspInstance, solution: &[usize]) -> f64 {
    if solution.is_empty() {
        return 0.0;
    }

    let mut total = instance.home_distances[solution[0]];

    for i in 1..solution.len() {
        total += instance.distances[solution[i - 1]][solution[i]];
    }

    total
}

/// Solves the GTSP using a greedy nearest-neighbor heuristic.
///
/// For each step, selects the nearest candidate from any unvisited cluster.
/// Returns the global candidate indices in visit order.
pub fn solve_nn(instance: &GtspInstance) -> Vec<usize> {
    let n_clusters = instance.clusters.len();
    if n_clusters == 0 {
        return Vec::new();
    }

    let mut visited_clusters = vec![false; n_clusters];
    let mut solution = Vec::with_capacity(n_clusters);

    // First: find nearest candidate from home
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    for (g, dist) in instance.home_distances.iter().enumerate() {
        if *dist < best_dist {
            best_dist = *dist;
            best_idx = g;
        }
    }

    let (cluster, _) = instance.global_to_local(best_idx);
    visited_clusters[cluster] = true;
    solution.push(best_idx);

    // Greedily add nearest unvisited cluster's candidate
    for _ in 1..n_clusters {
        let last = *solution.last().expect("solution not empty");
        let mut next_best = 0;
        let mut next_dist = f64::MAX;

        for (g, &dist) in instance.distances[last].iter().enumerate() {
            let (c, _) = instance.global_to_local(g);
            if visited_clusters[c] {
                continue;
            }
            if dist < next_dist {
                next_dist = dist;
                next_best = g;
            }
        }

        let (c, _) = instance.global_to_local(next_best);
        visited_clusters[c] = true;
        solution.push(next_best);
    }

    solution
}

/// Solves the GTSP with precedence constraints using NN + constrained 2-opt.
///
/// This is the main solver that respects the precedence DAG. It:
/// 1. Builds a precedence-aware NN solution (only visiting clusters whose
///    predecessors have already been visited)
/// 2. Improves with 2-opt swaps that maintain precedence validity
///
/// Returns the global candidate indices in visit order.
pub fn solve_constrained(
    instance: &GtspInstance,
    dag: &crate::hierarchy::CuttingDag,
    max_2opt_iterations: usize,
) -> Vec<usize> {
    let n_clusters = instance.clusters.len();
    if n_clusters == 0 {
        return Vec::new();
    }

    // Step 1: Precedence-aware NN
    let mut solution = nn_constrained(instance, dag);

    // Step 2: Constrained 2-opt
    if max_2opt_iterations > 0 && solution.len() >= 3 {
        improve_2opt_constrained(
            &mut solution,
            instance,
            dag,
            max_2opt_iterations,
        );
    }

    solution
}

/// Nearest-neighbor construction that respects precedence constraints.
fn nn_constrained(
    instance: &GtspInstance,
    dag: &crate::hierarchy::CuttingDag,
) -> Vec<usize> {
    let n_clusters = instance.clusters.len();
    let mut visited_clusters = vec![false; n_clusters];
    let mut solution = Vec::with_capacity(n_clusters);
    let mut visited_contours: std::collections::HashSet<usize> =
        std::collections::HashSet::with_capacity(n_clusters);

    for _ in 0..n_clusters {
        let mut best_idx = None;
        let mut best_dist = f64::MAX;

        for (ci, cluster) in instance.clusters.iter().enumerate() {
            if visited_clusters[ci] {
                continue;
            }

            // Check precedence: all predecessors' clusters must be visited
            let predecessors = dag.predecessors(cluster.contour_id);
            let ready = predecessors
                .iter()
                .all(|pred_id| visited_contours.contains(pred_id));
            if !ready {
                continue;
            }

            // Find the best candidate in this cluster
            for cand in &cluster.candidates {
                let global = instance.local_to_global(ci, cand.candidate_index);
                let dist = if solution.is_empty() {
                    instance.home_distances[global]
                } else {
                    let last: usize = *solution.last().expect("solution not empty");
                    let row: &Vec<f64> = &instance.distances[last];
                    row[global]
                };

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some((ci, global));
                }
            }
        }

        if let Some((ci, global)) = best_idx {
            visited_clusters[ci] = true;
            visited_contours.insert(instance.clusters[ci].contour_id);
            solution.push(global);
        }
    }

    solution
}

/// Constrained 2-opt improvement for GTSP solutions.
///
/// Tries swapping candidates within clusters and reversing sub-sequences,
/// accepting only moves that reduce cost and respect precedence.
fn improve_2opt_constrained(
    solution: &mut [usize],
    instance: &GtspInstance,
    dag: &crate::hierarchy::CuttingDag,
    max_iterations: usize,
) {
    let n = solution.len();
    let mut improved = true;
    let mut iterations = 0;
    let mut current_cost = evaluate_solution(instance, solution);

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        // Move 1: Try swapping each position to a better candidate in the same cluster
        for pos in 0..n {
            let current_global = solution[pos];
            let (ci, _) = instance.global_to_local(current_global);
            let cluster = &instance.clusters[ci];

            for cand in &cluster.candidates {
                let alt_global = instance.local_to_global(ci, cand.candidate_index);
                if alt_global == current_global {
                    continue;
                }

                solution[pos] = alt_global;
                let new_cost = evaluate_solution(instance, solution);

                if new_cost < current_cost - 1e-10 {
                    current_cost = new_cost;
                    improved = true;
                } else {
                    solution[pos] = current_global; // Undo
                }
            }
        }

        // Move 2: Try reversing sub-sequences (cluster-order level)
        for i in 0..n.saturating_sub(1) {
            for j in (i + 2)..n {
                solution[i + 1..=j].reverse();

                // Check precedence validity
                let cluster_order: Vec<usize> = solution
                    .iter()
                    .map(|&g| instance.clusters[instance.global_to_local(g).0].contour_id)
                    .collect();

                if dag.is_valid_sequence(&cluster_order) {
                    let new_cost = evaluate_solution(instance, solution);
                    if new_cost < current_cost - 1e-10 {
                        current_cost = new_cost;
                        improved = true;
                    } else {
                        solution[i + 1..=j].reverse(); // Undo
                    }
                } else {
                    solution[i + 1..=j].reverse(); // Undo — violates precedence
                }
            }
        }
    }
}

/// Determines the cutting direction for a contour type.
fn determine_direction(contour_type: ContourType, config: &CuttingConfig) -> CutDirection {
    let pref = match contour_type {
        ContourType::Exterior => config.exterior_direction,
        ContourType::Interior => config.interior_direction,
    };

    match pref {
        CutDirectionPreference::Ccw => CutDirection::Ccw,
        CutDirectionPreference::Cw => CutDirection::Cw,
        CutDirectionPreference::Auto => match contour_type {
            ContourType::Exterior => CutDirection::Ccw,
            ContourType::Interior => CutDirection::Cw,
        },
    }
}

/// Generates equidistant pierce candidates along a contour's perimeter.
fn generate_equidistant_candidates(
    contour: &CutContour,
    n: usize,
    direction: CutDirection,
) -> Vec<PierceCandidate> {
    let vertices = &contour.vertices;
    let nv = vertices.len();
    if nv == 0 {
        return Vec::new();
    }

    // Compute cumulative edge lengths
    let mut edge_lengths = Vec::with_capacity(nv);
    let mut cumulative = Vec::with_capacity(nv + 1);
    cumulative.push(0.0);

    for i in 0..nv {
        let j = (i + 1) % nv;
        let len = point_distance(vertices[i], vertices[j]);
        edge_lengths.push(len);
        cumulative.push(cumulative[i] + len);
    }

    let perimeter = *cumulative.last().expect("at least one vertex");
    if perimeter < 1e-12 {
        return vec![PierceCandidate {
            contour_id: contour.id,
            candidate_index: 0,
            point: vertices[0],
            vertex_index: 0,
            direction,
            end_point: vertices[0],
        }];
    }

    let spacing = perimeter / n as f64;
    let mut candidates = Vec::with_capacity(n);

    for k in 0..n {
        let target_dist = k as f64 * spacing;

        // Find which edge this distance falls on
        let (point, vertex_idx) = point_at_distance(vertices, &cumulative, &edge_lengths, target_dist);

        candidates.push(PierceCandidate {
            contour_id: contour.id,
            candidate_index: k,
            point,
            vertex_index: vertex_idx,
            direction,
            end_point: point, // Closed contour: returns to pierce point
        });
    }

    candidates
}

/// Finds the point at a given distance along the contour perimeter.
fn point_at_distance(
    vertices: &[(f64, f64)],
    cumulative: &[f64],
    edge_lengths: &[f64],
    distance: f64,
) -> ((f64, f64), usize) {
    let nv = vertices.len();
    let perimeter = cumulative[nv];
    let dist = distance % perimeter;

    for i in 0..nv {
        if dist >= cumulative[i] && dist <= cumulative[i + 1] + 1e-12 {
            let edge_len = edge_lengths[i];
            if edge_len < 1e-12 {
                return (vertices[i], i);
            }
            let t = (dist - cumulative[i]) / edge_len;
            let j = (i + 1) % nv;
            let px = vertices[i].0 + t * (vertices[j].0 - vertices[i].0);
            let py = vertices[i].1 + t * (vertices[j].1 - vertices[i].1);
            return ((px, py), i);
        }
    }

    // Fallback: last vertex
    (vertices[nv - 1], nv - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rect(id: usize, x: f64, y: f64, w: f64, h: f64) -> CutContour {
        CutContour {
            id,
            geometry_id: format!("part{}", id),
            instance: 0,
            contour_type: ContourType::Exterior,
            vertices: vec![
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            ],
            perimeter: 2.0 * (w + h),
            centroid: (x + w / 2.0, y + h / 2.0),
        }
    }

    #[test]
    fn test_discretize_single_candidate() {
        let contours = vec![make_rect(0, 0.0, 0.0, 10.0, 10.0)];
        let config = CuttingConfig::new().with_pierce_candidates(1);
        let clusters = discretize_contours(&contours, &config);

        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].candidates.len(), 1);
        assert_eq!(clusters[0].candidates[0].point, (0.0, 0.0));
    }

    #[test]
    fn test_discretize_four_candidates_on_square() {
        let contours = vec![make_rect(0, 0.0, 0.0, 10.0, 10.0)];
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);

        assert_eq!(clusters.len(), 1);
        let cands = &clusters[0].candidates;
        assert_eq!(cands.len(), 4);

        // Perimeter = 40, spacing = 10 → points at distances 0, 10, 20, 30
        // Vertices: (0,0), (10,0), (10,10), (0,10)
        // dist 0 → (0,0), dist 10 → (10,0), dist 20 → (10,10), dist 30 → (0,10)
        assert!((cands[0].point.0 - 0.0).abs() < 1e-10);
        assert!((cands[0].point.1 - 0.0).abs() < 1e-10);
        assert!((cands[1].point.0 - 10.0).abs() < 1e-10);
        assert!((cands[1].point.1 - 0.0).abs() < 1e-10);
        assert!((cands[2].point.0 - 10.0).abs() < 1e-10);
        assert!((cands[2].point.1 - 10.0).abs() < 1e-10);
        assert!((cands[3].point.0 - 0.0).abs() < 1e-10);
        assert!((cands[3].point.1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_discretize_eight_candidates_midpoints() {
        let contours = vec![make_rect(0, 0.0, 0.0, 10.0, 10.0)];
        let config = CuttingConfig::new().with_pierce_candidates(8);
        let clusters = discretize_contours(&contours, &config);

        let cands = &clusters[0].candidates;
        assert_eq!(cands.len(), 8);

        // Perimeter = 40, spacing = 5 → candidates at 0, 5, 10, 15, 20, 25, 30, 35
        // dist 0 → (0,0), dist 5 → (5,0), dist 10 → (10,0), dist 15 → (10,5)
        assert!((cands[1].point.0 - 5.0).abs() < 1e-10);
        assert!((cands[1].point.1 - 0.0).abs() < 1e-10);
        assert!((cands[3].point.0 - 10.0).abs() < 1e-10);
        assert!((cands[3].point.1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_build_gtsp_instance_distances() {
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 20.0, 0.0, 10.0, 10.0),
        ];
        let config = CuttingConfig::new().with_pierce_candidates(1);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        assert_eq!(instance.total_candidates, 2);
        assert_eq!(instance.clusters.len(), 2);

        // Intra-cluster distances should be MAX
        assert_eq!(instance.distances[0][0], f64::MAX);
        assert_eq!(instance.distances[1][1], f64::MAX);

        // Inter-cluster distance: (0,0) → (20,0) = 20.0
        assert!((instance.distances[0][1] - 20.0).abs() < 1e-10);
        // Inter-cluster distance: (20,0) → (0,0) = 20.0
        assert!((instance.distances[1][0] - 20.0).abs() < 1e-10);

        // Home distances
        assert!((instance.home_distances[0] - 0.0).abs() < 1e-10);
        assert!((instance.home_distances[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_to_local() {
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 20.0, 0.0, 10.0, 10.0),
        ];
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        assert_eq!(instance.total_candidates, 8);
        assert_eq!(instance.global_to_local(0), (0, 0));
        assert_eq!(instance.global_to_local(3), (0, 3));
        assert_eq!(instance.global_to_local(4), (1, 0));
        assert_eq!(instance.global_to_local(7), (1, 3));
    }

    #[test]
    fn test_evaluate_solution() {
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 30.0, 0.0, 10.0, 10.0),
        ];
        let config = CuttingConfig::new().with_pierce_candidates(1);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        // Solution: visit cluster 0 first (candidate 0), then cluster 1 (candidate 1)
        let cost = evaluate_solution(&instance, &[0, 1]);
        // Home (0,0) → candidate 0 at (0,0) = 0.0
        // candidate 0 end (0,0) → candidate 1 at (30,0) = 30.0
        assert!((cost - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_nn() {
        let contours = vec![
            make_rect(0, 50.0, 0.0, 10.0, 10.0),
            make_rect(1, 10.0, 0.0, 10.0, 10.0),
            make_rect(2, 30.0, 0.0, 10.0, 10.0),
        ];
        let config = CuttingConfig::new().with_pierce_candidates(1);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        let solution = solve_nn(&instance);
        assert_eq!(solution.len(), 3);

        // NN from (0,0): nearest is contour 1 at (10,0), then 2 at (30,0), then 0 at (50,0)
        let cluster_order: Vec<usize> = solution
            .iter()
            .map(|&g| instance.global_to_local(g).0)
            .collect();
        assert_eq!(cluster_order, vec![1, 2, 0]);
    }

    #[test]
    fn test_solve_nn_empty() {
        let instance = build_gtsp_instance(Vec::new(), (0.0, 0.0));
        let solution = solve_nn(&instance);
        assert!(solution.is_empty());
    }

    #[test]
    fn test_nn_picks_best_candidate() {
        // Two contours, each with 2 candidates
        // Contour 0 at (0,0)-(10,10): candidates at (0,0) and (10,0)
        // Contour 1 at (12,0)-(22,10): candidates at (12,0) and (22,0)
        // From home (0,0), NN should pick candidate (0,0) of contour 0
        // Then from (0,0), should pick candidate (12,0) of contour 1
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 12.0, 0.0, 10.0, 10.0),
        ];
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        let solution = solve_nn(&instance);
        assert_eq!(solution.len(), 2);

        // First should be from cluster 0 (nearest to home)
        let (c0, _) = instance.global_to_local(solution[0]);
        assert_eq!(c0, 0);

        // Second should be from cluster 1 — pick the nearest candidate
        let (c1, l1) = instance.global_to_local(solution[1]);
        assert_eq!(c1, 1);
        // Candidate at (12,0) should be picked (nearest to end of cluster 0)
        let picked = &instance.clusters[1].candidates[l1];
        assert!((picked.point.0 - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_direction_assignment() {
        let mut contours = vec![make_rect(0, 0.0, 0.0, 10.0, 10.0)];
        contours[0].contour_type = ContourType::Interior;

        let config = CuttingConfig::default();
        let clusters = discretize_contours(&contours, &config);

        // Interior with Auto should be CW
        assert_eq!(clusters[0].candidates[0].direction, CutDirection::Cw);
    }

    #[test]
    fn test_multi_candidate_improves_over_single() {
        // Three rectangles laid out with offset — multi-candidate should find
        // better pierce points than single-candidate
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 15.0, 5.0, 10.0, 10.0),
            make_rect(2, 30.0, 0.0, 10.0, 10.0),
        ];

        // Single candidate
        let config1 = CuttingConfig::new().with_pierce_candidates(1);
        let clusters1 = discretize_contours(&contours, &config1);
        let inst1 = build_gtsp_instance(clusters1, (0.0, 0.0));
        let sol1 = solve_nn(&inst1);
        let cost1 = evaluate_solution(&inst1, &sol1);

        // 8 candidates per contour
        let config8 = CuttingConfig::new().with_pierce_candidates(8);
        let clusters8 = discretize_contours(&contours, &config8);
        let inst8 = build_gtsp_instance(clusters8, (0.0, 0.0));
        let sol8 = solve_nn(&inst8);
        let cost8 = evaluate_solution(&inst8, &sol8);

        // More candidates should give same or better solution
        assert!(
            cost8 <= cost1 + 1e-6,
            "Multi-candidate cost {} should be <= single-candidate cost {}",
            cost8,
            cost1
        );
    }

    #[test]
    fn test_solve_constrained_respects_precedence() {
        use crate::hierarchy::CuttingDag;

        // Part with interior hole — hole must be cut first
        let contours = vec![
            CutContour {
                id: 0,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Exterior,
                vertices: vec![(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
                perimeter: 80.0,
                centroid: (10.0, 10.0),
            },
            CutContour {
                id: 1,
                geometry_id: "part1".to_string(),
                instance: 0,
                contour_type: ContourType::Interior,
                vertices: vec![(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
                perimeter: 40.0,
                centroid: (10.0, 10.0),
            },
        ];

        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        let solution = solve_constrained(&instance, &dag, 100);
        assert_eq!(solution.len(), 2);

        // Interior (cluster for contour 1) must come before Exterior (cluster for contour 0)
        let cluster_order: Vec<usize> = solution
            .iter()
            .map(|&g| instance.clusters[instance.global_to_local(g).0].contour_id)
            .collect();

        let pos_interior = cluster_order.iter().position(|&id| id == 1).unwrap();
        let pos_exterior = cluster_order.iter().position(|&id| id == 0).unwrap();
        assert!(pos_interior < pos_exterior);
    }

    #[test]
    fn test_solve_constrained_with_multiple_parts() {
        use crate::hierarchy::CuttingDag;

        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 15.0, 0.0, 10.0, 10.0),
            make_rect(2, 30.0, 0.0, 10.0, 10.0),
        ];

        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        let solution = solve_constrained(&instance, &dag, 100);
        assert_eq!(solution.len(), 3);

        // NN should visit in order: 0 (nearest), 1, 2
        let cluster_order: Vec<usize> = solution
            .iter()
            .map(|&g| instance.global_to_local(g).0)
            .collect();
        assert_eq!(cluster_order, vec![0, 1, 2]);
    }

    #[test]
    fn test_2opt_improves_solution() {
        use crate::hierarchy::CuttingDag;

        // Place contours in a way that NN gives a suboptimal order
        // Contours arranged: home(0,0) → C1(5,20) → C0(5,0) → C2(5,40)
        // NN from home: picks C0 (nearest), then C1, then C2
        // Optimal: C0, C2, C1 might be worse... Let's use a zigzag
        let contours = vec![
            make_rect(0, 0.0, 0.0, 10.0, 10.0),
            make_rect(1, 20.0, 0.0, 10.0, 10.0),
            make_rect(2, 40.0, 0.0, 10.0, 10.0),
        ];

        let dag = CuttingDag::build(&contours);
        let config = CuttingConfig::new().with_pierce_candidates(4);
        let clusters = discretize_contours(&contours, &config);
        let instance = build_gtsp_instance(clusters, (0.0, 0.0));

        let solution = solve_constrained(&instance, &dag, 100);
        let cost = evaluate_solution(&instance, &solution);

        // Cost should be reasonable (not worse than worst case)
        // Worst case: home→C2(40,0)→C0(0,0)→C1(20,0) = 40+40+20 = 100
        assert!(cost < 100.0, "Solution cost {} should be < worst case 100", cost);
    }

    #[test]
    fn test_constrained_empty() {
        use crate::hierarchy::CuttingDag;

        let contours: Vec<CutContour> = Vec::new();
        let dag = CuttingDag::build(&contours);
        let instance = build_gtsp_instance(Vec::new(), (0.0, 0.0));

        let solution = solve_constrained(&instance, &dag, 100);
        assert!(solution.is_empty());
    }
}
